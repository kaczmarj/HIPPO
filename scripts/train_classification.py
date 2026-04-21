"""Train an ABMIL model for classification.

The inputs to the model are bags of embedded H&E patches.

The outputs of the model are classification predictions.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import click
import data
import models
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC
from tqdm import tqdm, trange

# We don't want to overwhelm the node when we run multiple jobs at once.
# And we don't seem to get much benefit from more threads anyway.
torch.set_num_threads(4)

BEST_MODEL_NAME = "model_best.pt"
LAST_MODEL_NAME = "model_last.pt"
METADATA_NAME = "metadata.json"


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclasses.dataclass
class State:
    """Container for application state."""

    model: torch.nn.Module
    features_dir: Path
    output_dir: Path
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    case_id_colname: str
    slide_id_colname: str
    label_col: str
    num_classes: int
    embedding_size: int
    fold: int
    num_epochs: int
    seed: int
    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
    device: torch.device
    lr: float
    weight_decay: float
    best_epoch: int = -1
    accumulation_steps: int = 1

    def __post_init__(self):
        # Test our assumptions.
        assert self.features_dir.exists()
        assert isinstance(self.df_train, pd.DataFrame)
        assert isinstance(self.df_val, pd.DataFrame)
        assert isinstance(self.df_test, pd.DataFrame)
        for _df in [self.df_train, self.df_val, self.df_test]:
            assert self.case_id_colname in _df.columns
            assert self.slide_id_colname in _df.columns
            assert self.label_col in _df.columns
        assert isinstance(self.fold, int)
        assert isinstance(self.num_epochs, int)
        assert isinstance(self.num_classes, int)
        assert isinstance(self.seed, int)
        assert isinstance(self.loader_train, DataLoader)
        assert isinstance(self.loader_val, DataLoader)
        assert isinstance(self.loader_test, DataLoader)
        assert isinstance(self.lr, float)
        assert isinstance(self.weight_decay, float)
        assert isinstance(self.accumulation_steps, int)


def get_loader(
    *,
    features_dir: Path,
    label_col: str,
    slide_id_colname: str,
    df: pd.DataFrame,
    shuffle: bool,
):
    paths = [features_dir / f"{slideid}.pt" for slideid in df[slide_id_colname]]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
    labels = df[label_col].to_numpy()
    assert np.issubdtype(labels.dtype, np.int_), "expected int labels"
    # I/O can be a bottleneck, but we have plenty of RAM. We load
    # the entire dataset into memory to avoid I/O slowdown.
    dset = data.InMemoryWSIBagDatasetClassification(feature_paths=paths, labels=labels)
    return DataLoader(dset, batch_size=1, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    *,
    state: State,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict:
    """Train model for a single epoch."""

    state.model.train()
    total_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    auroc_all_fn = AUROC(
        task="multiclass", num_classes=state.num_classes, average="none"
    ).cpu()
    auroc_weighted_fn = AUROC(
        task="multiclass", num_classes=state.num_classes, average="weighted"
    ).cpu()

    x: torch.Tensor
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor

    for idx, (x, y) in tqdm(
        enumerate(state.loader_train),
        total=len(state.loader_train),
        desc="Iteration",
        unit="slides",
        leave=False,
    ):

        # Remove batch dimension introduced by DataLoader.
        x = x.squeeze(0)  # shape (N, in_features)
        # y shape is (1,)

        # Sanity checks.
        assert y.item() >= 0
        assert y.item() < state.num_classes

        # Transfer to GPU (if it's available).
        x = x.to(state.device, non_blocking=True)
        y = y.to(state.device, non_blocking=True)

        # Forward pass through the model.
        y_hat, _ = state.model(H=x)

        # Calculate loss.
        assert y_hat.shape[0] == y.shape[0]
        assert y_hat.shape[0] == 1
        assert y_hat.shape[1] == state.num_classes

        loss = loss_fn(input=y_hat, target=y)

        # Before reduction, we can consider weighting different loss components.
        loss_scaled = loss / state.accumulation_steps
        loss_scaled.backward()

        # Step through the optimizer on accum steps OR on the last sample of the dataset.
        # This takes care of the case where the length of our dataset is not divisible evenly
        # by the accumulation steps.
        if ((idx + 1) % state.accumulation_steps == 0) or (
            idx + 1 == len(state.loader_train)
        ):
            optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

        # Update running loss.
        total_loss += loss.detach().cpu().item()

        # Calculate performance metrics.
        # preds, target
        auroc_all_fn(y_hat.cpu(), y.detach().cpu())
        auroc_weighted_fn(y_hat.cpu(), y.detach().cpu())

    auroc_all = auroc_all_fn.compute().cpu().numpy()  # shape: (num_classes,)
    auroc: float = auroc_weighted_fn.compute().cpu().numpy().item()

    results = {
        "mean_loss": total_loss / len(state.loader_train),
        "auroc": auroc,
        "classwise_auroc": auroc_all,
    }
    return results


@torch.no_grad()
def evaluate(
    *, state: State, loader: DataLoader, load_best: bool = False
) -> tuple[dict, dict]:
    """Evaluate model."""

    if load_best:
        print("Loading best model...")
        sd = torch.load(
            state.output_dir / BEST_MODEL_NAME, map_location=torch.device("cpu")
        )
        state.model.load_state_dict(sd)
        del sd

    state.model.eval()
    total_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    auroc_all_fn = AUROC(
        task="multiclass", num_classes=state.num_classes, average="none"
    ).cpu()
    auroc_weighted_fn = AUROC(
        task="multiclass", num_classes=state.num_classes, average="weighted"
    ).cpu()

    x: torch.Tensor
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor

    # Containers for true and predicted values. We will save these,
    # so we can analyze them after the fact.
    all_y: list[list[float]] = []
    all_y_hat: list[list[float]] = []

    for x, y in tqdm(loader, desc="Iteration", unit="slides", leave=False):
        # Remove batch dimension introduced by DataLoader.
        x = x.squeeze(0)  # shape (N, in_features)
        # y shape is (1,)

        # Sanity checks.
        assert y.item() >= 0
        assert y.item() < state.num_classes

        # Transfer to GPU (if it's available).
        x = x.to(state.device, non_blocking=True)
        y = y.to(state.device, non_blocking=True)

        # Forward pass through the model.
        y_hat, _ = state.model(H=x)

        # Calculate loss.
        assert y_hat.shape[0] == y.shape[0]
        assert y_hat.shape[0] == 1
        assert y_hat.shape[1] == state.num_classes

        all_y.append(y.cpu().numpy().item())
        all_y_hat.append(y_hat.squeeze(0).cpu().numpy().tolist())

        loss = loss_fn(input=y_hat, target=y)

        # Update running loss.
        total_loss += loss.detach().cpu().item()

        # Calculate performance metrics.
        # preds, target
        auroc_all_fn(y_hat.cpu(), y.detach().cpu())
        auroc_weighted_fn(y_hat.cpu(), y.detach().cpu())

    auroc_all: list[float] = (
        auroc_all_fn.compute().cpu().numpy().tolist()
    )  # shape: (num_classes,)
    auroc: float = auroc_weighted_fn.compute().cpu().numpy().item()

    results = {
        "mean_loss": total_loss / len(state.loader_train),
        "auroc": auroc,
        "classwise_auroc": auroc_all,
    }
    return results, {"all_y": all_y, "all_y_hat": all_y_hat}


def train_and_evaluate(state: State):
    """Run training/val loop, and evaluate on test set at the end."""

    all_train_results: list[dict] = []
    all_val_results: list[dict] = []

    optimizer = torch.optim.AdamW(
        state.model.parameters(),
        lr=state.lr,
        betas=(0.9, 0.999),
        weight_decay=state.weight_decay,
    )
    num_iterations_per_epoch = len(state.loader_train)
    num_total_iterations = math.ceil(
        num_iterations_per_epoch * state.num_epochs / state.accumulation_steps
    )
    print("Number of iterations per epoch", num_iterations_per_epoch)
    print("Number of total iterations", num_total_iterations)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_total_iterations
    )

    state.output_dir.mkdir(exist_ok=True, parents=True)

    best_auroc: float = -math.inf

    iterator = trange(state.num_epochs, desc="Epochs", unit="epochs")
    for epoch_num in iterator:
        epoch_results = train_one_epoch(
            state=state, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        epoch_results["epoch"] = epoch_num
        print(f"\nTraining results for epoch {epoch_num}")
        for k, v in epoch_results.items():
            if isinstance(v, float):
                print(f"   {k} = {v:0.4f}")
            else:
                print(f"   {k} =", v)
        print()
        all_train_results.append(epoch_results)
        del epoch_results

        # Evaluate on validation set.
        val_results, _ = evaluate(state=state, loader=state.loader_val, load_best=False)
        print(f"\nValidation results after epoch {epoch_num}")
        val_results["epoch"] = epoch_num
        for k, v in val_results.items():
            if isinstance(v, float):
                print(f"   {k} = {v:0.4f}")
            else:
                print(f"   {k} =", v)
        all_val_results.append(val_results)
        print()

        # Save model if the validation Pearson r improved.
        if val_results["auroc"] > best_auroc:
            best_auroc = val_results["auroc"]
            torch.save(state.model.state_dict(), state.output_dir / BEST_MODEL_NAME)
            state.best_epoch = epoch_num

        del val_results

    # Run final evaluation on test set using last model.
    test_results_last_model, test_outputs_last_model = evaluate(
        state=state, loader=state.loader_test, load_best=False
    )
    with open(state.output_dir / "results_y_and_yhat_last_model.json", "w") as f:
        json.dump(test_outputs_last_model, f)
    print("\nTest set results (last model)")

    for k, v in test_results_last_model.items():
        if isinstance(v, float):
            print(f"   {k} = {v:0.4f}")
        else:
            print(f"   {k} =", v)

    # Run final evaluation on test set using best model.
    test_results_best_model, test_outputs_best_model = evaluate(
        state=state, loader=state.loader_test, load_best=True
    )
    with open(state.output_dir / "results_y_and_yhat_best_model.json", "w") as f:
        json.dump(test_outputs_best_model, f)
    print(
        f"\nTest set results (best model on validation set (epoch {state.best_epoch}))"
    )
    for k, v in test_results_best_model.items():
        if isinstance(v, float):
            print(f"   {k} = {v:0.4f}")
        else:
            print(f"   {k} =", v)

    # Save last epoch.
    torch.save(state.model.state_dict(), state.output_dir / LAST_MODEL_NAME)

    df_train_results = pd.DataFrame(all_train_results).assign(split="train")
    df_val_results = pd.DataFrame(all_val_results).assign(split="val")
    # In a list because there is only one item.
    df_test_results_last_model = pd.DataFrame([test_results_last_model]).assign(
        split="test"
    )
    # In a list because there is only one item.
    df_test_results_best_model = pd.DataFrame([test_results_best_model]).assign(
        split="test"
    )
    # Save results.
    df_train_results.to_csv(state.output_dir / "results_train.csv", index=False)
    df_val_results.to_csv(state.output_dir / "results_val.csv", index=False)
    df_test_results_last_model.to_csv(
        state.output_dir / "results_test_last_model.csv", index=False
    )
    df_test_results_best_model.to_csv(
        state.output_dir / "results_test_best_model.csv", index=False
    )

    # Save input information.
    state.df_train.to_csv(state.output_dir / "inputs_train.csv", index=False)
    state.df_val.to_csv(state.output_dir / "inputs_val.csv", index=False)
    state.df_test.to_csv(state.output_dir / "inputs_test.csv", index=False)


def dataclass_dict_factory(kv_pairs: list[tuple[str, Any]]) -> dict:
    d = {}
    for k, v in kv_pairs:
        if isinstance(v, pd.DataFrame):
            v = str(v.to_csv())
        elif isinstance(v, DataLoader):
            v = f"DataLoader using {v.dataset}"
        else:
            try:
                json.dumps(v)
            except TypeError:
                v = str(v)
        d[k] = v
    return d


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(
        [
            "AttentionMILModel",
            "AttentionMILMultiBranchModel",
            "AdditiveAttentionMILModel",
            "VisionTransformer",
        ]
    ),
    required=True,
)
@click.option(
    "--features-dir", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--csv", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--case-id-colname", type=str, default="STUDYID")
@click.option("--slide-id-colname", type=str, default="SLIDEID")
@click.option("--label-col", type=str, required=True)
@click.option("--num-classes", type=int, required=True)
@click.option("--embedding-size", type=int, required=True)
@click.option(
    "--split-json", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--fold", type=int, required=True)
@click.option("--num-epochs", type=int, default=20)
@click.option("--seed", type=int, default=0)
@click.option("--allow-cpu", is_flag=True, default=False)
@click.option("-L", "L", type=int, default=256)
@click.option("-D", "D", type=int, default=256)
@click.option("--dropout", type=float, default=0.25)
@click.option("--gated-attention/--no-gated-attention", default=True)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=1e-5)
@click.option("--accumulation-steps", type=int, default=1)
def main(
    *,
    model_name: str,
    features_dir: Path,
    output_dir: Path,
    csv: Path,
    case_id_colname: str,
    slide_id_colname: str,
    label_col: str,
    num_classes: int,
    embedding_size: int,
    split_json: Path,
    fold: int,
    num_epochs: int,
    seed: int,
    allow_cpu: bool,
    L: int,
    D: int,
    dropout: float,
    gated_attention: bool,
    lr: float,
    weight_decay: float,
    accumulation_steps: int,
):
    if output_dir.exists():
        if output_dir.joinpath(METADATA_NAME).exists():
            print("Model was already trained! Finishing early...")
            return
        else:
            print("Output directory exists but model was not fully trained.")
            print("DELETING OUTPUT DIRECTORY to start fresh.")
            # This is a dangerous part of code... we should consider removing
            # this....
            shutil.rmtree(output_dir)
            # raise click.BadParameter(f"output directory already exists: {output_dir}")

    if not allow_cpu:
        assert torch.cuda.is_available()

    print(f"Seeding everything with {seed=}")
    seed_everything(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    available_models = {
        "AttentionMILModel": models.abmil.AttentionMILModel,
        "AttentionMILMultiBranchModel": models.abmil.AttentionMILMultiBranchModel,
        "AdditiveAttentionMILModel": models.abmil.AdditiveAttentionMILModel,
        "VisionTransformer": models.vision_transformer.VisionTransformer,
    }
    model_cls = available_models[model_name]
    if model_name != "VisionTransformer":
        model: torch.nn.Module = model_cls(
            in_features=embedding_size,
            L=L,
            D=D,
            K=1,
            num_classes=num_classes,
            dropout=dropout,
            gated_attention=gated_attention,
        )
    else:
        model: torch.nn.Module = model_cls(
            num_classes=num_classes,
            in_features=embedding_size,
            dim_model=512,
            n_layers=2,
            n_heads=8,
            dim_feedforward=512,
            dropout=0.0,
        )
    model.train()
    model.to(device)

    # Convince mypy that model is, in fact, a module and not Unknown.
    assert isinstance(model, torch.nn.Module)

    # Load the K-fold splits.
    with open(split_json, "r") as f:
        splits = json.load(f)
    assert fold >= 0
    assert fold < len(splits["patient_ids"])
    patient_ids_train = splits["patient_ids"][fold]["train"]
    patient_ids_val = splits["patient_ids"][fold]["val"]
    patient_ids_test = splits["patient_ids"][fold]["test"]

    assert not set(patient_ids_train).intersection(set(patient_ids_val))
    assert not set(patient_ids_train).intersection(set(patient_ids_test))
    assert not set(patient_ids_test).intersection(set(patient_ids_val))

    # Load patient labels.
    df = pd.read_csv(csv, dtype={case_id_colname: str, slide_id_colname: str})

    df_train = df.loc[df[case_id_colname].isin(patient_ids_train), :].copy()
    assert not df_train.empty
    df_val = df.loc[df[case_id_colname].isin(patient_ids_val), :].copy()
    assert not df_val.empty
    df_test = df.loc[df[case_id_colname].isin(patient_ids_test), :].copy()
    assert not df_test.empty
    df[label_col].nunique() == num_classes
    del df

    # Panic if any labels are NA.
    for _df in [df_train, df_val, df_test]:
        _labels_na = _df[label_col].isna().any(axis=0)
        if _labels_na.any():  # type: ignore
            raise ValueError(_labels_na)

    loader_train = get_loader(
        features_dir=features_dir,
        label_col=label_col,
        slide_id_colname=slide_id_colname,
        df=df_train,
        shuffle=True,
    )
    loader_val = get_loader(
        features_dir=features_dir,
        label_col=label_col,
        slide_id_colname=slide_id_colname,
        df=df_val,
        shuffle=False,
    )
    loader_test = get_loader(
        features_dir=features_dir,
        label_col=label_col,
        slide_id_colname=slide_id_colname,
        df=df_test,
        shuffle=False,
    )

    state = State(
        model=model,
        features_dir=features_dir,
        output_dir=output_dir,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        case_id_colname=case_id_colname,
        slide_id_colname=slide_id_colname,
        num_classes=num_classes,
        label_col=label_col,
        embedding_size=embedding_size,
        fold=fold,
        num_epochs=num_epochs,
        seed=seed,
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        accumulation_steps=accumulation_steps,
    )

    meta_file = Path(output_dir / METADATA_NAME)

    print("Training...")
    train_and_evaluate(state)

    run_state = dataclasses.asdict(state, dict_factory=dataclass_dict_factory)
    with meta_file.open("w") as f:
        json.dump(run_state, f)


if __name__ == "__main__":
    main()
