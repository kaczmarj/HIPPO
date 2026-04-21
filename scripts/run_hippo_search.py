import json
from pathlib import Path
import math
import pandas as pd
import numpy as np
import torch
import search
import sys
import os
from json import JSONEncoder
import argparse
from models.abmil import AttentionMILModel
from models.vision_transformer import VisionTransformer


device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(description="Run HIPPO high-effect search")
    parser.add_argument(
        "--features_root",
        type=str,
        default=None,
        required=True,
        help="Path to the root directory containing the features."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation.",
    )
    parser.add_argument(
        "--slide_id",
        type=str,
        default="a195bae3-357f-11eb-b1e7-001a7dda7111",
        help="Slide ID to process.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        required=True,
        help="Path to the model root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="abmil",
        choices=["abmil", "vit"],
        help="Type of model to use (abmil or vit).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="minimize",
        choices=["minimize", "maximize", "smallest_difference"],
        help="Type of optimization to perform (minimize, maximize, or smallest_difference).",
    )
    parser.add_argument(
        "--output_index_to_optimize",
        type=int,
        default=1,
        help="Index of the model output to optimize during the search (e.g., 1 for the positive class probability).",
    )
    args = parser.parse_args()
    return args

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError for other types
        return JSONEncoder.default(self, obj)

def main(args):
    features_root = Path(args.features_root)
    model_root = Path(f"{args.model_root}/abmil-uni-128um_seed{args.fold}/")
    torch.set_float32_matmul_precision("medium")

    df = pd.read_csv(model_root / "inputs_test.csv")

    with open(model_root / "results_y_and_yhat_best_model.json") as f:
        d = json.load(f)
    y_true = np.array(d["all_y"])
    assert np.array_equal(df["binary_label_int"], y_true)
    y_prob = d["all_y_hat"]
    y_prob = torch.as_tensor(y_prob).softmax(1)[:, 1]
    y_pred = np.array(d["all_y_hat"]).argmax(1)
    df["y_pred"] = y_pred
    df["y_prob"] = y_prob

    row = df.loc[df["y_prob"].idxmax(), :].copy().to_dict()

    # slide_id = row["slide_id"]
    slide_id = args.slide_id
    features = torch.load(features_root / f"{slide_id}.pt", map_location="cpu").to(device)

    # Load ABMIL model.
    if args.model_type == "abmil":
        model = AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
    elif args.model_type == "vit":
        model = VisionTransformer(num_classes=2, in_features=512, dim_model=512, n_layers=2, n_heads=8, dim_feedforward=512, dropout=0.0)
    else:        
        raise ValueError(f"Invalid model type: {args.model_type}")
        
    model.eval()
    state_dict_path = model_root / "model_best.pt"
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)


    def model_probs_fn(features):
        with torch.inference_mode():
            logits, _ = model(features)
        # Shape of logits is 1xC, where C is number of classes.
        probs = logits.softmax(1).squeeze(0)  # C
        return probs
    
    num_rounds = math.ceil(len(features))
    print("features.shape:", features.shape)
    print("num_rounds:", num_rounds)

    optimizer_mapping = {
        "minimize": search.minimize,
        "maximize": search.maximize,
        "smallest_difference": search.smallest_difference,
    }

    results_single = search.greedy_search(
        features=features,
        model_probs_fn=model_probs_fn,
        num_rounds=num_rounds,
        output_index_to_optimize=args.output_index_to_optimize,
        # We use minimize because we want to minimize the model outputs
        # when the patches are *removed*.
        optimizer=optimizer_mapping.get(args.optimizer, search.minimize),
    )

    output_dir = f"{args.output_dir}/tp/abmil-uni-128um_k{args.fold}/hippok01"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{slide_id}.json"

    with open(output_path, "w") as f:
        json.dump(results_single, f, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    args = parse_args()
    main(args)