"""Data-loading utilities."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map


class WSIBagDataset(Dataset):
    """Dataset of whole slide image feature bags.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
    ):
        self.feature_paths = feature_paths
        self.labels = np.asarray(labels)

        assert len(labels) == len(feature_paths)

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shape of labels: {labels.shape}")

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.feature_paths[index]
        features: torch.Tensor = torch.load(path)
        assert isinstance(features, torch.Tensor)
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"

        return features, label


class InMemoryWSIBagDataset(Dataset):
    """Dataset of whole slide image feature bags, where all bags are in memory.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    The bag label is a Float32 type.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.labels = np.asarray(labels)
        assert len(labels) == len(feature_paths)

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shape of labels: {labels.shape}")

        print("Loading bags into memory...")
        self.bags: list[torch.Tensor] = thread_map(
            partial(torch.load, map_location="cpu"), self.feature_paths, max_workers=10
        )
        #self.bags: list[torch.Tensor] = thread_map(
        #    torch.load, self.feature_paths, max_workers=10
        #)
        print("Done loading bags into memory.")
        num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
        print(f"    {num_bytes/1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"

        return features, label


class InMemoryWSIBagDatasetClassification(Dataset):
    """Dataset of whole slide image feature bags, where all bags are in memory.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    The bag label is a Long type.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.labels = np.asarray(labels)
        assert len(labels) == len(feature_paths)

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shape of labels: {labels.shape}")

        print("Loading bags into memory...")
        self.bags: list[torch.Tensor] = thread_map(
            partial(torch.load, map_location="cpu"), self.feature_paths, max_workers=10
        )
        #self.bags: list[torch.Tensor] = thread_map(
        #    torch.load, self.feature_paths, max_workers=10
        #)
        print("Done loading bags into memory.")
        num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
        print(f"    {num_bytes/1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 0, f"Expected 0-dim tensor but got {label.ndim}-dim"

        return features, label

class InMemoryWSIBagDatasetClassificationWithRandPerm(InMemoryWSIBagDatasetClassification):
    """Dataset of whole slide image feature bags, where all bags are in memory.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    The bag label is a Long type.

    On each call to __getitem__, randomly permute the instances in the bag.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
        seed: int = 0,
    ):
        super().__init__(feature_paths, labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features[torch.randperm(features.shape[0])]
        features = features.float()
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 0, f"Expected 0-dim tensor but got {label.ndim}-dim"

        return features, label


class InMemoryWSIBagOneHotInputsDataset(Dataset):
    """Dataset of whole slide image feature bags, where all bags are in memory.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        one_hot_inputs: list[npt.NDArray],
        labels: npt.NDArray,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.one_hot_inputs = [np.asarray(arr) for arr in one_hot_inputs]
        self.labels = np.asarray(labels)
        assert len(labels) == len(feature_paths)
        for arr in self.one_hot_inputs:
            assert arr.ndim == 2

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shapes of one hot inputs: {[c.shape for c in self.one_hot_inputs]}")
        print(f"    Shape of labels: {labels.shape}")

        print("Loading bags into memory...")
        self.bags: list[torch.Tensor] = thread_map(
            torch.load, self.feature_paths, max_workers=10
        )
        print("Done loading bags into memory.")
        num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
        print(f"    {num_bytes/1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        one_hot_inputs = [
            torch.tensor(arr[index]).float() for arr in self.one_hot_inputs
        ]
        label = torch.tensor(self.labels[index])
        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"
        return features, one_hot_inputs, label


class InMemoryWSIBagOneHotInputsConcatDataset(Dataset):
    """Dataset of whole slide image feature bags, where all bags are in memory.

    Concatenate the one_hot_inputs with the instance features.

    Each item is a 2D tensor of num_patches x num_features, as well as the bag label.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        one_hot_inputs: list[npt.NDArray],
        labels: npt.NDArray,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.one_hot_inputs = [np.asarray(arr) for arr in one_hot_inputs]
        self.labels = np.asarray(labels)
        assert len(labels) == len(feature_paths)
        for arr in self.one_hot_inputs:
            assert arr.ndim == 2

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shapes of one hot inputs: {[c.shape for c in self.one_hot_inputs]}")
        print(f"    Shape of labels: {labels.shape}")

        print("Loading bags into memory...")
        self.bags: list[torch.Tensor] = thread_map(
            torch.load, self.feature_paths, max_workers=10
        )
        print("Done loading bags into memory.")
        num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
        print(f"    {num_bytes/1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        one_hot_inputs = torch.cat(
            [torch.tensor(arr[index]).float() for arr in self.one_hot_inputs]
        )
        assert one_hot_inputs.ndim == 1, "one_hot_inputs must have 1 dim"
        num_instances = features.shape[0]
        features = torch.cat([features, one_hot_inputs.repeat(num_instances, 1)], dim=1)
        label = torch.tensor(self.labels[index])
        assert features.shape[0] == num_instances
        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"
        return features, label


class WSIBagDatasetConcatCategories(Dataset):
    def __init__(
        self,
        feature_paths: list[Path | str],
        input_labels: np.ndarray,
        labels: np.ndarray,
    ):
        self.feature_paths = feature_paths
        self.input_labels = input_labels
        self.labels = labels

        if (
            input_labels is not None
            and len(feature_paths) != self.input_labels.shape[0]
        ):
            raise ValueError(
                "the number of f0eature paths is not equal to the length of"
                f" input labels {len(feature_paths)} versus"
                f" {self.input_labels.shape[0]}."
            )

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        if self.input_labels is not None:
            print(f"    Shape of input labels: {input_labels.shape}")
        print(f"    Shape of labels: {labels.shape}")

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.feature_paths[index]
        features: torch.Tensor = torch.load(path)
        if self.input_labels is not None:
            input_label = torch.tensor(self.input_labels[index])
            assert input_label.ndim == 1
            if input_label.shape == ():
                input_label = input_label.unsqueeze(0)
        label = torch.tensor(self.labels[index])
        # If the label is a scalar, make it a 1-dim tensor.
        assert label.ndim == 1, f"Got ndim == {label.ndim} but expected 1"
        features = features.float()
        label = label.float()

        # Concatenate the additional input labels to each patch embedding.
        # TODO: are there more powerful ways we can incorporate this information?
        # Perhaps there are ways to do this that learn interactions among the features?
        # Though linear layers will do this too probably.
        if self.input_labels is not None:
            features = torch.cat(
                [features, input_label.repeat(features.shape[0], 1)], dim=1
            )

        return features, label
