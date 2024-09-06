"""HIPPO search methods."""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm


def minimize(current: torch.Tensor, previous=None) -> int:
    if current.ndim != 1:
        raise ValueError(f"`current` must be 1D, got {current.ndim}D")
    return current.argmin().item()  # type: ignore


def maximize(current: torch.Tensor, previous=None) -> int:
    if current.ndim != 1:
        raise ValueError(f"`current` must be 1D, got {current.ndim}D")
    return current.argmax().item()  # type: ignore


def smallest_difference(current: torch.Tensor, previous: float) -> int:
    if current.ndim != 1:
        raise ValueError(f"`current` must be 1D, got {current.ndim}D")
    if not isinstance(previous, float):
        raise TypeError("`previous` must be a float.")
    result = (current - previous).abs().argmin().item()
    return result  # type: ignore


def _validate_model_probs_fn(
    *,
    model_probs_fn: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    output_index_to_test: int | None = None,
) -> None:
    """
    Validate the model probability function and its output.

    This function checks if the provided model probability function returns
    a valid output given the input features. It ensures that the output
    is a 1D torch.Tensor and optionally checks if a specified output index
    is within bounds.

    Parameters
    ----------
    model_probs_fn : callable
        A function that takes a 2D torch.Tensor as input and returns
        a 1D torch.Tensor of probabilities.
    features : torch.Tensor
        A tensor of input features to be passed to the model_probs_fn.
    output_index_to_test : int or None, optional
        If provided, checks if this index is within the bounds of the
        model's output. Default is None.

    Returns
    -------
    None
    """
    with torch.inference_mode():
        output = model_probs_fn(features)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(output)}")
    if output.ndim != 1:
        raise TypeError(f"Expected torch.Tensor but got {type(output)}")
    if output_index_to_test is not None:
        if output_index_to_test >= len(output):
            raise ValueError(
                "class index to test is out of bounds"
                f" ({output_index_to_test} >= {len(output)})"
            )


def _validate_optimizer(opt):
    if not callable(opt):
        raise TypeError("optimizer must be a callable")
    nparams = len(inspect.signature(opt).parameters)
    if nparams != 2:
        raise ValueError("optimizer must take two parameters")
    current = torch.ones(10)
    previous = 0.5
    try:
        result = opt(current, previous)
    except Exception as e:
        raise ValueError("error trying to run sample data in optimizer") from e
    if not isinstance(result, int):
        raise TypeError(f"type of optimizer return value must be int but got {type(result)}")


def drop_patch_and_infer(
    *,
    index_to_drop: int,
    features: torch.Tensor,
    model_probs_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Get the model outputs after dropping a single patch.

    Parameters
    ----------
    index_to_drop : int
        The index of the patch to be removed from the features.
    features : torch.Tensor
        A 2D tensor of input features, where each row represents a patch.
    model_probs_fn : callable
        A function that takes a 2D torch.Tensor of features as input and
        returns a 1D torch.Tensor of model probability outputs.

    Returns
    -------
    1D tensor of model outputs after one patch was removed.

    Notes
    -----
    The function uses torch.cat() to efficiently remove the specified patch,
    which has been benchmarked as the fastest method compared to boolean
    indexing or integer array indexing.
    """
    # Jakub benchmarked three methods to drop a single patch:
    #   1. Index with a boolean array. One boolean value would be False.
    #   2. Index with an integer array, omitting one patch.
    #   3. Slice the array twice to drop the patch and concat the slices.
    # Method 3 is the fastest, so we use it here.
    occluded_features = torch.cat([features[:index_to_drop], features[index_to_drop + 1 :]])
    with torch.inference_mode():
        return model_probs_fn(occluded_features)


def greedy_search(
    *,
    features: torch.Tensor,
    model_probs_fn: Callable[[torch.Tensor], torch.Tensor],
    output_index_to_optimize: int,
    optimizer: Callable[[torch.Tensor, float | None], int],
    num_rounds: int | None = None,
    constant_features: torch.Tensor | None = None,
    disable_progbar: bool = False,
) -> dict[str, int | npt.NDArray]:
    """Perform a greedy search to find important patches.

    This function iteratively removes patches from the input features and observes
    the change in model output. It aims to identify the patches that have the most
    significant impact on the model's predictions for a specified output class.

    Parameters
    ----------
    features : torch.Tensor
        A 2D tensor of input features, where each row represents a patch.
    model_probs_fn : callable
        A function that takes a 2D tensor of features as input and returns
        a 1D tensor of model probabilities.
    output_index_to_optimize : int
        The index of the output class to optimize for in the model's predictions.
    optimizer : callable
        A function that takes two arguments (a Tensor and float) and returns an
        int. The second parameter does not have to be used in the optimizer. The
        first parameter is a 1D Tensor of the probabilities of
        `output_index_to_optimize` for ablated patches. The second parameter is
        the baseline model output for that round of search.
    num_rounds : int or None, optional
        The number of rounds to perform in the greedy search. This determines
        how many patches will be removed. If None, num_rounds is equal to the
        number of patches (ie, rows) in `features` minus 1. Default is None.
    constant_features : torch.Tensor or None, optional
        The features (patches) that should always be present. If not None, must have
        the same second dimension size as `features`. There should be no overlap between
        the patches in `features` and `constant_features`, although this is not checked.

    Returns
    -------
    dict
        A dictionary containing the results of the greedy search:
        - 'optimized_class_index' : int
            The index of the output class that was optimized.
        - 'model_outputs' : np.ndarray
            A 2D array of model outputs after each round of patch removal. The first row
            in the array is the baseline model outputs, without patch removal.
        - 'ablated_patches' : np.ndarray
            An 1D array of indices indicating which patches were removed in each round.

    Notes
    -----
    The kth index of "model_outputs" corresponds to the model outputs after removing patches up
    to index k in "ablated_patches". Assuming the outputs of the search are stored in a dictionary
    `r` and `k` is the index, the value `r["model_outputs"][k]` is the output of the model after
    removing all of the patches in `r["ablated_patches"][:k]`.
    """
    features = features.clone()
    num_rounds = num_rounds or len(features) - 1
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if num_rounds > len(features):
        raise ValueError(f"num_rounds is greater than the number of patches ({len(features)})")
    _validate_model_probs_fn(
        features=features,
        model_probs_fn=model_probs_fn,
        output_index_to_test=output_index_to_optimize,
    )
    _validate_optimizer(optimizer)

    indices_original_frame = list(range(len(features)))
    # Patches that we drop after each iteration. These indices correspond to patches in
    # `features`.
    dropped_indices_original_frame: list[int] = []
    # Keep a running list of model outputs.
    model_outputs: list[torch.Tensor] = []

    # Add in constant features if the user wants them. We can concatenate the two
    # tensors here and still be sure that we only run greedy search on `features`
    # because we set the indices over which to search above. Of course, this relies
    # on features being first in the concatenated tensor.
    if constant_features is not None:
        _validate_model_probs_fn(
            features=constant_features,
            model_probs_fn=model_probs_fn,
            output_index_to_test=output_index_to_optimize,
        )
        if constant_features.shape[1] != features.shape[1]:
            raise ValueError("Second dimensions not equal between features and constant_features")
        # NOTE: never change the order of this concatenation.
        features = torch.cat([features, constant_features])

    # Get the baseline. Tensor of shape (C,).
    with torch.inference_mode():
        baseline = model_probs_fn(features).cpu()
    model_outputs.append(baseline)

    for _ in tqdm(range(num_rounds), desc="Rounds", disable=disable_progbar):
        # Sanity test...
        if set(indices_original_frame).intersection(set(dropped_indices_original_frame)):
            raise ValueError("shared values between indices to test and indices to drop")
        # Remove any patches that we identified in previous iterations.
        # In the first iteration, this uses all patches.
        features_ablated = features[indices_original_frame]

        # We break on one element because if we ablate the one element, we have
        # none left, and the model input would be empty.
        if features_ablated.numel() == 1:
            print("Only one element left. Breaking early...")
            break

        # Drop patches individually and get model outputs.
        num_patches_to_test = len(features_ablated)
        # tensor of NxC
        ablated_probs = torch.stack(
            [
                drop_patch_and_infer(
                    index_to_drop=index_to_drop,
                    features=features_ablated,
                    model_probs_fn=model_probs_fn,
                )
                for index_to_drop in range(num_patches_to_test)
            ]
        )
        if ablated_probs.ndim != 2:
            raise ValueError(
                f"Predictions for ablated features is not 2D, got {ablated_probs.ndim}"
            )

        with torch.inference_mode():
            baseline_this_round: float = model_probs_fn(features_ablated)[
                output_index_to_optimize
            ].item()
        current = ablated_probs[:, output_index_to_optimize]
        index_ablated_reference_frame = optimizer(current, baseline_this_round)
        if not isinstance(index_ablated_reference_frame, int):
            raise ValueError(f"Expected int but got {type(index_ablated_reference_frame)}")

        # Convert the reference space of "features_ablated" to "features".
        original_patch_index = indices_original_frame.pop(index_ablated_reference_frame)
        dropped_indices_original_frame.append(original_patch_index)
        model_outputs.append(ablated_probs[index_ablated_reference_frame].cpu())

    results: dict[str, int | npt.NDArray] = {
        "optimized_class_index": output_index_to_optimize,
        "model_outputs": torch.stack(model_outputs).cpu().numpy(),
        "ablated_patches": np.array(dropped_indices_original_frame),
    }
    return results
