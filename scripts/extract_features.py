"""Use CONCH to extract latent features from whole slide images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import openslide
import timm
import torch
from conch.open_clip_custom import create_model_from_pretrained
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PathType = Union[str, Path]


def read_patch_coords(path: PathType) -> Tuple[np.ndarray, int]:
    """Read HDF5 file of patch coordinates are return numpy array.

    Returned array has shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    coords: np.ndarray
    with h5py.File(path, mode="r") as f:
        coords = f["/coords"][()]  # type: ignore
        coords_metadata = f["/coords"].attrs
        if "patch_level" not in coords_metadata.keys():
            raise KeyError(
                "Could not find required key 'patch_level' in hdf5 of patch "
                "coordinates. Has the version of CLAM been updated?"
            )
        patch_level = coords_metadata["patch_level"]
        if patch_level != 0:
            raise NotImplementedError(
                f"This script is designed for patch_level=0 but got {patch_level}"
            )
        if coords.ndim != 2:
            raise ValueError(f"expected coords to have 2 dimensions, got {coords.ndim}")
        if coords.shape[1] != 2:
            raise ValueError(
                f"expected second dim of coords to have len 2 but got {coords.shape[1]}"
            )

        if "patch_size" not in coords_metadata.keys():
            raise KeyError("expected key 'patch_size' in attrs of coords dataset")

        return coords, coords_metadata["patch_size"]  # type: ignore


class WholeSlideImagePatches(Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to npy file with coordinates of input image.
    um_px : float
        Scale of the resulting patches. For example, 0.5 for ~20x magnification.
    patch_size : int
        The size of patches in pixels.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        patch_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.transform = transform

        assert Path(wsi_path).exists(), "wsi path not found"
        assert Path(patch_path).exists(), "patch path not found"

        self.patches, self.patch_size = read_patch_coords(self.patch_path)

        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        assert self.patches.shape[1] == 2, "expected second dimension to have len 2"

    def worker_init(self, worker_id: int | None = None) -> None:
        del worker_id
        self.slide = openslide.open_slide(self.wsi_path)

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        coords: Sequence[int] = self.patches[idx]
        assert len(coords) == 2, "expected 2 coords (minx, miny)"
        minx, miny = coords

        patch_im = self.slide.read_region(
            location=(minx, miny), level=0, size=(self.patch_size, self.patch_size)
        )
        patch_im = patch_im.convert("RGB")

        patch_im = self.transform(patch_im)

        return patch_im


def main(
    *,
    wsi_dir: PathType,
    patch_dir: PathType,
    save_dir: PathType,
    wsi_extension: str,
    num_workers: int = 8,
    batch_size: int = 64,
    slide_ids: Optional[Sequence[str]] = None,
):
    print("Arguments")
    print(f"  {wsi_dir=}")
    print(f"  {patch_dir=}")
    print(f"  {save_dir=}")
    print(f"  {wsi_extension=}")
    print(f"  {num_workers=}")
    print(f"  {batch_size=}")
    print("-" * 40)

    wsi_dir = Path(wsi_dir)
    patch_dir = Path(patch_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if slide_ids is None:
        patch_paths = sorted(patch_dir.glob("*.h5"))
        if not patch_paths:
            raise FileNotFoundError("no patches found")
        wsi_paths = [wsi_dir / p.with_suffix(wsi_extension).name for p in patch_paths]
        for p in wsi_paths:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        patch_paths = [patch_dir / f"{slide_id}.h5" for slide_id in slide_ids]
        patch_paths = [p for p in patch_paths if p.exists()]
        assert patch_paths
        patch_paths_not_found = len(slide_ids) - len(patch_paths)
        print(f"Could not find {patch_paths_not_found} patch files")
        wsi_paths = [wsi_dir / f"{p.stem}{wsi_extension}" for p in patch_paths]
        for p in wsi_paths:
            if not p.exists():
                raise FileNotFoundError(f"slide file not found: {p}")
        for p in patch_paths:
            if not p.exists():
                raise FileNotFoundError(f"patch h5 file not found: {p}")
    assert all(p.exists() for p in wsi_paths), "at least one slide not found"

    print(f"Embedding {len(wsi_paths):,} slides")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if args.encoder == "virchow2":
        # need to specify MLP layer and activation function for proper init
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    model = model.eval()
    model = model.to(device)

    for wsi_path, patch_path in tqdm(
        zip(wsi_paths, patch_paths), total=len(wsi_paths), desc="Slides"
    ):
        out_path = save_dir / wsi_path.with_suffix(".pt").name
        if out_path.exists():
            continue
        dset = WholeSlideImagePatches(
            wsi_path, patch_path=patch_path, transform=transform
        )
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=dset.worker_init,
            pin_memory=True,
            prefetch_factor=4,
        )
        embeddings: List[torch.Tensor] = []

        for batch in tqdm(loader, leave=False, desc="Patches"):
            batch = batch.to(device)
            with torch.inference_mode():
                if args.encoder == "virchow2":
                    output = model(batch)  # size: B x 261 x 1280

                    class_token = output[:, 0]  # size: B x 1280
                    patch_tokens = output[
                        :, 5:
                    ]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

                    # concatenate class token and average pool of patch tokens
                    # embedding = torch.cat(
                    #     [class_token, patch_tokens.mean(1)], dim=-1
                    # )  # size: B x 2560
                    embedding = class_token  # size: B x 1280

                embeddings.append(embedding.cpu())
        embeddings_tensor = torch.cat(embeddings)
        del embeddings
        assert embeddings_tensor.shape[0] == len(dset)

        torch.save(embeddings_tensor, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--encoder",
        choices=["virchow2"],
        help="Which encoder to use.",
    )
    p.add_argument(
        "--wsi-dir", required=True, help="Directory with whole slide images."
    )
    p.add_argument(
        "--patch-dir", required=True, help="Directory with patch HDF5 files."
    )
    p.add_argument("--save-dir", required=True)
    p.add_argument(
        "--wsi-extension",
        required=True,
        help="Whole slide image extension. Eg, '.svs', '.ndpi', '.tif'",
    )
    p.add_argument("--num-workers", default=8, type=int)
    p.add_argument(
        "--batch-size", default=64, type=int, help="Batch size for DataLoader."
    )
    p.add_argument("--slide-ids", nargs="+", default=None)
    args = p.parse_args()
    assert args.num_workers > 0

    main(
        wsi_dir=args.wsi_dir,
        patch_dir=args.patch_dir,
        save_dir=args.save_dir,
        wsi_extension=args.wsi_extension,
        slide_ids=args.slide_ids,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
