"""Input-ablation transforms applied to a post-collate batch dict.

Each transform is a pure function (dict) -> dict.  Applied in _run_evaluation
*before* InferenceDataset.move_batch_to_device, while tensors are still on CPU
in float32 / float64.

Dropout ablations mirror debug/make_dropout_variants.py (fixed for the real
eval-batch shapes: K is [B,V,3,3], not [B,V,4,4]).

Jitter is pre-voxel: noise is added to the raw semi_dense_points_orig coords,
out-of-bounds points are dropped (mirroring dataset/shaper_dataset.py:62), then
the SparseTensor is rebuilt via preprocess_point_cloud -- the same voxelizer
used by InferenceDataset.custom_collate at dataset/shaper_dataset.py:179.
"""

from typing import Callable, List, Tuple

import torch

from dataset.point_cloud import preprocess_point_cloud


BatchTransform = Callable[[dict], dict]

JITTER_SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1]

def drop_images(batch):
    batch["images"] = torch.zeros_like(batch["images"])
    return batch


def drop_masks(batch):
    batch["masks_ingest"] = torch.zeros_like(batch["masks_ingest"])
    return batch


def drop_boxes(batch):
    batch["boxes_ingest"] = torch.zeros_like(batch["boxes_ingest"])
    return batch


def drop_intext(batch):
    K = batch["camera_intrinsics"]   # [B, V, 3, 3]
    E = batch["camera_extrinsics"]   # [B, V, 4, 4]
    batch["camera_intrinsics"] = torch.eye(3, dtype=K.dtype).expand_as(K).contiguous()
    batch["camera_extrinsics"] = torch.eye(4, dtype=E.dtype).expand_as(E).contiguous()
    return batch


def make_jitter(sigma: float, num_bins: int, seed: int = 0) -> BatchTransform:
    gen = torch.Generator().manual_seed(seed)

    def fn(batch):
        noised_tensors = []
        noised_arrays = []
        for arr in batch["semi_dense_points_orig"]:
            t = torch.from_numpy(arr).float()
            t = t + torch.randn(t.shape, generator=gen, dtype=t.dtype) * sigma
            valid = torch.all(torch.abs(t) <= 1.0, dim=-1)
            t = t[valid]
            noised_tensors.append(t)
            noised_arrays.append(t.numpy())
        batch["semi_dense_points"] = preprocess_point_cloud(
            noised_tensors, num_bins=num_bins
        )
        batch["semi_dense_points_orig"] = noised_arrays
        return batch

    return fn


DROP = {
    "drop_images": drop_images,
    "drop_masks":  drop_masks,
    "drop_boxes":  drop_boxes,
    "drop_intext": drop_intext,
}


def list_all(num_bins: int, seed: int = 0) -> List[Tuple[str, BatchTransform]]:
    """Ordered (name, transform) list for the full ablation suite."""
    items: List[Tuple[str, BatchTransform]] = [("baseline", lambda b: b)]
    items += list(DROP.items())
    items += [
        (f"jitter_{s}", make_jitter(s, num_bins=num_bins, seed=seed))
        for s in JITTER_SIGMAS
    ]
    return items


def resolve(names, num_bins: int, seed: int = 0) -> List[Tuple[str, BatchTransform]]:
    """Resolve CLI names to (name, transform) pairs.  'all' expands to full suite."""
    if names == ["all"]:
        return list_all(num_bins=num_bins, seed=seed)
    registry = dict(list_all(num_bins=num_bins, seed=seed))
    out = []
    for n in names:
        if n not in registry:
            raise ValueError(
                f"Unknown ablation {n!r}. Known: {list(registry)}"
            )
        out.append((n, registry[n]))
    return out
