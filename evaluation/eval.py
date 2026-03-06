"""
Evaluation script for ShapeR evaluation dataset.

Metrics (Chamfer Distance, Normal Consistency, F-score) are computed following the
author's evaluation protocol (metrics_outline.py): both meshes are scaled by
max(bounds)*2, Chamfer is L1, F-score uses linspace(1/144, 2, 1000)[10].

"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scipy.spatial import cKDTree

from dataset.shaper_dataset import InferenceDataset
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper

# Same presets as infer_shape.py: (num_images, token_multiplier, num_denoising_steps)
preset_configs = {
    "quality": (16, 4, 50),
    "speed": (4, 2, 10),
    "balance": (16, 4, 25),
}

def get_threshold_percentage(dist, thresholds):
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32
        )
    return dist, normals_dot_product



def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of ShapeR reconstruction quality."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input pkl files.",
    )
    group.add_argument(
        "--input_pkls",
        type=str,
        nargs="+",
        help="List of input pkl file paths.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balance",
        choices=list(preset_configs.keys()),
        help="Inference preset config (default: balance).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for GPU inference (default: 1).",
    )
    parser.add_argument(
        "--is_local_path",
        action="store_true",
        help="Input paths are local (don't prepend data/ directory).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for output meshes and results (default: output).",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path for JSON results file (default: {output_dir}/eval_results.json).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=100000,
        help="Number of points to sample for evaluation (default: 100000).",
    )
    parser.add_argument(
        "--save_meshes",
        action="store_true",
        help="Save predicted mesh, GT mesh, and a pair visualization GLB.",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save visualization of prediction vs ground truth.",
    )

    args = parser.parse_args()
    if not args.input_dir and not args.input_pkls:
        parser.error("one of --input_dir or --input_pkls is required")
    return args


def collect_pkl_paths(args):
    """Collect all pkl file paths from arguments."""
    if args.input_pkls:
        paths = []
        for p in args.input_pkls:
            if args.is_local_path:
                paths.append(p)
            else:
                paths.append(os.path.join("data", p))
        return paths

    input_dir = args.input_dir
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        sys.exit(1)
    return pkl_files


def load_model(config_preset, device=None):
    """Load ShapeR model, VAE, and text feature extractor.

    Returns model components and inference parameters. token_count and embed_dim
    are returned separately so token_shape can be built per-batch with the
    correct batch dimension.
    """
    num_images, token_multiplier, num_steps = preset_configs[config_preset]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_file = "checkpoints/019-0-bfloat16.ckpt"
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)
    yaml_file = "checkpoints/config.yaml"
    config = omegaconf.OmegaConf.load(yaml_file)

    print(f"Loading model on {device}...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)

    vae = MichelangeloLikeAutoencoderWrapper(
        "checkpoints/vae-088-0-bfloat16.ckpt", device
    )

    text_feature_extractor = TextFeatureExtractor(device=device)
    text_feature_extractor = text_feature_extractor.to(torch.bfloat16)

    model = torch.compile(model, fullgraph=True)
    model = model.eval()
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    embed_dim = vae.get_embed_dim()
    use_shifted_sampling = (
        getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"
    )

    return (
        model, vae, text_feature_extractor, config, device,
        num_images, num_steps, token_count, embed_dim, use_shifted_sampling,
    )


def _eval_single_mesh(pred_verts, pred_faces, gt_verts, gt_faces, n_points, bounds):
    """Evaluate a single mesh pair following the author's protocol from metrics_outline.py.

    Both meshes are scaled by ``max(bounds) * 2`` before computing metrics.
    Chamfer-L1 (not L2) is used; F-score uses ``linspace(1/144, 2, 1000)[10]``.
    """
    metric_scale = float(np.max(bounds)) * 2.0

    pred_mesh = trimesh.Trimesh(
        vertices=pred_verts * metric_scale, faces=pred_faces, process=False
    )
    gt_mesh = trimesh.Trimesh(
        vertices=gt_verts * metric_scale, faces=gt_faces, process=False
    )

    if pred_mesh.vertices.shape[0] <= 3:
        return {"chamfer-L1": 2.0, "normals": 0.0, "f-score": 0.0}

    pointcloud_pred, idx_pred = pred_mesh.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = pred_mesh.face_normals[idx_pred]

    pointcloud_tgt, idx_tgt = gt_mesh.sample(n_points, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = gt_mesh.face_normals[idx_tgt]

    thresholds = np.linspace(1.0 / 144, 2, 1000)

    # Completeness: GT → Pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: Pred → GT
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    chamfer = 0.5 * (completeness + accuracy)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-5)
        for i in range(len(precision))
    ]

    return {
        "chamfer": float(chamfer),
        "normals": float(normals_correctness),
        "f-score": float(F[10]),
    }


def _postprocess_pred_mesh_for_eval(pred_mesh):
    from postprocessing.helper import remove_floating_geometry

    processed = pred_mesh.copy()
    try:
        processed = remove_floating_geometry(processed)
        if len(processed.faces) > 125000:
            processed = processed.simplify_quadric_decimation(face_count=125000)
    except Exception as exc:
        print(f"  [WARN] post-processing failed: {exc}")
        processed = pred_mesh.copy()
    return processed


def print_metrics(name, metrics):
    cd_paper = metrics["chamfer"] * 100.0
    print(
        f"  {name}: "
        f"CD(×10²)={cd_paper:.4f}  "
        f"NC={metrics['normals']:.4f}  "
        f"F1={metrics['f-score']:.4f}"
    )


def print_summary(all_results):
    if not all_results:
        print("No results to summarize.")
        return

    metric_keys = ["chamfer", "normals", "f-score"]
    print("\n" + "=" * 70)
    print(f"Aggregate Results ({len(all_results)} samples)")
    print("=" * 70)
    summary = {}
    for key in metric_keys:
        values = [r["metrics"][key] for r in all_results if key in r["metrics"]]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            summary[key] = {"mean": float(mean), "std": float(std)}
            if key == "chamfer":
                mean_paper = mean * 100.0
                std_paper = std * 100.0
                summary["CD(×10²)"] = {
                    "mean": float(mean_paper),
                    "std": float(std_paper),
                }
                print(
                    f"  {'CD(×10²)':20s}: "
                    f"{mean_paper:.4f} +/- {std_paper:.4f}"
                )
            else:
                print(f"  {key:20s}: {mean:.6f} +/- {std:.6f}")
    print("=" * 70)
    return summary


def _run_evaluation(pkl_paths, args, device, worker_id=None):
    """Run inference and evaluation on a subset of samples.

    Core loop extracted for reuse in single-GPU and multi-GPU paths.
    """
    output_dir = Path(args.output_dir)

    (
        model, vae, text_feature_extractor,
        config, _, num_images, num_steps,
        token_count, embed_dim, use_shifted_sampling,
    ) = load_model(args.config, device=device)

    all_results = []
    skipped = []

    inference_dataset = InferenceDataset(
        config,
        paths=pkl_paths,
        override_num_views=num_images,
    )
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=inference_dataset.custom_collate,
    )

    desc = f"GPU {worker_id}" if worker_id is not None else "Inference"

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc=desc)):
            actual_bs = len(batch["name"])
            batch_names = list(batch["name"])

            has_gt = "vertices" in batch and "faces" in batch

            batch = InferenceDataset.move_batch_to_device(
                batch, device, dtype=torch.bfloat16
            )

            token_shape = (actual_bs, token_count, embed_dim)

            latents_pred = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_feature_extractor,
                num_steps=num_steps,
                use_shifted_sampling=use_shifted_sampling,
            )

            pred_meshes = vae.infer_mesh_from_latents(latents_pred)

            for i in range(actual_bs):
                name = batch_names[i]

                if not has_gt:
                    print(f"  [SKIP] {name}: no ground truth mesh in pickle")
                    skipped.append(name)
                    continue

                pred_mesh = pred_meshes[i]
                pred_mesh_eval = pred_mesh
                gt_verts = batch["vertices"][i]
                gt_faces = batch["faces"][i]
                if isinstance(gt_verts, torch.Tensor):
                    gt_verts = gt_verts.cpu().numpy()
                if isinstance(gt_faces, torch.Tensor):
                    gt_faces = gt_faces.cpu().numpy()

                sample_bounds = batch["bounds"][i]
                if isinstance(sample_bounds, torch.Tensor):
                    sample_bounds = sample_bounds.float().cpu().numpy()

                metrics = _eval_single_mesh(
                    np.array(pred_mesh_eval.vertices, dtype=np.float32),
                    np.array(pred_mesh_eval.faces),
                    np.array(gt_verts, dtype=np.float32),
                    np.array(gt_faces),
                    args.n_points,
                    sample_bounds,
                )

                all_results.append({"name": name, "metrics": metrics})
                if worker_id is None:
                    print_metrics(name, metrics)

                if args.save_visualization:
                    from postprocessing.helper import visualize_prediction_and_groundtruth

                    gt_mesh = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
                    vis_points = batch["semi_dense_points_orig"][i]
                    vis_images = batch["images"][i].float().cpu().numpy()
                    vis_masks = batch["images"][i].float().cpu().clone().numpy()
                    vis_masks[:, 1, :, :] = batch["masks_ingest"][i].float().cpu().numpy()

                    visualize_prediction_and_groundtruth(
                        pred_mesh_eval.copy(),
                        gt_mesh,
                        vis_points,
                        vis_images,
                        vis_masks,
                        batch["caption"][i],
                        sample_name=name,
                        save_path=str(output_dir / f"VIS__{name}.jpg"),
                    )

                if args.save_meshes:
                    sample_idx = batch["index"][i].item()

                    pred_mesh_save = pred_mesh.copy()
                    pred_mesh_save = inference_dataset.rescale_back(
                        sample_idx, pred_mesh_save, do_transform_to_world=False
                    )
                    pred_tmp_path = f"/tmp/mesh_pred_{name}.obj"
                    pred_mesh_save.export(pred_tmp_path)
                    pred_mesh_save = trimesh.load(pred_tmp_path, force="mesh")
                    pred_mesh_save.export(
                        output_dir / (name + ".glb"), include_normals=True
                    )

                    gt_mesh_save = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
                    gt_mesh_save = inference_dataset.rescale_back(
                        sample_idx, gt_mesh_save, do_transform_to_world=False
                    )
                    gt_tmp_path = f"/tmp/mesh_gt_{name}.obj"
                    gt_mesh_save.export(gt_tmp_path)
                    gt_mesh_save = trimesh.load(gt_tmp_path, force="mesh")
                    gt_mesh_save.export(
                        output_dir / f"GT__{name}.glb", include_normals=True
                    )

                    pair_scene = trimesh.Scene()
                    pred_vis = pred_mesh_save.copy()
                    gt_vis = gt_mesh_save.copy()
                    pred_vis.visual.face_colors = [255, 192, 0, 180]
                    gt_vis.visual.face_colors = [34, 139, 34, 180]
                    pair_scene.add_geometry(pred_vis, geom_name="pred")
                    pair_scene.add_geometry(gt_vis, geom_name="gt")
                    pair_scene.export(output_dir / f"PAIR__{name}.glb")

    return all_results, skipped


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output = args.eval_output or str(output_dir / "eval_results.json")

    pkl_paths = collect_pkl_paths(args)
    print(f"Found {len(pkl_paths)} samples to evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\nRunning inference (config={args.config}, batch_size={args.batch_size})...\n"
    )
    all_results, skipped = _run_evaluation(pkl_paths, args, device)

    summary = print_summary(all_results)

    output_data = {
        "config": args.config,
        "batch_size": args.batch_size,
        "n_points": args.n_points,
        "num_samples": len(all_results),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "summary": summary,
        "per_sample": all_results,
    }
    with open(eval_output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {eval_output}")


if __name__ == "__main__":
    main()
