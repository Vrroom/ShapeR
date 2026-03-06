# Evaluation

This script is modified from `infer_shape.py`. It runs **inference and metric evaluation simultaneously** — for each input sample, it generates a predicted mesh and immediately computes metrics against the ground truth.

You can also modify this script to decouple the two stages: generate meshes first, then compute metrics separately.

## Data Format

Ground truth vertices, faces, and bounds should be stored **inside the input `.pkl` files** (same format as `infer_shape.py`). The pkl must contain `vertices`, `faces`, and `bounds` fields for evaluation. Samples missing these fields will be skipped.

## Usage

```bash
# Evaluate all pkl files in a directory
python evaluation/eval.py --input_dir data/eval_samples/

# Evaluate specific pkl files (paths relative to data/)
python evaluation/eval.py --input_pkls sample1.pkl sample2.pkl

# Evaluate local pkl files (absolute or relative paths used as-is)
python evaluation/eval.py --input_pkls /path/to/sample.pkl --is_local_path

# Save predicted/GT meshes and pair visualization
python evaluation/eval.py --input_dir data/eval_samples/ --save_meshes --save_visualization
```


## Output

Results are saved to `eval_results.json` (or the path specified by `--eval_output`), containing per-sample metrics and aggregate statistics (mean ± std).
