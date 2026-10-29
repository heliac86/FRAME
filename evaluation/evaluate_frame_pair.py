#!/usr/bin/env python3
"""Evaluate one reconstructed MRI volume against a high-resolution reference.

Usage (from FRAME repository root, in the evaluation Conda environment):
    python evaluation/evaluate_frame_pair.py \
        --gt /path/to/ground_truth.nii.gz \
        --recon /path/to/reconstructed.nii.gz \
        --out-dir outputs/evaluation/example \
        --device cuda:0

The published six metrics use unchanged evaluation functions in frame_metrics.py.
Additional SNR/VIF values and per-axis diagnostic scores are also retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

PRIMARY_METRICS = ("PSNR", "SSIM", "NRMSE", "LPIPS", "FSIM", "HFEN")
ADDITIONAL_METRICS = ("SNR", "VIF")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = ("torch", "numpy", "nibabel", "scipy", "scikit-image", "pandas",
             "lpips", "torchmetrics", "piq")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, type=Path,
                        help="High-resolution reference NIfTI volume (.nii/.nii.gz).")
    parser.add_argument("--recon", required=True, type=Path,
                        help="Reconstructed NIfTI volume (.nii/.nii.gz).")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="New, empty output folder for metrics and metadata.")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or cuda:0 (default: auto).")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate image files and geometry without computing metrics or writing outputs.")
    args = parser.parse_args()

    gt = args.gt.expanduser().resolve()
    recon = args.recon.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    for label, path in (("GT", gt), ("reconstruction", recon)):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"{label} file is missing/empty: {path}")
    if gt == recon:
        parser.error("GT and reconstruction paths must be different files.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        parser.error("Output directory must not already contain files: " + str(out_dir))

    # Deliberately late import: '--help' should work without installing imaging libraries.
    import frame_metrics as metrics
    if not metrics.PIQ_AVAILABLE:
        parser.error("piq is required for FSIM. Activate the analysis/evaluation environment.")
    torch = metrics.torch
    selected = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    try:
        device = torch.device(selected)
    except (TypeError, RuntimeError) as exc:
        parser.error(f"Invalid --device value {selected!r}: {exc}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in the selected evaluation environment.")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        parser.error("Only CPU and CUDA evaluation devices are supported.")

    gt_img = metrics.nib.load(str(gt))
    rec_img = metrics.nib.load(str(recon))
    shape, offset = metrics.validate_geometry(gt_img, rec_img)
    original = gt_img.get_fdata()
    if not metrics.np.isfinite(original).all() or not (original > 0).any():
        parser.error("GT contains nonfinite values or has an empty positive-intensity brain mask.")
    del original

    print(f"[CHECK] GT: {gt} ({gt_img.shape})", flush=True)
    print(f"[CHECK] RECON: {recon} ({rec_img.shape})", flush=True)
    print(f"[CHECK] Geometry: oriented_shape={shape}, voxel_offset={offset.tolist()}", flush=True)
    print(f"[ENV] Python: {sys.executable} | Device: {device}", flush=True)
    if args.check_only:
        print("[CHECK_ONLY OK] Geometry validated; no metrics or outputs created.", flush=True)
        return 0

    # LPIPS loads pretrained AlexNet weights: may require a local model cache/network
    # on the first run. The reference code also uses lpips.LPIPS(net='alex').
    lpips_fn = metrics.lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    results = metrics.evaluate_pair(str(gt), str(recon), lpips_fn, device)
    for name in PRIMARY_METRICS + ADDITIONAL_METRICS + ("k_scale",):
        if name not in results or results[name] is None or not math.isfinite(float(results[name])):
            raise ValueError(f"Missing/nonfinite metric {name}. Refusing to save partial results.")

    record = {
        "evaluation": "single-reference-pair; GT used only for scoring/alignment",
        "gt_path": str(gt),
        "recon_path": str(recon),
        "gt_sha256": _hash_file(gt),
        "recon_sha256": _hash_file(recon),
        "metric_engine_sha256": _hash_file(Path(metrics.__file__).resolve()),
        "evaluation_script_sha256": _hash_file(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "device": str(device),
        "package_versions": _package_versions(),
        "settings": {
            "brain_mask": "GT > 0",
            "normalization": "least-squares intensity matching, GT 0.1/99.9 percentiles, clip [0,1]",
            "ssim_bbox_crop": metrics.USE_SSIM_BBOX_CROP,
            "lpips_3axes": metrics.LPIPS_3AXES,
            "vif_3axes": metrics.VIF_3AXES,
            "fsim_3axes": metrics.FSIM_3AXES,
            "hfen_sigma": metrics.HFEN_SIGMA,
            "minimum_slice_voxels": metrics.MIN_SLICE_VOXELS,
        },
        "primary_metrics": list(PRIMARY_METRICS),
        "additional_metrics": list(ADDITIONAL_METRICS),
        "metrics": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["gt_path", "recon_path", *results.keys()])
        writer.writeheader()
        writer.writerow({"gt_path": str(gt), "recon_path": str(recon), **results})
    print("[RESULTS] Paper metrics:")
    for name in PRIMARY_METRICS:
        print(f"  {name}: {results[name]:.6f}")
    print(f"[DONE] Saved metrics.json and metrics.csv to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
