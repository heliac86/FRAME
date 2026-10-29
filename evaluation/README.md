# FRAME single-volume MRI reconstruction evaluation

This directory supplies a minimal public interface to the **unchanged** scoring functions extracted from the research script `evaluate_ff_hparam_search.py` (through its `validate_geometry` function). It does **not** run the original 12-method hyperparameter search, retrain a network, or recompute the 48-patient statistics.

## Setup

Run in the separate **analysis/evaluation** Conda environment, not necessarily the FRAME/NeSVoR training environment. Dependencies: `numpy`, `nibabel`, `scipy`, `scikit-image`, `torch`, `torchmetrics`, `lpips`, `piq`, `pandas` (and their transitive requirements). LPIPS initializes AlexNet, whose pretrained weights may require a cached download on the first run.

## Check an image pair without scoring

```bash
python evaluation/evaluate_frame_pair.py \
  --gt /path/to/reference.nii.gz \
  --recon /path/to/frame_reconstruction.nii.gz \
  --out-dir outputs/evaluation/check1 \
  --device cuda:0 --check-only
```

## Compute metrics

```bash
python evaluation/evaluate_frame_pair.py \
  --gt /path/to/reference.nii.gz \
  --recon /path/to/frame_reconstruction.nii.gz \
  --out-dir outputs/evaluation/case1 \
  --device cuda:0
```

Outputs: `metrics.json` contains all scores, environment settings, and SHA-256 provenance; `metrics.csv` contains one-row numerical scores. Both require a new or empty output directory.

**Paper metrics:** PSNR, SSIM, NRMSE, LPIPS, FSIM, HFEN. Additional diagnostics provided by the research scoring implementation: SNR and VIF, plus per-axis metrics and `k_scale` (GT-derived intensity matching used **only** at evaluation).

**Limitations:** An image pair must be geometry-compatible. The score is a *single-volume* score, not a patient or population mean; the published principal results first average FLAIR and T1ce within each patient and then aggregate across the 48 held-out patients. Different library versions may affect learned metrics. Do not use high-resolution GT as a target during FRAME optimization.
