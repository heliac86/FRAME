# FRAME

> **테크니컬 포트폴리오 — 심사 항목별 증빙**
>
> 심사 대상: 1번 Python 데이터 처리 · 2번 Linux/Docker 연구 환경 · 7번 공개 머신러닝 결과물
>
> 본 저장소는 3차원 뇌 MRI 영상 복원을 위한 딥러닝 연구 프로젝트 FRAME의 구현 결과물입니다. 아래에서는 선택한 각 심사 항목에 대해 수행한 작업, 실제 구현 위치, 코드에서 확인할 사항, 검증 결과를 구분하여 제시합니다.

## 심사 항목 1. Python 등 데이터 처리 언어 활용 능력

**평가 기준:** Python 등의 프로그래밍 언어를 사용하여 데이터를 처리하고 분석하는 능력

**수행 내용:** Python을 이용하여 MRI 전처리 파이프라인, 딥러닝 모델의 핵심 구성 요소, 주파수 영역 손실 함수 및 영상 품질 평가 코드를 구현하였습니다.

| 구현 기능 | 실제 코드 위치 | 확인 방법 |
|---|---|---|
| MRI 열화 시뮬레이션 | [brats_pre4.py — 9~37행](preprocessing/brats_pre4.py#L9-L37) | `degrade_mri_volume()`에서 Gaussian blur, 다운샘플링, 슬라이스 강도 변동 구현 확인 |
| FFT 기반 손실 함수 | [losses.py — 24~102행](nesvor/inr/losses.py#L24-L102) | `FocalFrequencyLoss` 클래스와 `torch.fft.fft2()` 연산 확인 |
| 공간 적응형 Gating | [models.py — 137~236행](nesvor/inr/models.py#L137-L236) | `GatingMLP`에서 입력 특징을 이용한 수준별 Gating 가중치 예측 확인 |
| 영상 품질 지표 | [frame_metrics.py — 123~369행](evaluation/frame_metrics.py#L123-L369) | PSNR, SSIM, NRMSE, LPIPS, FSIM, HFEN 계산 함수 확인 |

단순히 라이브러리를 사용한 것이 아니라, MRI 데이터 처리 과정과 딥러닝 모델의 구성 요소, 손실 함수 및 정량 평가 기능을 Python 코드로 직접 구현하였음을 확인하실 수 있습니다.

---

## 심사 항목 2. Linux·Docker 기반 연구 및 데이터 분석 환경 구축 능력

**평가 기준:** Linux, Docker 및 가상환경 등을 이용한 데이터 분석·연구 환경 구축 능력

**수행 내용:** Linux 기반 NVIDIA GPU 서버에서 Docker 및 Conda를 활용하여 FRAME 연구 환경을 운영하였으며, CUDA·PyTorch 버전의 차이에 따른 의존성 충돌을 방지하기 위해 학습 환경과 분석 환경을 분리하였습니다.

| 환경 구성 | 실제 증빙 위치 | 확인 방법 |
|---|---|---|
| MRI 복원·학습 환경 | [reconstruction.yml](environments/reconstruction.yml) | Python 3.10, PyTorch 1.13.1, CUDA 11.7 설정 확인 |
| 분석·정량 평가 환경 | [analysis.yml](environments/analysis.yml) | Python 3.9, PyTorch 2.5.1, CUDA 12.1 설정 확인 |
| 패키지 의존성 관리 | [environments/README.md](environments/README.md) | 환경별 역할, 설치 절차와 CUDA 호환성 설명 확인 |
| 학습 환경 실행 코드 | [run_frame_volume.py](scripts/run_frame_volume.py) | GPU 기반 복원 실행 명령 구성 확인 |

**실제 실행 환경 증빙**

- 증빙 화면 A: Docker 컨테이너 내부 FRAME 학습 환경 — Python, PyTorch, CUDA 및 GPU 인식 결과
- <img width="338" height="88" alt="training_evi" src="https://github.com/user-attachments/assets/41bf405f-0f1c-4566-8a2f-69a2fa1ff8a6" />
- 증빙 화면 B: Docker 컨테이너 내부 분석 환경 — Python, PyTorch, CUDA 및 주요 분석 라이브러리 버전
- <img width="338" height="155" alt="analysis_evi" src="https://github.com/user-attachments/assets/84d7e09b-1cbe-4b63-b6eb-8257a1a6e8e8" />
- 증빙 화면 C: Docker 호스트의 실행 중인 컨테이너 정보 — 확보 가능한 경우 별도 첨부
- <img width="925" height="59" alt="docker_containter" src="https://github.com/user-attachments/assets/9d4447b5-d568-456e-9e92-a690b880fc1a" />


서로 다른 딥러닝·분석 환경의 의존성을 분리하여 관리하고, 실제 GPU 서버에서 해당 환경들을 사용하였음을 확인할 수 있습니다.

**증빙 범위:** 공개된 YAML 파일은 기존 실험 환경을 바탕으로 정리한 설치 구성안입니다.

---

## 심사 항목 7. 머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개

**평가 기준:** 머신러닝 라이브러리를 활용한 개발 결과물을 구현하고, 타인이 확인할 수 있도록 공개하는 능력

**수행 내용:** PyTorch 기반 NeSVoR의 MRI 복원 구조를 확장하여 FRAME을 개발하였으며, Multi-resolution Hash Grid Encoding에 Focal Frequency Loss 및 Spatially Adaptive Gating을 결합하였습니다. 해당 모델의 구현과 학습·복원·평가 절차를 GitHub에 공개하였습니다.

| 개발 결과물 | 실제 코드 위치 | 확인 방법 |
|---|---|---|
| FRAME 모델 및 Gating 구현 | [models.py — 239~554행](nesvor/inr/models.py#L239-L554) | `INR` 클래스에서 Hash Grid 특징에 공간 Gating을 적용하는 코드 확인 |
| PyTorch 학습 코드 | [train.py — 22행부터](nesvor/inr/train.py#L22-L130) | `train()` 함수에서 최적화, 손실 함수 통합 및 Gating 파라미터 관리 확인 |
| 복원 실행기 | [run_frame_volume.py — 28~114행](scripts/run_frame_volume.py#L28-L114) | 입력 MRI, 학습 인자, 출력 및 복원 명령 구성 확인 |
| Gating 분석 | [analysis/gating/](analysis/gating/) | 학습된 가중치 추출 및 통계 분석 코드 확인 |
| 자동 테스트 | [test_4mod.py](analysis/gating/test_4mod.py) | Gating 데이터 처리 및 분석에 관한 회귀 테스트 확인 |
| 영상 품질 평가 | [evaluate_frame_pair.py](evaluation/evaluate_frame_pair.py) | 실제 복원 결과와 기준 MRI 간 정량 평가 실행 코드 확인 |
| 환경 구성 및 실행 안내 | [environments/](environments/) 및 README의 실행 방법 | 설치 구성, 복원 및 평가 실행 명령 확인 |

**실제로 수행한 검증**

| 검증 항목 | 확인된 결과 |
|---|---|
| Gating 분석 자동 테스트 | 15개 테스트 통과 |
| GPU 체크포인트 사전 검증 | 실제 FRAME 체크포인트 로딩 및 Gating 가중치 조회 성공 |
| 정량 평가 실행 | 실제 MRI 한 쌍에 대한 평가 및 CSV·JSON 결과 저장 성공 |
| 전체 48명 실험의 신규 환경 재현 | 아직 미검증 |

머신러닝 모델의 핵심 구현부터 실행 인터페이스, 정량 평가 및 자동 테스트까지 공개되어 있어, 개발 결과물의 구조와 검증 절차를 직접 확인할 수 있습니다.

---

## FRAME 연구 내용 및 실험 결과

아래부터는 연구 배경, FRAME의 세부 알고리즘, 논문에 보고된 실험 결과, 설치 및 실행 방법을 설명합니다.

**Frequency-Regulated Adaptive Multi-resolution Encoding for Implicit Neural MRI Slice-to-Volume Reconstruction**

FRAME is a **volume-specific implicit neural representation (INR)** framework for reconstructing three-dimensional MRI from sparsely sampled, anisotropic slice observations. It combines a point-spread-function (PSF) observation model, **patch-based Focal Frequency Loss**, and **spatially adaptive gating of multi-resolution hash-grid features**.

> **Project status:** Public research-code release. The original experiments were run in an existing Linux/Docker/NVIDIA GPU environment. The provided public-facing commands have passed the checks listed below, but fresh Conda environment creation and a full end-to-end training rerun have **not** yet been validated.

## Overview

FRAME optimizes **one representation per input volume** from its observed, degraded MRI slices. The high-resolution ground truth is **not** used as a training target. It is used only for reference-based evaluation and selected post-hoc analyses.

The method uses two complementary adaptations:

1. **Frequency-adaptive supervision:** patch-based Focal Frequency Loss (FF Loss) weights Fourier-domain reconstruction errors on **predicted versus observed slice patches**, after the acquisition/PSF forward model. It emphasizes *currently mismatched frequencies*, not exclusively high frequencies.
2. **Spatially adaptive representation:** a small gating network conditions on spatial coordinates and local content context to predict weights for individual multi-resolution hash-grid levels. These weights modulate the encoded features during prediction; they are **not** direct measurements of Fourier frequency bands.

The underlying PSF-aware reconstruction pipeline is derived from **NeSVoR**, with FRAME-specific modifications for frequency supervision and level-wise spatial gating. Please review upstream provenance and licensing before redistributing or relicensing the full source tree.

## Main result

In the revised research manuscript, the principal experiment evaluates **48 held-out BraTS2020 patients** under synthetic **4× through-plane degradation** with Gaussian blur and controlled alternating-slice intensity perturbations. FLAIR and T1ce scores are averaged **within each patient**, then across the 48 patients.

| Method | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | LPIPS ↓ | FSIM ↑ | HFEN ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trilinear | 25.7114 | 0.8233 | 0.1497 | 0.08187 | 0.97107 | 0.5360 |
| Cubic B-spline | 25.8155 | 0.8233 | 0.1475 | 0.07831 | 0.97097 | 0.5487 |
| NeSVoR | 25.0632 | 0.8831 | 0.1596 | 0.03836 | 0.98034 | 0.3554 |
| **FRAME** | **27.3072** | **0.8995** | **0.1247** | **0.03590** | **0.98218** | **0.3152** |

These are **manuscript-reported cohort means**, not metrics regenerated by the scripts in this repository during release preparation. The manuscript also compares FRAME with CuNeRF, ArSSR, ECLARE, and GSVR. Results concern the specified **synthetic** degradation protocol and do not establish performance on all clinical acquisition artifacts.

## Repository layout

```text
FRAME/
├── nesvor/                         # Local, FRAME-modified INR / reconstruction implementation
├── scripts/
│   └── run_frame_volume.py         # Single-volume reconstruction entry point
├── preprocessing/
│   └── brats_pre4.py               # Synthetic BraTS through-plane degradation
├── analysis/
│   └── gating/
│       ├── extract_4x_gating.py    # GPU checkpoint loading and weight extraction
│       ├── gating_backend.py
│       ├── gating_common.py
│       ├── analyze_4x_modalities.py # CPU/analysis-env statistics and figures
│       └── test_4mod.py            # Gating analysis regression tests
├── evaluation/
│   ├── frame_metrics.py            # Research scoring / alignment functions
│   ├── evaluate_frame_pair.py      # Optional GT-versus-reconstruction evaluation
│   └── README.md
├── environments/
│   ├── reconstruction.yml          # Proposed, not yet clean-install validated
│   ├── analysis.yml                # Proposed, not yet clean-install validated
│   └── README.md
└── setup.py
```

## Requirements and installation

- Linux, an NVIDIA GPU, appropriate NVIDIA driver, and a working CUDA build toolchain for the CUDA-enabled reconstruction backend.
- Two **separate environments**: reconstruction and gating extraction use PyTorch **1.13.1 / CUDA 11.7**; statistical analysis and optional reference-based evaluation use PyTorch **2.5.1 / CUDA 12.1**.
- The CUDA `tinycudann` binding must be built **against the chosen reconstruction PyTorch/CUDA environment**. The exported `tinycudann==2.0` package version alone does not identify the original source commit or ensure binary compatibility.

The proposed environment files are **curated installation candidates**, not exact lockfiles. They are based on working environments (Python **3.10.19** for reconstruction, **3.9.23** for analysis). See [environments/README.md](environments/README.md) for build cautions and the corresponding setup steps.

```bash
# From the FRAME repository root; creates NEW named environments.
conda env create -f environments/reconstruction.yml
conda env create -f environments/analysis.yml

# After installing/building a compatible tiny-cuda-nn binding:
conda activate frame-reconstruction
python -m pip install -e . --no-deps
python scripts/run_frame_volume.py --help
```

Do **not** update the original working research environments in place as a shortcut. Do not assume that a successful YAML parse proves that a clean environment can be solved and installed.

## Quick start: reconstruct one MRI volume

FRAME fits a separate INR from an **already prepared, degraded** 3D MRI NIfTI file. This interface does **not** require a high-resolution GT volume.

```bash
conda activate frame-reconstruction

# Inspect the exact underlying CLI without starting training.
python scripts/run_frame_volume.py \
  --input /path/to/degraded_flair.nii.gz \
  --output outputs/example/frame_flair.nii.gz \
  --mask /path/to/degraded_flair.nii.gz \
  --device 0 \
  --dry-run

# Remove --dry-run to perform volume-specific optimization.
python scripts/run_frame_volume.py \
  --input /path/to/degraded_flair.nii.gz \
  --output outputs/example/frame_flair.nii.gz \
  --mask /path/to/degraded_flair.nii.gz \
  --device 0
```

The wrapper writes the reconstructed volume (`.nii.gz`) and a volume-specific model checkpoint (`.pt`). It refuses to overwrite existing outputs. Its current defaults follow the local research runner, including **2,000 iterations**, **4,096 observed pixels per batch**, **48×48 patches**, **up to 8 patches**, FF Loss weight **0.0125**, exponent **2**, seed **42**, and gating learning-rate scale **1.0**.

**Reproduction boundary:** The manuscript specifies **256 Monte Carlo PSF samples per pixel**, and the current wrapper explicitly passes `--n-samples 256`. The other historical runner settings and the full 48-patient results still need end-to-end revalidation before claiming complete experimental reproduction.

## Prepare synthetic BraTS inputs (optional)

The main BraTS experiment uses **4× through-plane subsampling** after Gaussian blur along the third volume axis, with alternating-slice intensity changes. The original preprocessing script is preserved to avoid silently changing the synthetic data protocol.

```bash
conda activate frame-reconstruction
python preprocessing/brats_pre4.py \
  --input_dir /path/to/BraTS2020_original \
  --output_dir /path/to/BraTS2020_degraded_4x \
  --factor 4 \
  --sigma 1.5 \
  --gain_min 0.05 \
  --gain_max 0.10 \
  --seed 42
```

The original script uses a **global NumPy random stream** and does not guarantee an invariant processing order across filesystems. For exact replication of historically generated inputs, preserve both input file traversal order and random-seed behavior. Preprocessing output may include segmentation files; do not treat segmentation or high-resolution reference images as FRAME training targets.

The BraTS data are **not included** here. Obtain and use each dataset subject to its own access, privacy, and license conditions.

## Inspect the spatial gating mechanism (optional)

**Extraction** loads FRAME `.pt` checkpoints and queries the learned hash-grid gating weights. It requires the **reconstruction CUDA environment**, not the separate analysis environment. Original high-resolution MRI is used to define a reference brain ROI for **post-hoc analysis**, not to train the INR.

The following is a **48-patient, four-modality template**; replace every path with your own data locations. `--out` should be a new or empty directory.

```bash
conda activate frame-reconstruction

# Discovery only: file matching; no model inference.
python analysis/gating/extract_4x_gating.py \
  --recon-root /path/to/frame_reconstructions_4x \
  --gt-root /path/to/BraTS2020_original \
  --test-csv /path/to/test.csv \
  --expected-patients 48 \
  --modalities flair t1ce t1 t2 \
  --nesvor-root "$(pwd)" \
  --out outputs/gating/discovery \
  --discovery-only

# Actual extraction; omit --discovery-only, provide a different output path.
python analysis/gating/extract_4x_gating.py \
  --recon-root /path/to/frame_reconstructions_4x \
  --gt-root /path/to/BraTS2020_original \
  --test-csv /path/to/test.csv \
  --expected-patients 48 \
  --modalities flair t1ce t1 t2 \
  --nesvor-root "$(pwd)" \
  --out outputs/gating/extracted_4x \
  --points 50000 \
  --seed 42 \
  --device cuda:0
```

The `--dry-run` mode of **this extraction tool** is a *real GPU checkpoint preflight*: it loads matched model weights and queries 64 coordinates per checkpoint; it is **not** a command-printing dry run. Use `--modalities flair --expected-patients 1` with a suitable one-patient CSV for a small preflight.

Then analyze extracted files in the **separate analysis environment**:

```bash
conda activate frame-analysis
python analysis/gating/analyze_4x_modalities.py \
  --inputs outputs/gating/extracted_4x \
  --out outputs/gating/figures_4x
```

The analysis verifies extraction manifests and produces statistics and visualizations. The number of hash-grid levels is determined by the model's geometry and grid configuration. **`L10` in historical experiment/output names is a run label, not an assertion that the model has ten grid levels.**

Run the existing regression tests:

```bash
conda activate frame-analysis
python -m unittest discover -s analysis/gating -p 'test_4mod.py' -v
```

## Quantitative evaluation (optional)

The evaluation utility requires a high-resolution reference NIfTI and is **not required to reconstruct a volume**. It validates NIfTI geometry, aligns the reconstructed grid, performs GT-derived intensity matching **only for scoring**, and computes the manuscript's six main metrics: PSNR, SSIM, NRMSE, LPIPS, FSIM, and HFEN. It also records SNR, VIF, per-axis scores, and SHA-256 provenance.

```bash
conda activate frame-analysis

# Validate geometry without calculating image-quality metrics.
python evaluation/evaluate_frame_pair.py \
  --gt /path/to/reference_flair.nii.gz \
  --recon /path/to/reconstructed_flair.nii.gz \
  --out-dir outputs/evaluation/preflight \
  --device cuda:0 \
  --check-only

# Compute metrics for one GT/reconstruction pair.
python evaluation/evaluate_frame_pair.py \
  --gt /path/to/reference_flair.nii.gz \
  --recon /path/to/reconstructed_flair.nii.gz \
  --out-dir outputs/evaluation/flair_case \
  --device cuda:0
```

The full evaluation writes `metrics.json` and `metrics.csv`. LPIPS may require pretrained AlexNet weights on the first run. A **single-volume** result must not be presented as the manuscript's **48-patient mean**; the latter requires both FLAIR and T1ce results for each patient and the manuscript's patient-level aggregation. See [evaluation/README.md](evaluation/README.md).

## Validation performed for this repository preparation

Checks below were performed in the existing research environments; these are **not** a new-machine reproducibility certificate.

| Check | Observed result |
| --- | --- |
| Main reconstruction wrapper syntax / `--dry-run` | Passed; the underlying command was printed, training not run |
| Preprocessing and gating Python syntax | Passed |
| Gating analysis regression tests in the analysis environment | **15/15 passed** |
| Gating checkpoint discovery and GPU `--dry-run` | Passed for **BraTS patient 151, FLAIR**, using the original 4× checkpoint |
| Single-pair quantitative evaluation geometry preflight | Passed for **BraTS patient 003, FLAIR** |
| Actual single-pair evaluation | Passed; JSON and CSV produced, PSNR **26.7128 dB**, SSIM **0.9265** |
| Public Conda YAML parse and no-`prefix` check | Passed for both files |
| Fresh public Conda installations | **Not yet tested** |
| Full end-to-end retraining and reproduction of the 48-patient results | **Not yet tested** |

## Provenance, limitations, and responsible use

- **Source:** FRAME extends the NeSVoR reconstruction codebase. Verify third-party notices, licenses, and redistribution permissions before publishing the forked source; a new README alone does not grant permissions.
- **Data:** Do not commit BraTS/UPENN data, patient-level private information, checkpoints, generated reconstructions, or local machine-specific environment exports. Use `.gitignore` and a final staged-file review.
- **Research scope:** Results were obtained under specified synthetic degradations. This research prototype is **not a medical device** and has not been validated for clinical decision-making.
- **Configuration provenance:** The public wrapper follows the supplied executed research-runner settings (including gating learning-rate scale **1.0**). Confirm all effective CLI defaults, including PSF sampling, before asserting exact manuscript-level reproduction.
- **Paper status:** The source for the numerical values above is the author's revised FRAME manuscript; no venue, DOI, or publication status is asserted here. Add a formal citation/BibTeX entry when publication details are confirmed.

## Citation

**FRAME: Frequency-Regulated Adaptive Multi-resolution Encoding for Implicit Neural MRI Slice-to-Volume Reconstruction**. Research manuscript, revised draft (2026). Full bibliographic information and DOI: **to be added when available**.
