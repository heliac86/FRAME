# FRAME environments (proposed public setup)

The files `reconstruction.yml` and `analysis.yml` are **curated installation candidates** derived from working October 28, 2026 Conda exports. They are **not** exact replicas of those environments, have **not** been installed/tested from scratch, and do not constitute proof of bitwise reproduction of the reported experiments. They deliberately omit a long list of transitive packages, mixed CUDA 11/12/13 toolkits, `prefix:` paths, and duplicated Conda/pip PyTorch installations.

## Observed, already tested configurations

| Role | Local Conda Python | Working PyTorch | CUDA runtime | NumPy |
| --- | --- | --- | --- | --- |
| Reconstruction / gating extraction | 3.10.19 | 1.13.1+cu117 | 11.7 | 1.26.4 |
| Scoring / statistical plots | 3.9.23 | 2.5.1+cu121 | 12.1 | 2.0.2 |

Both used an NVIDIA RTX 3090. The exact driver, compiler toolchain, `tinycudann` source revision, and full set of imports from the checked-out `nesvor/` must still be established for independently reproducible installation.

## Proposed creation (new environments only)

From the FRAME repository root on Linux with NVIDIA GPU support:

```bash
conda env create -f environments/reconstruction.yml
conda env create -f environments/analysis.yml
```

Do **not** run `conda env update` against the research environments `nesvor` or `nerf_tcnn` while validating these files. Creating a new environment is safer. If solving fails, preserve the error instead of force-upgrading PyTorch or mixing CUDA 12+ pip runtimes into the reconstruction environment.

### Reconstruction environment

```bash
conda activate frame-reconstruction
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

FRAME depends on the **CUDA-compiled tiny-cuda-nn PyTorch bindings**. These must be compiled separately against the installed PyTorch and available CUDA compiler. Official source build instructions: <https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension>. For actual reproducibility, select and record a **specific compatible tiny-cuda-nn commit** (this has not yet been recovered from the working server); do not assume the current upstream HEAD works with PyTorch 1.13.1. The installed package version `tinycudann==2.0` in the export does **not** pin a source commit or compiled binary compatibility.

After tiny-cuda-nn is built, from the FRAME root run:

```bash
python -m pip install -e . --no-deps
python -c 'import torch, tinycudann, nesvor; print(torch.__version__, torch.version.cuda, tinycudann.__file__, nesvor.__file__)'
python scripts/run_frame_volume.py --help
```

`--no-deps` is deliberate: `setup.py` and third-party dependency declarations must be audited before allowing pip to change the selected CUDA/PyTorch stack. If your FRAME checkout requires a missing module, install a version compatible with the pinned core and document it; do not treat this initial subset as exhaustive.

Gating checkpoint extraction uses this same environment, e.g.:

```bash
python analysis/gating/extract_4x_gating.py --help
```

### Analysis and optional quantitative evaluation environment

```bash
conda activate frame-analysis
python -c 'import torch, numpy, pandas, scipy, matplotlib, nibabel, lpips, piq; print(torch.__version__, torch.version.cuda, numpy.__version__)'
python -m unittest discover -s analysis/gating -p test_4mod.py -v
python evaluation/evaluate_frame_pair.py --help
```

The CLI for extracting gating weights must **not** be executed in this environment. Plots and evaluation are separated from neural reconstruction. LPIPS may fetch pretrained AlexNet model weights during the first full metric evaluation.

## Historical export caveats

* Both input exports contain absolute `name:` and `prefix:` values tied to the original server. These were removed.
* The reconstruction export combines `pytorch=1.13.1`/`pytorch-cuda=11.7` with pip `torch==1.13.1+cu117`, many pip `nvidia-*-cu12` packages, and Conda toolchain packages spanning CUDA 11.7 to 13.1. This reflects an installed environment; it is **not** a coherent installation recipe to replay.
* The analysis export includes the CUDA 12.1 toolchain and matching pip CUDA 12.1 wheels, which are not required to be listed manually when using the published Conda `pytorch-cuda=12.1` runtime.
* `nesvor==0.6.0rc1` from the installed export is **not** used as a dependency: the FRAME repo's own `nesvor/` implementation is authoritative.
* The exports also list packages not selected for these minimal candidates (such as MONAI, torchaudio, radiomics, OpenCV). Add them only if a tested FRAME command or `setup.py` inspection demonstrates they are needed.
* These proposed environments install the same **major** torch and CUDA versions, but use a simplified Conda-based install path. Differences from the original pip binary distributions must be validated with real training and image evaluation before asserting experimental reproducibility.

For the baseline official PyTorch version/compatibility instructions see <https://pytorch.org/get-started/previous-versions/>.
