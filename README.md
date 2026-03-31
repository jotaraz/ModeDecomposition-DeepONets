# The Error of Deep Operator Networks Is the Sum of Its Parts

Code for the [The Error of Deep Operator Networks Is the Sum of Its Parts: Branch-Trunk and Mode Error Decompositions](https://arxiv.org/pdf/2602.21910) paper, investigating trunk basis selection and spectral bias in Deep Operator Networks (DeepONet) applied to parametric PDEs.

## Overview

This repository implements and analyses DeepONet architectures for learning solution operators of PDEs (advection-diffusion, KdV, Burgers). Key research questions addressed:

- How do different trunk basis choices (learned, SVD-based, Legendre, Chebyshev, trigonometric) affect operator learning accuracy?
- What spectral bias do neural operator networks exhibit?
- How do SGD and Adam optimizers differ in training dynamics?

## Dependencies

The code requires Python 3.x and the following packages:

```
jax
flax
optax
numpy
scipy
matplotlib
```

Install with:

```bash
pip install jax flax optax numpy scipy matplotlib
```

> **Note:** For GPU support, install the appropriate JAX version for your CUDA version. See the [JAX installation guide](https://github.com/google/jax#installation).

## Repository Structure

```
.
├── src/
│   ├── don_code.py          # Core DeepONet implementation (networks, losses, data loading)
│   ├── execute_don.py       # Single training run execution (called by run.py)
│   ├── run.py               # Batch training orchestration
│   └── analysis/
│       ├── RELEVANT/        # Analysis and plotting scripts for all figures
│       └── spectral_bias/   # Spectral bias analysis scripts
├── data/
│   ├── datasets/            # PDE solution datasets (not tracked in git)
│   ├── nets/                # Trained model checkpoints (not tracked in git)
│   ├── sb_data/             # Spectral bias analysis data (not tracked in git)
│   └── fourier_data/        # Fourier coefficient data
├── figures/
│   ├── pngs/                # Output figures (PNG)
│   └── pdfs/                # Output figures (PDF)
├── produce_all_figures.py   # Master script to regenerate all publication figures
├── list_needed_nets.py      # Lists all trained networks required for reproducibility
└── colors_cb.txt            # Colorblind-friendly color palette
```

## Training Models

### Single training run

Run `src/execute_don.py` from the repository root. It must be called from `data/` or with the nets directory accessible. The script takes 18 positional arguments:

```bash
cd data
python ../src/execute_don.py \
  <Nepochs> <vtag> <depth> <width> <llw> <doplot> \
  <batch_name> <lrstag> <init_lr> <decay_rate> \
  <num_data> <which_T> <dotruesigma> <uendtag> \
  <sigmascale> <exponent> <doadam> <dostacked>
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `Nepochs` | Number of training epochs |
| `vtag` | Random seed (version tag) |
| `depth`, `width` | Network depth and width |
| `llw` | Truncation rank (number of modes) |
| `batch_name` | Dataset name (e.g. `kdvnx401_dt0.0001_nc5_m5000`) |
| `which_T` | Trunk type: `-1`=learned, `0`=SVD, `1`=Legendre, `2`=Chebyshev, `7`=trigonometric, etc. |
| `dotruesigma` | `1` to scale by singular values, `0` for uniform scaling |
| `sigmascale` | Scaling factor or `"First"` (normalize by largest singular value) |
| `exponent` | Exponent for mode-dependent loss weighting |
| `doadam` | `1` for Adam, `0` for SGD |
| `dostacked` | `1` for stacked DeepONet, `0` for standard |

Example (SVD trunk, Adam, KdV dataset):

```bash
cd data
python ../src/execute_don.py 4000 0 5 335 50 0 \
  kdvnx401_dt0.0001_nc5_m5000 32 1e-4 0.95 \
  1000 0 1 1999 1.0 0.0 1 0
```

Trained models are saved under `data/nets/<stem>/` containing:
- `lambdas.txt` — model parameters
- `log.txt` — training log
- `errorcurve.png` — loss curves

### Batch training

Use `src/run.py` to launch multiple runs. Pass a mode integer to select which group of experiments to run:

```bash
python -m src.run <mode>
```

Use `mode=-1` to run all experiments. See `src/run.py` for the available modes and their configurations.

## Reproducing Figures

First ensure all required trained networks are present in `data/nets/`. You can check which networks are needed by running:

```bash
python list_needed_nets.py
```

Then generate all figures at once:

```bash
python produce_all_figures.py
```

To regenerate a specific figure, edit `produce_all_figures.py` and set `take_keys` to only the desired figure key, or run the corresponding command directly. The mapping is:

| Key | Figure | Command |
|-----|--------|---------|
| `1_top_right` | Fig. 1 (top right) | `python -m src.analysis.RELEVANT.analyze_whichTs_ga` |
| `1_bottom_right` | Fig. 1 (bottom right) | `python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses_ga` |
| `2` | Fig. 2 | `python -m src.analysis.RELEVANT.analyze_whichTs_newlayout2 3 40 -1 d5_w100` |
| `3` | Fig. 3 | `python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2 3 1 1 2` |
| `4` | Fig. 4 | `python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 4 1 32` |
| `5` | Fig. 5 | `python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2 3 0 1 2` |
| `6` | Fig. 6 | `python src/analysis/spectral_bias/plot_res3_sidebyside_mat_gridspec.py 0.2` |
| `7` | Fig. 7 | `python -m src.analysis.RELEVANT.investigate_branch_sb_scale2 2 1` |
| `8` | Fig. 8 | `python -m src.analysis.RELEVANT.analyze_mode_losses_rotate2 3 0 2 32` |
| `9a` | Fig. 9a | `python -m src.analysis.RELEVANT.show_components_mult_multsizes 3 -1 4000 1` |
| `9b` | Fig. 9b | `python -m src.analysis.RELEVANT.show_components_2x2 3 0 3 4000 10000` |
| `10` | Fig. 10 | `python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 5 1 32` |
| `11` | Fig. 11 | `python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 11 2 32` |
| `12` | Fig. 12 | `python src/analysis/RELEVANT/synth_freq_comp_fromFILES.py` |

All scripts should be run from the repository root.

## Data

The large data directories are not tracked in git due to their size (~20 GB total). They are hosted on HuggingFace and managed with two helper scripts.

| Directory | Size | Contents |
|-----------|------|----------|
| `data/datasets/` | ~229 MB | PDE solution datasets |
| `data/sb_data/` | ~49 MB | Spectral bias analysis data |
| `data/nets/` | ~20 GB | Trained model checkpoints |

### Downloading data (`hf_pull.py`)

After cloning the repository, fetch the data with:

```bash
pip install huggingface_hub
python hf_pull.py username/deeponet-data
```

To skip the 20 GB model checkpoints and only download the datasets needed for training or re-running analysis from scratch:

```bash
python hf_pull.py username/deeponet-data --subset datasets sb_data
```

Files are placed directly into the correct locations under `data/` in the repository root.

**Options:**

| Flag | Description |
|------|-------------|
| `--subset nets datasets sb_data` | Choose which subdirectories to download (default: all three) |
| `--token <hf_token>` | HuggingFace API token; can also be set via the `HF_TOKEN` environment variable |

The script prints a per-directory file count after downloading so you can verify completeness.


### Dataset format

The PDE datasets used are numerical solutions to:

- **Advection-diffusion** (`advdiffnx201_*`): 1D advection-diffusion equation, 201 spatial points
- **KdV** (`kdvnx401_*`): Korteweg-de Vries equation, 401 spatial points
- **Burgers** (`burgers_*`): Burgers equation

Each dataset consists of three text files:
- `<name>_U.txt` — solution snapshots (output functions)
- `<name>_P.txt` — input parameters/initial conditions
- `<name>_R.txt` — spatial domain points
