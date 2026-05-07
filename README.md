# FlowSN: Neural Simulation-Based Inference under Realistic Selection Effects applied to Supernova Cosmology

Code from [Boyd et al. (2026)](https://arxiv.org/abs/2603.11165).

## Environment Setup

```bash
# Perlmutter (NERSC) only
module load python
module load cudatoolkit/12.4

conda env create -f environment.yml
conda activate flowsn_env
```

---

## Simple Model Code Overview

Conda dependencies in `environment.yml`.

---

All scripts can be run from the repo root or any other directory — paths are resolved relative to the script locations.

### 1. Data Generation (`simple_model/generate_training.py`)

Generates 20 million synthetic supernovae (18M half-normal + 2M uniform magnitude prior) and saves them as a single `.npy` file.

```bash
python simple_model/generate_training.py --name training_data
```

**Output:** `simple_model/training_data/training_data.npy` — 20M rows, 18 columns: observed $(m, c, x)$ (cols 0–2) + 15 latent parameters/errors.

---

### 2. Training (`simple_model/train.py`)

Trains a `MaskedAutoregressiveFlow` on the residuals with a physical log-Jacobian correction.

```bash
python simple_model/train.py --data training_data --name sn_model --epochs 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Training file name (with or without `.npy`) |
| `--name` | `sn_model` | Name used for output files |
| `--nn_width` | 32 | Hidden layer width |
| `--nn_depth` | 2 | Hidden layers per MAF block |
| `--no_flows` | 4 | Number of MAF layers |
| `--batch_size` | 8192 | Minibatch size |
| `--epochs` | 50 | Training epochs |

**Outputs:**
- `simple_model/scaling/<name>_std.npz` — standardisation statistics
- `simple_model/weights/<name>.eqx` — trained model weights
- `simple_model/weights/<name>_arch.yml` — architecture config (read automatically at inference)

---

### 3. Inference (`simple_model/inference.py`)

Simulates a synthetic SN dataset (using the forward model), then runs 4-chain NUTS MCMC. Chains are saved to `simple_model/chains/<save_name>_chains/`.

**wCDM with flow likelihood:**

```bash
python simple_model/inference.py --model_type flow --name sn_model --rep 0
```

**Analytical or naive likelihood:**

```bash
python simple_model/inference.py --model_type analytical --rep 0
python simple_model/inference.py --model_type naive --rep 0
```

**Flat ΛCDM (fit Ω_de instead of w):**

```bash
python simple_model/inference.py --model_type flow --name sn_model --lcdm --rep 0
```

**Include CMB shift-parameter prior:**

```bash
python simple_model/inference.py --model_type flow --name sn_model --cmb --rep 0
```

**Vary dark-energy equation-of-state evolution (w + w_a):**

```bash
python simple_model/inference.py --model_type flow --name sn_model --wa --rep 0
```

**Fit host-galaxy mass step γ:**

```bash
python simple_model/inference.py --model_type flow --name sn_model --gamma --rep 0
```

**Flags combine freely**, e.g.:

```bash
python simple_model/inference.py --model_type flow --name sn_model --cmb --gamma --rep 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--rep` | 0 | Random seed / realisation index |
| `--name` | `base_name` | Model name; must match files saved in Step 2 |
| `--model_type` | `flow` | `flow`, `analytical`, or `naive` |
| `--lcdm` | off | Fit flat ΛCDM (Ω_de free) instead of wCDM |
| `--cmb` | off | Add Gaussian prior on CMB shift parameter R |
| `--wa` | off | Vary dark-energy w_a (uses jax-cosmo backend) |
| `--gamma` | off | Fit host-galaxy mass step γ (threshold at log M = 10) |

**Output chain naming** — suffix encodes flags used:

| Flags | `save_name` | Output prefix |
|-------|-------------|---------------|
| (none) | `sn_model_flow` | `w{rep}.npz` |
| `--lcdm` | `sn_model_flow` | `l{rep}.npz` |
| `--cmb` | `sn_model_cmb_flow` | `w{rep}.npz` |
| `--wa --gamma` | `sn_model_wa_gamma_flow` | `w{rep}.npz` |

---

## SNANA Experiments

All scripts can be run from the repo root (`flowsn/`) or any other directory — paths are resolved relative to the script locations.

### Required data layout

The SNANA FITRES files are expected one level above this repo:

```
clauding/
├── flowsn/                         ← this repo
│   └── SNANA_experiments/
└── SNANA_files/
    ├── training_files/
    │   ├── TRAINING1.FITRES
    │   ├── TRAINING2.FITRES
    │   ├── TRAINING3.FITRES
    │   └── TRAINING_FLAT.FITRES
    └── test_files/
        ├── cosmo1/
        │   ├── TESTING1.FITRES … TESTING100.FITRES
        ├── cosmo2/
        │   └── …
        └── cosmo3/
            └── …
```

---

### Step 1 — Build the training set (`SNANA_make_training.py`)

Reads the four FITRES training files, applies JLA-style selection cuts, transforms columns according to `SNANA_KEYS`, and saves a single NumPy array.

```bash
python SNANA_experiments/SNANA_make_training.py
```

**Output:** `SNANA_experiments/SNANA_training_set.npy` — shape `(N, 15)`.

Column layout:

| Index | Content |
|-------|---------|
| 0 | `zHEL` |
| 1 | `zHD` |
| 2 | `zHDERR` |
| 3 | `mB` |
| 4 | `c` |
| 5 | `x1` |
| 6 | `mBERR` |
| 7 | `cERR` |
| 8 | `x1ERR` |
| 9 | `COV_c_x0` (scaled by −2.5/ln10 / x0) |
| 10 | `COV_x1_x0` (scaled by −2.5/ln10 / x0) |
| 11 | `COV_x1_c` |
| 12 | `SIM_DLMAG + SIM_MUSHIFT + TEMPLATE_M0` |
| 13 | `−SIM_alpha` |
| 14 | `SIM_beta` |
| 15 | `log(M_stellar)` — host-galaxy log-mass (randomly sampled for this work) |

---

### Step 2 — Train the normalising flow (`SNANA_train.py`)

Trains a masked autoregressive flow (MAF) on the training set. Standardisation statistics and model weights are saved to `SNANA_experiments/flow_weights/`.

```bash
python SNANA_experiments/SNANA_train.py \
    --name my_model \
    --nn_width 32 \
    --nn_depth 2 \
    --no_flows 4 \
    --epochs 200 \
    --batch_size 8092 \
    --val_prop 0.1 \
    --max_pat 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | auto-generated | Name used for output files |
| `--nn_width` | 32 | Hidden layer width in each MAF block |
| `--nn_depth` | 2 | Number of hidden layers per MAF block |
| `--no_flows` | 4 | Number of MAF layers |
| `--epochs` | 10 | Maximum training epochs |
| `--batch_size` | 8092 | Minibatch size |
| `--val_prop` | 0.1 | Fraction of data held out for validation |
| `--max_pat` | 10 | Early-stopping patience (epochs without val improvement) |
| `--save_all` | off | Save a checkpoint every 10 epochs (default: only the best) |

**Outputs:**
- `SNANA_experiments/flow_weights/<name>.eqx` — trained model weights
- `SNANA_experiments/flow_weights/<name>_std.npz` — standardisation statistics (`mu`, `std`)
- `SNANA_experiments/flow_weights/<name>_arch.yml` — architecture config (read automatically at inference)

---

### Step 3 — Build test sets (`SNANA_make_testing.py`)

Processes 100 FITRES test files for a given cosmology and saves individual `.npy` arrays. Run once for each of the three cosmologies.

```bash
python SNANA_experiments/SNANA_make_testing.py --cosmo 1
python SNANA_experiments/SNANA_make_testing.py --cosmo 2
python SNANA_experiments/SNANA_make_testing.py --cosmo 3
```

**Output:** `SNANA_experiments/testing_sets/cosmo{N}/SNANA_test{0..99}.npy` — shape `(M, 16)`. Columns 0–14 match the training set; column 15 is a uniform fake host-galaxy mass in [8, 12].

---

### Step 4 — Inference (`infer.py`)

Runs 4-chain NUTS MCMC (500 warmup + 500 samples per chain) on a single test realisation using the trained flow as the SN likelihood. All chains are saved to `SNANA_experiments/SNANA_chains_<name><suffix>/`.

**Basic wCDM run (fit *w* and Ω_m):**

```bash
python SNANA_experiments/infer.py \
    --name my_model \
    --rep 0 \
    --cosmo 1
```

**wCDM with CMB shift-parameter prior:**

```bash
python SNANA_experiments/infer.py --name my_model --rep 0 --cosmo 1 --cmb
```

**Flat ΛCDM (fit Ω_m and Ω_de instead of *w*):**

```bash
python SNANA_experiments/infer.py --name my_model --rep 0 --cosmo 1 --lcdm
```

**wCDM + evolving dark energy (vary *w_a*):**

```bash
python SNANA_experiments/infer.py --name my_model --rep 0 --cosmo 1 --wa
```

**Include host-mass nuisance parameter γ:**

```bash
python SNANA_experiments/infer.py --name my_model --rep 0 --cosmo 1 --gamma
```

**Flags can be combined freely**, e.g. CMB prior + host mass:

```bash
python SNANA_experiments/infer.py --name my_model --rep 0 --cosmo 2 --cmb --gamma
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--rep` | 0 | Test-set index (0–99) |
| `--name` | `paper` | Model name; must match the files saved in Step 2 |
| `--cosmo` | 1 | Cosmology ID (1, 2, or 3) |
| `--cmb` | off | Add a Gaussian prior on the CMB shift parameter R |
| `--lcdm` | off | Use flat ΛCDM (fit Ω_de) instead of wCDM (fit w) |
| `--wa` | off | Vary the dark-energy equation-of-state evolution parameter w_a |
| `--gamma` | off | Fit a host-galaxy mass step γ |

**Outputs** (saved in `SNANA_experiments/SNANA_chains_<name>[_cmb]_cosmo<N>/`):

- wCDM: `wflow_SNANA{rep}.npz` with keys `w`, `Om0`, `M0`, `alpha`, `beta`
- ΛCDM: `lflow_SNANA{rep}.npz` with keys `Om0`, `Omde`, `M0`, `alpha`, `beta`

