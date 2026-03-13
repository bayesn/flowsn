# FlowSN: Neural Simulation-Based Inference under Realistic Selection Effects applied to Supernova Cosmology

Code from [Boyd et al. (2026)](https://arxiv.org/abs/2603.11165).

## Simple Model Code Overview

Conda dependencies in `environment.yml`.

---

### 1. Data Generation (`simple_model/generate_data.py`)
Generates 20 million synthetic supernovae using a mixture of priors and saves them as a single, contiguous `.npy` file.

* **Usage:** `python simple_model/generate_data.py --name training_data`
* **Output:** `simple_model/training_data/training_data.npy` (20M rows).
* **Data Structure:** 18 columns total: [Observed $m, c, x$ (3 cols)] + [Latent parameters/errors (15 cols)].


---

### 2. Training (`simple_model/train.py`)
Trains a `MaskedAutoregressiveFlow` on the residuals. It applies a physical log-Jacobian correction to ensure the learned density is consistent with the original data units.

* **Usage:** `python simple_model/train.py --data training_data --name sn_model --epochs 100`
* **Key Features:**
    * **Standardszation:** Automatically saves `simple_model/scalings/model_std.npz` (contains $\mu$ and $\sigma$ for scaling).
    - Baseline$ (using columns 3–5).
    * **Training Weights:** Automatically saves `simple_model/weights/sn_model_weights.eqx` (saves training weights).

---

### 3. Inference (`simple_model/inference.py`)
Computes the log-likelihood of new supernova observations.

* **Usage:** `python simple_model/inference.py --model_type flow --name sn_model`
* **Process:**
    1. Loads the `.eqx` model weights.
    2. Applies the $\mu$ and $\sigma$ constants from the `.npz` file to scale the input.
    3. Posterior sampling of cosmology and SN parameters/
    4. Chains saved in `simple_model/chains/sn_model_flow_chains/`.

* **Usage:** `python simple_model/inference.py --model_type analytical`
* **Process:**
    1. Posterior sampling of cosmology and SN parameters using analytical model.
    2. Chains saved in `simple_model/chains/analytical_chains/`.

* **Usage:** `python simple_model/inference.py --model_type naive`
* **Process:**
    1. Posterior sampling of cosmology and SN parameters using naive model.
    2. Chains saved in `simple_model/chains/naive_chains/`.


* **Usage:** `python simple_model/inference.py --model_type flow --name sn_model --rep 3 --cmb`
* **Process:**
    1. Posterior sampling of cosmology and SN parameters using flow. Include a CMB prior and change the random seed.
    2. Chains saved in `simple_model/chains/sn_model_cmb_flow_chains/`.

* **Usage:** `python simple_model/inference.py --model_type flow --name sn_model --lcdm --rep 5`
* **Process:**
    1. Posterior sampling of different non-flat LCDM cosmology.
    2. Chains saved in `simple_model/chains/sn_model_cmb_flow_chains/l5.npz`.


---

## SNANA Experiments

Python code used in SNANA experiments may be found in  `SNANA_experiments/`.

