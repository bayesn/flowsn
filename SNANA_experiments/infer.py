#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import wcosmo
from importlib import import_module
from numpyro.infer import MCMC, NUTS, init_to_value

from utils import (
    STD_NORM, VARY_CMB, SIGMA_RCMB,
    R_CMB_OBS_DEFAULT,
    load_normalisation, parse_test_set,
)
from flow_dist import load_flow
from numpyro_model import numpyro_model


# =========================
# Setup
# =========================

def setup():
    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(4)

    # Point wcosmo at JAX backends
    setattr(wcosmo.wcosmo, "xp", import_module("jax.numpy"))
    setattr(wcosmo.utils,  "xp", import_module("jax.numpy"))
    setattr(wcosmo.utils, "toeplitz", import_module("jax.scipy.linalg").toeplitz)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep",       type=int,  default=0)
    parser.add_argument("--name",      type=str,  default="paper")
    parser.add_argument("--nn_width",  type=int,  default=32)
    parser.add_argument("--nn_depth",  type=int,  default=2)
    parser.add_argument("--no_flows",  type=int,  default=4)
    parser.add_argument("--gamma",       action="store_true")
    parser.add_argument("--cmb",       action="store_true")
    parser.add_argument("--lcdm",       action="store_true")
    parser.add_argument("--wa",       action="store_true")
    parser.add_argument("--cosmo",  type=int,  default = 1)
    return parser.parse_args()


# =========================
# Run management
# =========================

def find_pending_runs(directory, total=100):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    done  = {int(f[11:][:-4]) for f in files}
    return [r for r in range(total) if r not in done]


def load_test_set(rep, cosmo):
    path = (
        f"./testing_sets/cosmo{str(cosmo)}/SNANA_test{rep}.npy"
    )
    X = np.load(path)
    X = X[np.logical_and(X[:, 1] > 0.05, X[:, 1] < 1.1)]
    print(f"  Loaded test set {rep}: {X.shape}")
    return X


# =========================
# Main
# =========================

def main():
    setup()
    args = parse_args()

    if args.lcdm:
        WCDM_BOOL = False
    else:
        WCDM_BOOL = True

    # --- CMB obs value ---
    R_cmb_obs = R_CMB_OBS_DEFAULT[args.cosmo]
    if VARY_CMB:
        key_cmb, _ = jr.split(jr.PRNGKey(args.rep))
        R_cmb_obs  = R_cmb_obs + float(jax.random.normal(key_cmb)) * SIGMA_RCMB

    # --- Load flow + normalisation ---
    norm_params = load_normalisation(args.name, std_norm=STD_NORM)
    flow        = load_flow(args.name, args.no_flows, args.nn_width, args.nn_depth)

    print(f"\nLoaded flow : {args.name}.eqx")
    print(f"JAX devices : {jax.devices()}")

    # --- Output directory ---
    suffix   = ("_cmb" if args.cmb else "") + ("_cosmo"+str(args.cosmo))
    run_name = args.name + suffix
    out_dir  = f"SNANA_chains_{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    save_labels = ["w", "Om0", "M0", "alpha", "beta"] if WCDM_BOOL else ["Om0", "Omde", "M0", "alpha", "beta"]
    prefix      = "wflow" if WCDM_BOOL else "lflow"



    print(f"{'='*50}")
    print(f"  Run {args.rep}")
    print(f"{'='*50}")

    X    = load_test_set(args.rep, args.cosmo)
    data = parse_test_set(X)

    init_dict = {
        "w0": -0.9, "wa": 0.0, "Om0": 0.3, "Omde": 0.65,
        "M0": -19.36, "alpha": -0.12, "beta": 3.0,
        "eps": jnp.zeros(len(data["z_hd"])),
    }

    nuts_kernel = NUTS(
        numpyro_model,
        adapt_step_size=True,
        max_tree_depth=7,
        init_strategy=init_to_value(values=init_dict),
    )
    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500, num_chains=4)
    mcmc.run(
        jr.PRNGKey(0),
        data["z_hd"], data["z_hd_err"], data["z_hel"],mass = data["mass"],
        data_s=data["d_hat"],
        data_err_s=data["d_err"],
        gamma_bool = args.gamma,
        wCDM=WCDM_BOOL,
        wa_bool = args.wa,
        cmb_bool=args.cmb,
        R_cmb_obs=float(R_cmb_obs),
        flow=flow,
        scale_mu=norm_params['mu'],
        scale_std= norm_params['std']

    )
    mcmc.print_summary()

    posterior = mcmc.get_samples()
    out_path  = f"{out_dir}/{prefix}_SNANA{args.rep}.npz"
    np.savez(out_path, **{k: posterior[k] for k in save_labels})
    print(f"  Saved -> {out_path}\n")


if __name__ == "__main__":
    main()