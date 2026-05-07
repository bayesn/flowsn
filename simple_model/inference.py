#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

jax.config.update("jax_enable_x64", True)
import numpyro

numpyro.set_host_device_count(4)

import yaml
import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

from utils import set_backend, sample_redshifts
from numpyro_models import sample_model, mcmc_model

MODEL_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep",        type=int, default=0)
    parser.add_argument("--name",       type=str, default="base_name")
    parser.add_argument("--model_type", type=str, default="flow")
    parser.add_argument("--cmb",   action="store_true", help="CMB shift-parameter prior")
    parser.add_argument("--lcdm",  action="store_true", help="Flat LCDM (fit Omde instead of w)")
    parser.add_argument("--wa",    action="store_true", help="Vary dark-energy wa parameter")
    parser.add_argument("--gamma", type=float, default=0.0, help="True host-mass step gamma injected into simulation (0 = no mass step)")
    args = parser.parse_args()

    if args.model_type == "flow" and args.name == "base_name":
        raise FileNotFoundError(
            'Provide the model name with "--name weights_name"'
        )

    set_backend("jax")

    wCDM_bool = not args.lcdm

    # --- CMB prep ---
    vary_cmb = True
    sigma_Rcmb = 0.007
    R_cmb_obs_default = 1.7579698042257326

    if vary_cmb:
        key = random.PRNGKey(args.rep)
        key_cmb, _ = random.split(key)
        R_cmb_obs = float(R_cmb_obs_default + random.normal(key_cmb) * sigma_Rcmb)
    else:
        R_cmb_obs = R_cmb_obs_default

    # --- Flow model setup ---
    flow_kwargs = {}
    if args.model_type == "flow":
        arch_path = MODEL_DIR / "weights" / (args.name + "_arch.yml")
        with open(arch_path) as f:
            arch = yaml.safe_load(f)

        file_ = np.load(MODEL_DIR / "scaling" / (args.name + "_std.npz"))
        mu_, std_ = file_["mu"], file_["std"]
        add_ = jnp.sum(jnp.log(std_[:3]))

        key, _ = random.split(random.PRNGKey(2))
        skel_flow = masked_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros(3)),
            cond_dim=15,
            nn_activation=jax.nn.gelu,
            flow_layers=arch["no_flows"],
            nn_width=arch["nn_width"],
            nn_depth=arch["nn_depth"],
        )
        flow_model = eqx.tree_deserialise_leaves(
            str(MODEL_DIR / "weights" / (args.name + ".eqx")), skel_flow
        )
        flow_kwargs = {"flow_model": flow_model, "mu_": mu_, "std_": std_, "add_": add_}

    # --- Simulate test data ---
    print(f"Running seed: {args.rep}")
    n_samples = 11500
    rng_key = random.PRNGKey(args.rep)
    rng_key, rng_key_1 = random.split(rng_key)
    z_s_ = sample_redshifts(n_samples, rng_key_1, Om0=0.315, w0=-1, beta_rate=1.5)

    rng_key, rng_key_2 = random.split(rng_key)
    z_pec = random.normal(rng_key_2, (n_samples,)) * (300 / 299792.458)
    z_s = (1 + z_s_) * (1 + z_pec) - 1

    prior_predictive = Predictive(sample_model, num_samples=1)
    rng_key, rng_key_3 = random.split(rng_key)
    prior_predictions = prior_predictive(rng_key_3, z_s_, z_pec, gamma=args.gamma)

    sel_sim = prior_predictions["sel_s"][0, :]
    sel_sim_mask = np.logical_and(sel_sim == 1, np.logical_and(z_s > 0.05, z_s < 1.1))

    d_s = prior_predictions["d_hat_s"][0, :, :][sel_sim_mask, :]
    mass_sim = np.array(prior_predictions["log_mass"][0, :][sel_sim_mask])

    d_err_s = jnp.array(
        [
            jnp.exp(prior_predictions["log_mag_err_s"][0, :][sel_sim_mask]) ** 2,
            jnp.exp(prior_predictions["log_c_err_s"][0, :][sel_sim_mask]) ** 2,
            jnp.exp(prior_predictions["log_x_err_s"][0, :][sel_sim_mask]) ** 2,
            prior_predictions["cov_m_c"][0, :][sel_sim_mask],
            prior_predictions["cov_m_x"][0, :][sel_sim_mask],
            prior_predictions["cov_c_x"][0, :][sel_sim_mask],
        ]
    ).T

    z_s = z_s[sel_sim_mask]
    print(f"Selected samples: {len(z_s)}")

    # --- MCMC ---
    init_dict = {
        "w": -1.1,
        "Om0": 0.35,
        "Omde": 0.65,
        "M0": -19.35,
        "c0": -0.04,
        "x0": -0.4,
        "sigma_res": 0.1,
        "sigma_c": 0.05,
        "sigma_x": 1,
        "alpha": -0.11,
        "beta": 3.0,
        "alpha_c": -0.01,
    }
    gamma_bool = args.gamma != 0.0
    if args.wa:
        init_dict["wa"] = 0.0
    if gamma_bool:
        init_dict["gamma"] = 0.0

    nuts_kernel = NUTS(
        mcmc_model,
        adapt_step_size=True,
        max_tree_depth=7,
        init_strategy=init_to_value(values=init_dict),
    )
    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500, num_chains=4)

    mcmc.run(
        random.PRNGKey(0),
        z_s,
        data_s=d_s,
        data_err_s=d_err_s,
        wCDM=wCDM_bool,
        wa_bool=args.wa,
        gamma_bool=gamma_bool,
        mass=mass_sim,
        model_type=args.model_type,
        flow_kwargs=flow_kwargs,
        cmb_bool=args.cmb,
        R_cmb_obs=R_cmb_obs,
        sigma_Rcmb=sigma_Rcmb,
    )

    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()

    # --- Save ---
    suffix = ("_cmb" if args.cmb else "") + ("_wa" if args.wa else "") + ("_gamma" if gamma_bool else "")
    if args.model_type != "flow":
        save_name = args.model_type + suffix
    else:
        save_name = args.name + suffix + "_flow"

    dir_out = MODEL_DIR / "chains" / f"{save_name}_chains"
    dir_out.mkdir(parents=True, exist_ok=True)

    prefix = "w" if wCDM_bool else "l"
    np.savez(dir_out / f"{prefix}{args.rep}.npz", **posterior_samples)


if __name__ == "__main__":
    main()
