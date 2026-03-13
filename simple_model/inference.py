#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import os
current_dir = os.getcwd()
if 'simple_model' not in current_dir:
    current_dir+='/simple_model'
import argparse
import numpy as np



import jax
import jax.numpy as jnp
import jax.random as random
jax.config.update('jax_enable_x64', True)
import numpyro
numpyro.set_host_device_count(4)
# ... rest of your imports (utils, models, etc.)
import os
import argparse
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow

from utils import set_backend, sample_redshifts
from numpyro_models import sample_model, mcmc_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--name", type=str, default='base_name')
    parser.add_argument("--nn_width", type=int, default=32)
    parser.add_argument("--nn_depth", type=int, default=2)
    parser.add_argument("--no_flows", type=int, default=4)
    parser.add_argument('--model_type', type=str, default='flow')
    parser.add_argument('--cmb', action='store_true', help='CMB prior')
    parser.add_argument('--lcdm', action='store_true', help='CMB prior')
    args = parser.parse_args()

    if args.model_type == 'flow' and args.name == 'base_name':
        raise FileNotFoundError('You need to provide the name of the weights with "--name weights_name"')
    # --- Config ---
    set_backend("jax")

    std_norm = True
    if args.lcdm:
        wCDM_bool = False
    else:
        wCDM_bool = True
    n_samples = 11500

    # --- CMB Prep ---
    vary_cmb = True
    sigma_Rcmb = 0.007
    R_cmb_obs_default = 1.7579698042257326
    
    if vary_cmb:
        key = random.PRNGKey(args.rep)  
        key_cmb, _ = random.split(key)
        cmb_draw = random.normal(key_cmb)
        R_cmb_obs = R_cmb_obs_default + cmb_draw * sigma_Rcmb
    else:
        R_cmb_obs = R_cmb_obs_default

    cmb_kwargs = {"cmb_bool": args.cmb, "sigma_Rcmb": sigma_Rcmb, "R_cmb_obs": R_cmb_obs}

    # --- Flow Model Setup ---
    flow_kwargs = {}
    if args.model_type == 'flow':
        if std_norm:
            file_ = np.load(current_dir+'/scaling/'+args.name + '_std.npz')
            mu_, std_ = file_['mu'], file_['std']
            add_ = jnp.sum(jnp.log(std_[:3]))
        else:
            file_ = np.load(current_dir+'/scaling/'+args.name + '_minmax.npz')
            min_, max_ = file_['min'], file_['max']
            add_ = jnp.sum(jnp.log(max_ - min_)[:3])

        key, _ = random.split(random.PRNGKey(2))
        skel_flow = masked_autoregressive_flow(
            key=key, base_dist=Normal(jnp.zeros(3)), cond_dim=15, nn_activation=jax.nn.gelu,
            flow_layers=args.no_flows, nn_width=args.nn_width, nn_depth=args.nn_depth
        )
        flow_model = eqx.tree_deserialise_leaves(current_dir+'/weights/'+ args.name + '_weights.eqx', skel_flow)
        
        flow_kwargs = {'flow_model': flow_model, 'mu_': mu_, 'std_': std_, 'add_': add_}

    # --- Data Simulation ---
    print(f"Running seed: {args.rep}")
    rng_key = random.PRNGKey(args.rep)
    rng_key, rng_key_1 = random.split(rng_key)
    z_s_ = sample_redshifts(n_samples, rng_key_1, Om0=0.315, w0=-1, beta_rate=1.5)

    prior_predictive = Predictive(sample_model, num_samples=1)
    rng_key, rng_key_2 = random.split(rng_key)
    prior_predictions = prior_predictive(rng_key_2, z_s_)

    rng_key, rng_key_3 = random.split(rng_key)
    z_pec = random.normal(rng_key_3, (n_samples,)) * (300 / 299792.458)
    z_s = (1 + z_s_) * (1 + z_pec) - 1

    sel_sim = prior_predictions["sel_s"][0, :]
    sel_sim_mask = np.logical_and(sel_sim == 1, np.logical_and(z_s > 0.05, z_s < 1.1))
    
    d_hat_sim = prior_predictions["d_hat_s"][0, :, :]
    d_s = d_hat_sim[sel_sim_mask, :]
    
    d_err_s = jnp.array([
        jnp.exp(prior_predictions["log_mag_err_s"][0, :][sel_sim_mask])**2,
        jnp.exp(prior_predictions["log_c_err_s"][0, :][sel_sim_mask])**2,
        jnp.exp(prior_predictions["log_x_err_s"][0, :][sel_sim_mask])**2,
        prior_predictions["cov_m_c"][0, :][sel_sim_mask],
        prior_predictions["cov_m_x"][0, :][sel_sim_mask],
        prior_predictions["cov_c_x"][0, :][sel_sim_mask]
    ]).T

    z_s = z_s[sel_sim_mask]
    print(f"Selected Samples: {len(z_s)}")

    # --- MCMC Setup & Run ---
    init_dict = {'w': -1.1, 'Om0': 0.35, 'Omde': 0.65, 'M0': -19.35, 'c0': -0.04, 'x0': -0.4, 
                 'sigma_res': 0.1, 'sigma_c': 0.05, 'sigma_x': 1, 'alpha': -0.11, 'beta': 3.0, 'alpha_c': -0.01}

    nuts_kernel = NUTS(mcmc_model, adapt_step_size=True, max_tree_depth=7, init_strategy=init_to_value(values=init_dict))
    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500, num_chains=4)
    
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, z_s, data_s=d_s, data_err_s=d_err_s, wCDM=wCDM_bool, 
             model_type=args.model_type, flow_kwargs=flow_kwargs, cmb_kwargs=cmb_kwargs)
    
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()

    # --- Save Output ---
    if args.model_type != 'flow':
        save_name = args.model_type + '_cmb' if args.cmb else args.model_type
    else:
        save_name = args.name + '_cmb_flow' if args.cmb else args.name+'_flow'
    
    dir_out = f"{current_dir}/chains/{save_name}_chains"
    os.makedirs(dir_out, exist_ok=True)

    prefix = 'w' if wCDM_bool else 'l'
    np.savez(os.path.join(dir_out, f"{prefix}{args.rep}.npz"), **posterior_samples)

if __name__ == "__main__":
    main()