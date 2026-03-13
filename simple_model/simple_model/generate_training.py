#!/usr/bin/env python
import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
import numpy as onp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def sample_model(n_sne, m0_prior="halfnormal", m_cut=24.0, sigma_cut=0.25, a=-0.1, b=-1.0):
    with numpyro.plate("plate_i", n_sne):
        # Priors
        if m0_prior == "halfnormal": m0 = 25.0 - numpyro.sample("m0", dist.HalfNormal(0.5))
        else: m0 = numpyro.sample("m0", dist.Uniform(17, m_cut))
        alpha, beta, alpha_c = numpyro.sample("alpha", dist.Uniform(-0.2, -0.1)), numpyro.sample("beta", dist.Uniform(2.5, 3.5)), numpyro.sample("alpha_c", dist.Uniform(-0.02, 0.0))
        c0, x0 = numpyro.sample("c0", dist.Uniform(-0.1, 0.0)), numpyro.sample("x0", dist.Uniform(-0.6, -0.2))
        sigma_int, sigma_c, sigma_x = numpyro.sample("sigma_int", dist.Uniform(1e-4, 0.2)), numpyro.sample("sigma_c", dist.Uniform(1e-4, 0.1)), numpyro.sample("sigma_x", dist.Uniform(1e-4, 1.5))
        
        # Supernova Physical Model
        x_s = numpyro.sample("x_s", dist.Normal(x0, sigma_x))
        c_s = numpyro.sample("c_s", dist.Normal(c0 + alpha_c * x_s, sigma_c))
        m_s = numpyro.sample("m_s", dist.Normal(m0 + alpha * x_s + beta * c_s, sigma_int))

        # Observational Errors
        log_x_err, log_c_err, log_mag_err = numpyro.sample("log_x_err_s", dist.Normal(-1.5, 0.5)), numpyro.sample("log_c_err_s", dist.Normal(-3.5, 0.3)), numpyro.sample("log_mag_err_s", dist.Normal(0.1 * (m_s - 56), 0.6))
        
        # Covariances
        cov_m_c = numpyro.sample("cov_m_c", dist.Normal(0.000520 + 0.288792999 * jnp.exp(log_mag_err)**2, jnp.exp(log_mag_err)**2 * 0.125979))
        cov_m_x = numpyro.sample("cov_m_x", dist.Normal(0.0008097 + 0.03835550 * jnp.exp(log_x_err)**2, jnp.exp(log_x_err)**2 * 0.02032))
        cov_c_x = numpyro.sample("cov_c_x", dist.Normal(0.000168655 + 0.01358504 * jnp.exp(log_x_err)**2, jnp.exp(log_x_err)**2 * 0.011840861))

        d_hat = numpyro.sample("d_hat_s", dist.MultivariateNormal(jnp.column_stack((m_s, c_s, x_s)), jnp.stack([
            jnp.column_stack((jnp.exp(log_mag_err)**2, cov_m_c, cov_m_x)),
            jnp.column_stack((cov_m_c, jnp.exp(log_c_err)**2, cov_c_x)),
            jnp.column_stack((cov_m_x, cov_c_x, jnp.exp(log_x_err)**2))
        ], axis=1)))
        
        # Selection Bias
        p_s = norm.cdf(-(d_hat[:, 0] + a * d_hat[:, 2] + b * d_hat[:, 1]), loc=-m_cut, scale=sigma_cut)
        numpyro.sample("sel_s", dist.Bernoulli(p_s))

def build_training_rows(pred):
    sel = (pred["sel_s"][0] == 1)
    d_hat = pred["d_hat_s"][0][sel]
    params = jnp.column_stack((pred["m0"][0][sel], pred["c0"][0][sel], pred["x0"][0][sel], jnp.log(pred["sigma_int"][0][sel]), jnp.log(pred["sigma_c"][0][sel]), jnp.log(pred["sigma_x"][0][sel]), pred["alpha"][0][sel], pred["beta"][0][sel], pred["alpha_c"][0][sel], pred["log_mag_err_s"][0][sel], pred["log_c_err_s"][0][sel], pred["log_x_err_s"][0][sel], pred["cov_m_c"][0][sel], pred["cov_m_x"][0][sel], pred["cov_c_x"][0][sel]))
    return jnp.concatenate([d_hat, params], axis=1)

def get_samples(key, m0_prior, target_n):
    collected = []
    current_count = 0
    while current_count < target_n:
        key, subkey = jr.split(key)
        # 2.5M chunks to stay under RAM limits
        pred = Predictive(sample_model, num_samples=1)(subkey, n_sne=2_500_000, m0_prior=m0_prior)
        rows = build_training_rows(pred)
        collected.append(onp.array(rows))
        current_count += len(rows)
    return onp.concatenate(collected, axis=0)[:target_n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="training_data", help="Output file name")
    args = parser.parse_args()
    
    key = jr.PRNGKey(42)
    all_batches = []
    
    # 20 batches of 1M samples
    for i in range(20):
        key, k1, k2 = jr.split(key, 3)
        X1 = get_samples(k1, "halfnormal", 900_000)
        X2 = get_samples(k2, "uniform", 100_000)
        
        all_batches.append(onp.concatenate([X1, X2], axis=0))
        print(f"Batch {i+1}/20 done. Total: {sum(len(b) for b in all_batches)}")
        
    # Save as one final file
    onp.save(f"training_data/{args.name}.npy", onp.concatenate(all_batches, axis=0))
    print(f"Full dataset (20M rows) saved to {args.name}.npy")

if __name__ == "__main__": 
    main()