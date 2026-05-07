import jax
import jax.numpy as jnp
from jax import grad

import numpyro
import numpyro.distributions as dist
from jax_cosmo import Cosmology, background
import wcosmo

from utils import H0, SIGMA_RCMB, C_LIGHT, Z_OF_CMB, N_S, SIGMA8
from flow_dist import FlowSNP3D


def numpyro_model(
    z_s, z_s_err, z_hel, mass = None, data_s=None, data_err_s=None,
    h=H0 / 100, sigma_pec=300, gamma_bool = False,
    wCDM=True, wa_bool= False, cmb_bool=False,
    R_cmb_obs=None, sigma_Rcmb=SIGMA_RCMB,
    flow=None, scale_mu = 1., scale_std = 1.0
):
    # --- Cosmological parameters ---

    if wCDM:
        Om0  = numpyro.sample("Om0", dist.Uniform(0.01, 1.0))
        w    = numpyro.sample("w0",   dist.Uniform(-2.0, 0.0))
        Omde = 1.0 - Om0
        if wa_bool:
            wa = numpyro.sample("wa", dist.Uniform(-3., 3.))
        else:
            wa = 0.
    else:
        Om0  = numpyro.sample("Om0",  dist.Uniform(-2, 2))
        Omde = numpyro.sample("Omde", dist.Uniform(-2, 2))
        w    = -1.0



    # --- SN nuisance parameters ---
    alpha   = numpyro.sample("alpha", dist.Uniform(-0.2, -0.1))
    beta    = numpyro.sample("beta",  dist.Uniform(2.5,  3.5))
    M0      = numpyro.sample("M0",    dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
    if gamma_bool:
        gamma   = numpyro.sample("gamma", dist.Uniform(-0.5, 0.5))
    else:
        gamma = 0.0

    M_split = 10  # For now, can make a sampled parameter later

    mass_mask = jnp.where(mass > M_split, 1, 0)

    cosmo_jax = Cosmology(
        Omega_c=Om0, h=h, w0=w, Omega_b=0, n_s=N_S,
        sigma8=SIGMA8, Omega_k=1 - (Om0 + Omde), wa=wa,
    )
    n_sne = len(z_s)

    # --- Distance modulus ---
    def mu_func(z, zpec, zhel):
        if wCDM and not wa_bool:
            d  = wcosmo.FlatwCDM(H0, Om0, w).comoving_distance(z)
            mu = 5 * jnp.log10(d * (1 + zpec) ** 2 * (1 + zhel) * (1 + z)) + 25
        else:
            d  = background.transverse_comoving_distance(cosmo_jax, 1 / (1 + z))
            mu = 5 * jnp.log10((1 + zpec) ** 2 * (1 + zhel) * (1 + z) / h * d) + 25
            mu = mu[0]
        return mu
    
        # --- CMB shift parameter constraint ---
    if cmb_bool:

        def R_calc():
            z = jnp.array(Z_OF_CMB)
            if wCDM and not wa_bool:
                # Pass the physical H0 to get distance in physical Mpc
                d_physical = wcosmo.FlatwCDM(h*100, Om0, w).comoving_distance(z)
            else:
                d_mpc_h = background.transverse_comoving_distance(cosmo_jax, 1 / (1 + z))
                # CONVERT: Mpc/h -> physical Mpc by dividing by h
                d_physical = d_mpc_h / h

            # Standard Physical Formula: R = sqrt(Om) * H0 * D_physical / c
            return jnp.squeeze(jnp.sqrt(Om0) * h * 100 * d_physical / C_LIGHT)
        

        numpyro.sample(
            "cmb_obs",
            dist.Normal(jnp.array([R_calc()]), sigma_Rcmb),
            obs=jnp.array([R_cmb_obs]),
        )

    mu_vmap           = jax.vmap(mu_func,                   in_axes=(0, 0, 0))
    mu_grad_vmap      = jax.vmap(grad(mu_func, argnums=0),  in_axes=(0, 0, 0))
    mu_vpec_grad_vmap = jax.vmap(grad(mu_func, argnums=1),  in_axes=(0, 0, 0))

    with numpyro.plate("plate_i", n_sne):
        mu_s  = mu_vmap(z_s, jnp.zeros(n_sne), z_hel)
        dmu_z = mu_grad_vmap(z_s, jnp.zeros(n_sne), z_hel)
        dmu_v = mu_vpec_grad_vmap(z_s, jnp.zeros(n_sne), z_hel)

        err1  = dmu_z * z_s_err
        err2  = dmu_v * sigma_pec / C_LIGHT
        cov   = dmu_z * dmu_v * (sigma_pec / C_LIGHT) ** 2

        eps = numpyro.sample(
            "eps",
            dist.Normal(jnp.zeros(n_sne), jnp.sqrt(err1 ** 2 + err2 ** 2 - 2 * cov)),
        )

        numpyro.sample(
            "obs",
            FlowSNP3D(
                mu_s + M0 + gamma*mass_mask + eps,
                jnp.repeat(alpha, n_sne),
                jnp.repeat(beta,  n_sne),
                data_err_s[:, 0], data_err_s[:, 1], data_err_s[:, 2],
                data_err_s[:, 3], data_err_s[:, 4], data_err_s[:, 5],
                (z_s + 1.0) * (z_hel + 1.0) - 1.0,
                flow = flow,
                scale_mu = scale_mu,
                scale_std = scale_std,
            ),
            sample_shape=(1,),
            obs=data_s,
        )
