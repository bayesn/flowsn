import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm
import jax_cosmo as jc
from jax_cosmo import Cosmology, background
import wcosmo

from utils import H0, time_jax_comoving_distance
from dists.flowsn_nf import FlowSNP3D
from dists.SkewNormalPlus3DCov_FULL import SkewNormalPlus3D
from dists.NaiveCov_FULL import Naive

def sample_model(z_s, d_hat_s=None, log_x_err=None, log_c_err=None, log_mag_err=None, cov_m_c=None, cov_c_x=None, 
                 cov_m_x=None, sel_s=None, M0=-19.365, sigma_int=0.1, alpha=-0.14, alpha_c=-0.008, beta=3.1, 
                 x0=-0.432, sigma_x=1.124, c0=-0.061, sigma_c=0.065, h=H0/100, Om0=0.315, w=-1, m_cut=24, 
                 sigma_cut=0.25, a=-0.1, b=-1):

    n_sne = len(z_s)
    with numpyro.plate("plate_i", n_sne):
        d_s = time_jax_comoving_distance(z_s, jnp.array([Om0]), jnp.array([w]))
        mu_s = 5 * jnp.log10(d_s * (1 + z_s) * 1e6 / 10)
    
        x_s_ = numpyro.sample("x_s", dist.Normal(x0, sigma_x))
        c_s_ = numpyro.sample("c_s", dist.Normal(c0 + alpha_c * x_s_, sigma_c))
        m_s_ = numpyro.sample("m_s_", dist.Normal(M0 + mu_s + alpha * x_s_ + beta * c_s_, sigma_int))
        
        log_x_err = numpyro.sample("log_x_err_s", dist.Normal(-1.5, 0.5), obs=log_x_err)
        log_c_err = numpyro.sample("log_c_err_s", dist.Normal(-3.5, 0.3), obs=log_c_err)
        log_mag_err = numpyro.sample("log_mag_err_s", dist.Normal(0.1 * (m_s_ - 56), 0.6), obs=log_mag_err)
        
        cov_m_c = numpyro.sample("cov_m_c", dist.Normal(0.000520 + 0.288792999 * jnp.exp(log_mag_err)**2, jnp.exp(log_mag_err)**2 * 0.125979), obs=cov_m_c)
        cov_m_x = numpyro.sample("cov_m_x", dist.Normal(0.0008097 + 0.03835550 * jnp.exp(log_x_err)**2, jnp.exp(log_x_err)**2 * 0.02032), obs=cov_m_x)
        cov_c_x = numpyro.sample("cov_c_x", dist.Normal(0.000168655 + 0.01358504 * jnp.exp(log_x_err)**2, jnp.exp(log_x_err)**2 * 0.011840861), obs=cov_c_x)

        W_s = jnp.array([[jnp.exp(log_mag_err)**2, cov_m_c, cov_m_x],
                         [cov_m_c, jnp.exp(log_c_err)**2, cov_c_x],
                         [cov_m_x, cov_c_x, jnp.exp(log_x_err)**2]]).T
    
    d_hat_s = numpyro.sample("d_hat_s", dist.MultivariateNormal(jnp.column_stack((m_s_, c_s_, x_s_)), W_s), obs=d_hat_s)
    p_s = norm.cdf(-(d_hat_s[:,0] + a * d_hat_s[:,2] + b * d_hat_s[:,1]), loc=-m_cut, scale=sigma_cut)
    sel_s = numpyro.sample("sel_s", dist.Bernoulli(p_s), obs=sel_s)

def mcmc_model(z_s, data_s=None, data_err_s=None, h=H0/100, m_cut=24, sigma_cut=0.25, a=-0.1, b=-1, 
               sigma_pec=300, wCDM=True, model_type='analytical', 
               flow_kwargs=None, cmb_kwargs=None):
    
    if wCDM:
        Om0 = numpyro.sample('Om0', dist.Uniform(0, 1))
        w = numpyro.sample('w', dist.Uniform(-2, 2))
        Omde = 1 - Om0
    else:
        Om0 = numpyro.sample('Om0', dist.Uniform(-2, 2))
        Omde = numpyro.sample('Omde', dist.Uniform(-2, 2))
        w = -1
        
    alpha = numpyro.sample('alpha', dist.Uniform(-0.2, -0.1))
    beta = numpyro.sample('beta', dist.Uniform(2.5, 3.5))
    alpha_c = numpyro.sample('alpha_c', dist.Uniform(-0.02, 0.0))
    
    M0 = numpyro.sample('M0', dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
    c0 = numpyro.sample("c0_int", dist.Uniform(-0.1, 0.))
    x0 = numpyro.sample("x0", dist.Uniform(-0.6, -0.2))

    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(1))
    sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(1))
    sigma_x = numpyro.sample("sigma_x", dist.HalfNormal(2))

    cosmo_jax = Cosmology(Omega_c=Om0, h=h, w0=w, Omega_b=0, n_s=0.96, sigma8=200000, Omega_k=1-(Om0+Omde), wa=0)
    n_sne = len(z_s)

    def mu_func(z):
        if wCDM:
            d__ = wcosmo.FlatwCDM(H0, Om0, w).comoving_distance(z)
            return 5 * jnp.log10(d__ * (1 + z) * 1e6 / 10)
        else:
            d__ = background.transverse_comoving_distance(cosmo_jax, 1 / (1 + z))
            return (5 * jnp.log10((1 + z) / h * d__ * 1e6 / 10))[0]
    
    mu_grad_vmap = jax.vmap(jax.grad(mu_func), in_axes=(0))
    mu_vmap = jax.vmap(mu_func, in_axes=(0))
    
    if cmb_kwargs and cmb_kwargs.get("cmb_bool"):
        z_cmb = jnp.array(1089.0)
        d_cmb = wcosmo.FlatwCDM(1., Om0, w).comoving_distance(z_cmb) / (1 + z_cmb)
        R_value = jnp.sqrt(Om0) * (1 + z_cmb) * d_cmb / 299792.458
        numpyro.sample("cmb_obs", dist.Normal(jnp.array([R_value]), cmb_kwargs["sigma_Rcmb"]), obs=jnp.array([cmb_kwargs["R_cmb_obs"]]))

    with numpyro.plate("plate_i", n_sne):
        mu_s = mu_vmap(z_s)
        mu_err = jnp.absolute(mu_grad_vmap(z_s) * (sigma_pec / 299792.458))
        sigma_int_ = jnp.sqrt(sigma_int**2 + mu_err**2)
        
        if model_type == 'analytical':
            numpyro.sample("obs", SkewNormalPlus3D(mu_s + M0, jnp.repeat(c0, n_sne), jnp.repeat(x0, n_sne), sigma_int_, sigma_c, sigma_x,
                       alpha, beta, alpha_c, data_err_s[:,0], data_err_s[:,1], data_err_s[:,2], data_err_s[:,3], data_err_s[:,4], data_err_s[:,5], m_cut, sigma_cut, a, b), sample_shape=(1,), obs=data_s)
        elif model_type == 'naive':
            numpyro.sample("obs", Naive(mu_s + M0, jnp.repeat(c0, n_sne), jnp.repeat(x0, n_sne), sigma_int_, sigma_c, sigma_x,
                       alpha, beta, alpha_c, data_err_s[:,0], data_err_s[:,1], data_err_s[:,2], data_err_s[:,3], data_err_s[:,4], data_err_s[:,5], m_cut, sigma_cut, a, b), sample_shape=(1,), obs=data_s)
        elif model_type == 'flow':
            # This matches the signature in the new __init__ above
            numpyro.sample("obs", FlowSNP3D(
                mu_s + M0, jnp.repeat(c0, n_sne), jnp.repeat(x0, n_sne), 
                sigma_int_, jnp.repeat(sigma_c, n_sne), jnp.repeat(sigma_x, n_sne),
                jnp.repeat(alpha, n_sne), jnp.repeat(beta, n_sne), jnp.repeat(alpha_c, n_sne), 
                data_err_s[:,0], data_err_s[:,1], data_err_s[:,2], 
                data_err_s[:,3], data_err_s[:,4], data_err_s[:,5], 
                m_cut, sigma_cut, a, b,
                flow_model=flow_kwargs['flow_model'],
                mu=flow_kwargs['mu_'],
                std=flow_kwargs['std_'],
                log_det_std=flow_kwargs['add_']
            ), obs=data_s)