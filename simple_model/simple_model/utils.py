import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax import random, jit
import wcosmo

# --- Constants ---
H0 = 70.
ZMIN = 0.01
ZMAX = 1.2
z_values = jnp.linspace(ZMIN, ZMAX, 1000)

mask = np.ones((len(z_values), len(z_values)))
for i in range(len(z_values)):
    mask[i, i+1:] = 0
mask = jnp.array(mask)

def set_backend(backend):
    from importlib import import_module
    np_modules = {"numpy": "numpy", "jax": "jax.numpy", "cupy": "cupy"}
    linalg_modules = {"numpy": "scipy.linalg", "jax": "jax.scipy.linalg", "cupy": "cupyx.scipy.linalg"}
    setattr(wcosmo.wcosmo, "xp", import_module(np_modules[backend]))
    setattr(wcosmo.utils, "xp", import_module(np_modules[backend]))
    toeplitz = getattr(import_module(linalg_modules[backend]), "toeplitz")
    setattr(wcosmo.utils, "toeplitz", toeplitz)

@jit
def time_jax_comoving_distance(jdata, omega_m, w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).comoving_distance(jdata)

@jit
def jax_dvcdz(jdata, omega_m, w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).differential_comoving_volume(jdata)

@jit
def integrate_row(pdf_row, x_row):
    return trapezoid(pdf_row, x=x_row, axis=-1)

@jit
def vmap_trap(pdf, x=None):
    return jax.vmap(integrate_row, in_axes=(0, None))(pdf, x)

@jit
def cdf(pdf_values):
    pdf_values_batch = jnp.repeat(pdf_values.reshape(-1, 1), len(pdf_values), axis=1).T
    pdf_values_batch = jnp.array(pdf_values_batch * mask)
    return vmap_trap(pdf_values_batch, z_values)

@jit
def pdf(z, omega_m, w, beta_rate):
    prob = (1+z)**(beta_rate)/(1+z) * jax_dvcdz(z, jnp.array([omega_m]), jnp.array([w]))
    all_prob = (1+z_values)**(beta_rate)/(1+z_values) * jax_dvcdz(z_values, jnp.array([omega_m]), jnp.array([w]))
    integ = integrate_row(all_prob, z_values)
    return prob / integ

@jit
def find_ids(b, u):
    u = u.reshape(u.shape[0], u.shape[1], 1)
    u = jnp.repeat(u, b.shape[1], axis=2)
    b = b.reshape(1, b.shape[0], b.shape[1])
    b = jnp.repeat(b, u.shape[0], axis=0)
    return jnp.absolute(u - b).argmin(axis=-1)

@jit
def interp(x, xp, fp):
    ids = find_ids(xp, x)
    expanded_ids = jnp.expand_dims(ids, axis=1) 
    m, k = ids.shape
    xi = xp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0], expanded_ids.shape[2])
    s = jnp.sign(x - xi).astype(int).reshape(expanded_ids.shape[0], 1, expanded_ids.shape[2])
    fi = fp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0], expanded_ids.shape[2])
    a = (fp[jnp.arange(k), expanded_ids + s].reshape(expanded_ids.shape[0], expanded_ids.shape[2]) - fi) / (
        xp[jnp.arange(k), expanded_ids + s].reshape(expanded_ids.shape[0], expanded_ids.shape[2]) - xi)
    b = fi - a * xi
    return a * x + b

def sample_redshifts(size, key, Om0=0.315, w0=-1, beta_rate=1.5):
    rng, _ = random.split(key)
    u_samps = random.uniform(rng, (size,)).reshape(1, -1)
    samps = interp(u_samps, cdf(pdf(z_values, Om0, w0, beta_rate)).reshape(1, -1), z_values.reshape(1, -1))[0, :]
    return samps

# --- Scaling Utils ---
@jit
def minmax_fit_and_scale(X):
    max_val = jnp.max(X, axis=0)
    min_val = jnp.min(X, axis=0)
    return (X - min_val) / (max_val - min_val), min_val, max_val

@jit
def minmax_scale(X, min_val, max_val):
    return (X - min_val) / (max_val - min_val)

@jit
def minmax_unscale(X, min_val, max_val):
    return X * (max_val - min_val) + min_val

@jit
def std_fit_and_scale(X):
    mu_ = jnp.mean(X, axis=0)
    std_ = jnp.std(X, axis=0)
    return (X - mu_) / std_, mu_, std_

@jit
def std_scale(X, mu, std):
    return (X - mu) / std

@jit
def std_unscale(X, mu, std):
    return X * std + mu