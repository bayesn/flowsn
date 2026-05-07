import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from pathlib import Path

SNANA_DIR = Path(__file__).resolve().parent

# =========================
# Constants
# =========================

H0          = 70
ZMIN        = 0.01
ZMAX        = 1.4
STD_NORM    = True
VARY_CMB    = True
SIGMA_RCMB  = 0.007
C_LIGHT     = 299_792.458
Z_OF_CMB    = 1089.0
N_S         = 0.96
SIGMA8      = 200_000
TEMPLATE_M0 = -19.365

# key: (coeff, const) such that coeff*df[key] + const is appended
std = (1, 0)  # gives just df[key]
SNANA_KEYS = {
    'zHEL': std,
    'zHD': std,
    'zHDERR': std,
    'mB': std,
    'c': std,
    'x1': std,
    'mBERR': std,
    'cERR': std,
    'x1ERR': std,
    'COV_c_x0': [-2.5/np.log(10), 0],  # coeff needs /df['x0']
    'COV_x1_x0': [-2.5/np.log(10), 0], # coeff needs /df['x0']
    'COV_x1_c': std,
    'SIM_DLMAG': (1, TEMPLATE_M0),
    'SIM_alpha': (-1, 0),
    'SIM_beta': std
}

R_CMB_OBS_DEFAULT = {
    1: 1.75796,   # cosmo 1
    2: 1.74336,    # cosmo 2 
    3: 1.74935 # cosmo 3 
}

# =========================
# Scaling
# =========================

safe_log = lambda x: jnp.log(jnp.clip(x, a_min=1e-8))

@jit
def std_scale(X, mu, std):
    return (X - mu) / std

@jit
def std_unscale(X, mu, std):
    return X * std + mu

@jit
def minmax_scale(X, mn, mx):
    return (X - mn) / (mx - mn)

@jit
def minmax_unscale(X, mn, mx):
    return X * (mx - mn) + mn

# =========================
# Normalisation loading
# =========================

def load_normalisation(name, std_norm=True):
    if std_norm:
        f = np.load(SNANA_DIR / "flow_weights" / (name + "_std.npz"))
        mu, std = f["mu"], f["std"]
        return dict(mu=mu, std=std, kind="std")
    else:
        f = np.load(SNANA_DIR / "flow_weights" / (name + "_minmax.npz"))
        mn, mx = f["min"], f["max"]
        return dict(min=mn, max=mx, kind="minmax")

# =========================
# Data helpers
# =========================

def transform_cov(cov):
    return cov


def parse_test_set(X):
    return dict(
        z_hel    = jnp.array((X[:, 0] + 1) / (X[:, 1] + 1) - 1),
        z_hd     = jnp.array(X[:, 1]),
        z_hd_err = jnp.array(X[:, 2]),
        d_hat    = jnp.array(X[:, 3:6]),
        d_err    = jnp.array([
            X[:, 6] ** 2, X[:, 7] ** 2, X[:, 8] ** 2,
            transform_cov(X[:, 9]),
            transform_cov(X[:, 10]),
            transform_cov(X[:, 11]),
        ]).T,
        mass = jnp.array(X[:,-1])
    )
