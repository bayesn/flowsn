import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import is_prng_key
from jax.tree_util import register_pytree_node

from utils import safe_log, std_scale, std_unscale

def load_flow(name, no_flows, nn_width, nn_depth):
    key, subkey = jr.split(jr.key(2))
    skeleton = masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(3)),
        cond_dim=10,
        nn_activation=jax.nn.gelu,
        flow_layers=no_flows,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )
    return eqx.tree_deserialise_leaves("flow_weights/" + name + ".eqx", skeleton)

class FlowSNP3D(Distribution):
    arg_constraints = {
        "m0":    constraints.real,
        "alpha": constraints.real,
        "beta":  constraints.real,
        "W_mm":  constraints.real,
        "W_cc":  constraints.real,
        "W_xx":  constraints.real,
        "W_mc":  constraints.real,
        "W_mx":  constraints.real,
        "W_cx":  constraints.real,
        "z_hel": constraints.real,
        "scale_mu": constraints.real,
        "scale_std": constraints.real,
    }
    support = constraints.real
    
    def __init__(self, m0, alpha, beta,
                 W_mm, W_cc, W_xx, W_mc, W_mx, W_cx,
                 z_hel, flow, scale_mu, scale_std, *, validate_args=False):
        self.m0    = m0
        self.alpha = alpha
        self.beta  = beta
        self.W_mm, self.W_cc, self.W_xx = W_mm, W_cc, W_xx
        self.W_mc, self.W_mx, self.W_cx = W_mc, W_mx, W_cx
        self.z_hel       = z_hel
        self.scale_mu = scale_mu
        self.scale_std = scale_std
        self.flow = flow
        
        super().__init__(
            batch_shape=jnp.shape(m0),
            event_shape=(3,),
            validate_args=validate_args,
        )

    def tree_flatten(self):
        # Arrays/Tracers that JAX should track for gradients
        params = (self.m0, self.alpha, self.beta, self.W_mm, self.W_cc, self.W_xx, 
                  self.W_mc, self.W_mx, self.W_cx, self.z_hel, self.scale_mu, self.scale_std)
        # Static metadata (the flow itself and validation flags)
        aux_data = {'flow': self.flow, 'validate_args': self._validate_args}
        return (params, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, **aux_data)

    def log_prob(self, value):
        # 1. Build the 10D condition vector
        cond = jnp.column_stack((
            self.m0, self.alpha, self.beta,
            safe_log(self.W_mm ** 0.5),
            safe_log(self.W_cc ** 0.5),
            safe_log(self.W_xx ** 0.5),
            self.W_mc, self.W_mx, self.W_cx,
            self.z_hel,
        ))

        v_shifted = value.at[:, 0].set(value[:, 0] - self.m0)   
        X = jnp.column_stack((v_shifted, cond))

        X = std_scale(X, self.scale_mu, self.scale_std)
        

        lp = self.flow.log_prob(X[:, :3], condition=X[:, 3:])
        return lp  - jnp.sum(jnp.log(self.scale_std[:3]))

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        
        cond = jnp.column_stack((
            self.m0, self.alpha, self.beta,
            safe_log(self.W_mm ** 0.5),
            safe_log(self.W_cc ** 0.5),
            safe_log(self.W_xx ** 0.5),
            self.W_mc, self.W_mx, self.W_cx,
            self.z_hel,
        ))
        
        # Scale the condition (indices 3 to 13)
        cond_scaled = std_scale(cond, self.scale_mu[3:], self.scale_std[3:])
        
        # Sample from flow
        samp_scaled = self.flow.sample(key, sample_shape, condition=cond_scaled)
        
        # Unscale the 3D sample (indices 0, 1, 2) and add mean back
        samp = std_unscale(samp_scaled, self.scale_mu[:3], self.scale_std[:3])
        return samp + cond[:, :3]

# Register with a check to prevent "Duplicate custom PyTreeDef" errors
try:
    register_pytree_node(FlowSNP3D, FlowSNP3D.tree_flatten, FlowSNP3D.tree_unflatten)
except ValueError:
    pass
