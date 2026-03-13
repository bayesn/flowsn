import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import is_prng_key

@jax.jit
def std_scale(X, mu, std):
    return (X - mu) / std


class FlowSNP3D(Distribution):
    # ... (arg_constraints and support stay the same)

    def __init__(self, m0, c0, x0, sigma_res, sigma_c, sigma_x, alpha, beta, alpha_c, W_mm, W_cc, W_xx, 
                 W_mc, W_mx, W_cx, flow_model, mu, std, log_det_std, *, validate_args=False):
        
        # Save the logic-required non-parameters
        self.flow_model = flow_model
        self.mu_ = mu
        self.std_ = std
        self.add_ = log_det_std

        # Parameters that NumPyro tracks
        self.m0, self.c0, self.x0, self.sigma_res, self.sigma_c, self.sigma_x = m0, c0, x0, sigma_res, sigma_c, sigma_x
        self.alpha, self.beta, self.alpha_c = alpha, beta, alpha_c
        self.W_mm, self.W_cc, self.W_xx = W_mm, W_cc, W_xx
        self.W_mc, self.W_mx, self.W_cx = W_mc, W_mx, W_cx

        super(FlowSNP3D, self).__init__(
            batch_shape=jnp.shape(m0), 
            event_shape=(3,), 
            validate_args=validate_args
        )

    def log_prob(self, value):
        # Ensure we are using the attributes saved in __init__
        X = jnp.column_stack((
            value, # This is the [m, c, x] from the data
            self.m0, self.c0, self.x0, 
            jnp.log(self.sigma_res), jnp.log(self.sigma_c), jnp.log(self.sigma_x),
            self.alpha, self.beta, self.alpha_c, 
            jnp.log(self.W_mm**0.5), jnp.log(self.W_cc**0.5), jnp.log(self.W_xx**0.5), 
            self.W_mc, self.W_mx, self.W_cx
        ))

        # Shift values relative to predicted means
        X = X.at[:, :3].set(X[:, :3] - X[:, 3:6])
        
        # Scale using the saved mu_ and std_
        X_scaled = std_scale(X, self.mu_, self.std_)
        
        return self.flow_model.log_prob(X_scaled[:, :3], condition=X_scaled[:, 3:]) - self.add_