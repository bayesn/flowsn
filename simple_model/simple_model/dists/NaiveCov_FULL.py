import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import jax.scipy.stats as js
import jax.random as random
from numpyro.distributions import (
    Distribution,
    constraints)
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample
from jax.scipy.special import erf

import numpyro.distributions as dist




class Naive(Distribution):
    arg_constraints = {"m0": constraints.real,"c0": constraints.real,"x0": constraints.real,
    "sigma_int": constraints.real,"sigma_c": constraints.real,"sigma_x": constraints.real,
    "alpha": constraints.real,"beta": constraints.real,"alpha_c": constraints.real
    ,"m_cut": constraints.real, "sigma_cut": constraints.real,"a":constraints.real,"b":constraints.real,
    "W_mm": constraints.real,"W_cc": constraints.real,"W_xx": constraints.real,"W_mc": constraints.real,"W_mx": constraints.real,"W_cx": constraints.real}

    support = constraints.real
    reparametrized_params = ["m0", "c0","x0","sigma_int","sigma_c","sigma_x",
    "alpha","beta","alpha_c", "W_mm","W_cc","W_xx","W_mc","W_mx","W_cx","m_cut", "sigma_cut","a","b"]

    def __init__(self,m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b,*,validate_args=False,res=1000):

        
        self.m0,self.c0,self.x0,self.sigma_int,self.sigma_c,self.sigma_x,self.alpha,self.beta,self.alpha_c,self.W_mm,self.W_cc,self.W_xx,self.W_mc,self.W_mx,self.W_cx,self.m_cut,self.sigma_cut,self.a,self.b=(m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b)

   
        super(Naive, self).__init__(
            batch_shape=(m0.shape[0],),
            event_shape=(3,),
            validate_args=validate_args,
        )
    

    def sample(self, key,sample_shape):
        n_sne = self.m0.shape[0]

        # ---------------- Mean vector ----------------
        exp_x = self.x0
        exp_c = self.c0 + self.alpha_c * self.x0
        exp_m = self.m0 + self.alpha*self.x0 + self.beta*(self.alpha_c*self.x0 + self.c0)

        phi_s = jnp.stack([exp_m, exp_c, exp_x], axis=-1)  # (n_sne, 3)

        # ---------------- Covariance terms ----------------
        var_m = self.sigma_int**2 + (self.alpha + self.beta*self.alpha_c)**2*self.sigma_x**2 + self.beta**2*self.sigma_c**2 + self.W_mm
        var_c = self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2 + self.W_cc
        var_x = self.sigma_x**2 + self.W_xx

        cov_m_c = self.alpha*self.alpha_c*self.sigma_x**2 + self.beta*(self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2) + self.W_mc
        cov_m_x = self.alpha*self.sigma_x**2 + self.beta*self.alpha_c*self.sigma_x**2 + self.W_mx
        cov_c_x = self.alpha_c*self.sigma_x**2 + self.W_cx

        # ---------------- Full covariance matrix (n_sne,3,3) ----------------
        W_s = jnp.stack(
            [
                jnp.stack([var_m,   cov_m_c, cov_m_x], axis=-1),
                jnp.stack([cov_m_c, var_c,   cov_c_x], axis=-1),
                jnp.stack([cov_m_x, cov_c_x, var_x],   axis=-1),
            ],
            axis=-2,
        )

        # ---------------- Distribution ----------------
        mvn = dist.MultivariateNormal(phi_s, covariance_matrix=W_s)

        return mvn.sample(key,sample_shape=(1,))  # shape (n_sne, 3)

    @jax.jit
    @validate_sample
    def log_prob(self, value,mcmc=False):

        n_sne = self.m0.shape[0]


        exp_x = self.x0

        exp_c = self.c0 + self.alpha_c*self.x0

        exp_m = self.m0+self.alpha*self.x0+self.beta*(self.alpha_c*self.x0+self.c0)
        
                
        phi_s=jnp.column_stack((exp_m,
                                exp_c,
                               exp_x))


        
        var_m = self.sigma_int**2+(self.alpha+self.beta*self.alpha_c)**2*self.sigma_x**2+self.beta**2*self.sigma_c**2+self.W_mm
        var_c = self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2+self.W_cc
        var_x = self.sigma_x**2+self.W_xx
                               
        cov_m_c = self.alpha*self.alpha_c*self.sigma_x**2 +self.beta*(self.sigma_c**2+self.alpha_c**2*self.sigma_x**2) + self.W_mc
        
        cov_m_x = self.alpha*self.sigma_x**2 + self.beta*self.alpha_c*self.sigma_x**2 + self.W_mx
            
        cov_c_x = self.alpha_c * self.sigma_x**2 + self.W_cx
                               
                               
        W_s = jnp.array([[var_m,cov_m_c,cov_m_x],
                         [cov_m_c,var_c,cov_c_x],
                         [cov_m_x,cov_c_x,var_x]]).T

        

    
        prob = dist.MultivariateNormal(phi_s, W_s).log_prob(value)  


        return prob

