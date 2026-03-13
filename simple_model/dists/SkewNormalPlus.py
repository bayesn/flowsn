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
from jax.scipy.stats.norm import cdf as cdf1d




 
@jax.jit
def case1(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))


@jax.jit
def case2(p, q):
    return cdf1d(p) * cdf1d(q)


@jax.jit
def case3(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


@jax.jit
def case4(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


@jax.jit
def case5(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)

@jax.jit
def binorm(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    p = (x1 - mu1) / sigma1
    q = (x2 - mu2) / sigma2


    rho = jnp.repeat(rho.reshape(rho.size,1),p.shape[1],axis=1)

    a = -rho / jnp.sqrt(1 - rho * rho)


    b = p / jnp.sqrt(1 - rho * rho)

    condition_1 = jnp.logical_and(a > 0, a * q + b >= 0)
    condition_2 = a == 0
    condition_3 = jnp.logical_and(a > 0, a * q + b < 0)
    condition_4 = jnp.logical_and(a < 0, a * q + b >= 0)
    condition_5 = jnp.logical_and(a < 0, a * q + b < 0)

    result = jnp.where(condition_1, case1(p, q, rho, a, b),
              jnp.where(condition_2, case2(p, q),
              jnp.where(condition_3, case3(p, q, rho, a, b),
              jnp.where(condition_4, case4(p, q, rho, a, b),
              jnp.where(condition_5, case5(p, q, rho, a, b), 0)))))

    return result

class SkewNormalPlus(Distribution):
    arg_constraints = {"m_int": constraints.real, "sigma_int": constraints.real,"m_cut": constraints.real, "sigma_cut": constraints.real}
    support = constraints.real
    reparametrized_params = ["m_int", "sigma_int","m_cut","sigma_cut"]

    def __init__(self,m_int,sigma_int,m_cut,sigma_cut,*, validate_args=None,res=1000):


        
        self.m_int, self.sigma_int,self.m_cut,self.sigma_cut = promote_shapes(m_int,sigma_int,m_cut,sigma_cut)
        self.sigma_cut=self.sigma_cut*(-1)

        batch_shape = lax.broadcast_shapes(jnp.shape(m_int), jnp.shape(sigma_int),jnp.shape(m_int), jnp.shape(sigma_int))

        self.b = self.sigma_int/self.sigma_cut

        self.a = (self.m_int-self.m_cut)/self.sigma_cut
        
        rho = -self.b / jnp.sqrt(1 + self.b**2)
    
        n_obj = m_int.shape[0]
        
        
        self.n_obj=n_obj
        self.res = res
        
        
        self.trials=jnp.repeat(jnp.linspace(-5,5,res).reshape(1,res),n_obj,axis=0)
        
        self.T=self.a/jnp.sqrt(1+self.b**2)


        
        self.T=jnp.repeat(self.T.reshape(n_obj,1),res,axis=1)
        
        
        self.grid=binorm(self.T,self.trials,0,0,1,1,rho)
        super(SkewNormalPlus, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )
        

    @jax.jit
    def find_ids(self,b,u):
        u=u.reshape(u.shape[0],u.shape[1],1)
    
        u = jnp.repeat(u,b.shape[1],axis=2)
    
        b = b.reshape(1,b.shape[0],b.shape[1])
    
        b = jnp.repeat(b,u.shape[0],axis=0)
    
        return jnp.absolute(u-b).argmin(axis=-1)


        
    @jax.jit
    def interp(self,x, xp, fp):

        
        ids=self.find_ids(xp,x)

    
        expanded_ids = jnp.expand_dims(ids, axis=1) 
        
        m,k=ids.shape


        xi = xp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0],expanded_ids.shape[2])

        s = jnp.sign(x-xi).astype(int).reshape(expanded_ids.shape[0],1,expanded_ids.shape[2])

        fi = fp[jnp.arange(k), expanded_ids].reshape(expanded_ids.shape[0],expanded_ids.shape[2])
    
        a = (fp[jnp.arange(k), expanded_ids+  s].reshape(expanded_ids.shape[0],expanded_ids.shape[2]) - fi) / (
        xp[jnp.arange(k), expanded_ids+ s].reshape(expanded_ids.shape[0],expanded_ids.shape[2]) - xi)
        b = fi - a * xi
        return a * x + b


    
    def sample(self,key, sample_shape=()):
        assert is_prng_key(key)
        
        u = random.uniform(key, shape=sample_shape)*(jnp.max(self.grid,axis=1)-jnp.min(self.grid,axis=1))-jnp.min(self.grid,axis=1)
        
   
        x = self.interp(u,self.grid, self.trials)

        return self.sigma_int*x + self.m_int



    

    @validate_sample
    def log_prob(self, value,mcmc=False):

        

        if mcmc:
            
            value = (value-self.m_int.reshape(len(value),))/self.sigma_int
            value = value.reshape(1,-1)
        else:
            value = (value-self.m_int)/self.sigma_int
        

        return js.norm.logpdf(value) + js.norm.logcdf(self.b * value +self.a) - js.norm.logcdf(self.a / jnp.sqrt(1 + self.b**2))-jnp.log(self.sigma_int)
