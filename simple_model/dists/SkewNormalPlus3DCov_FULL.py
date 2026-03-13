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
from dists.SkewNormalPlus import SkewNormalPlus
import numpyro.distributions as dist




class SkewNormalPlus3D(Distribution):
    arg_constraints = {"m0": constraints.real,"c0": constraints.real,"x0": constraints.real,
    "sigma_int": constraints.real,"sigma_c": constraints.real,"sigma_x": constraints.real,
    "alpha": constraints.real,"beta": constraints.real,"alpha_c": constraints.real
    ,"m_cut": constraints.real, "sigma_cut": constraints.real,"a":constraints.real,"b":constraints.real,
    "W_mm": constraints.real,"W_cc": constraints.real,"W_xx": constraints.real, "W_mc": constraints.real,"W_mx": constraints.real,"W_cx": constraints.real}

    support = constraints.real
    reparametrized_params = ["m0", "c0","x0","sigma_int","sigma_c","sigma_x",
    "alpha","beta","alpha_c", "W_mm","W_cc","W_xx","W_mc","W_mx","W_cx","m_cut", "sigma_cut","a","b"]

    def __init__(self,m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b,*,validate_args=False,res=10000):

        
        self.m0,self.c0,self.x0,self.sigma_int,self.sigma_c,self.sigma_x,self.alpha,self.beta,self.alpha_c,self.W_mm,self.W_cc,self.W_xx,self.W_mc,self.W_mx,self.W_cx,self.m_cut,self.sigma_cut,self.a,self.b=(m0,c0,x0,sigma_int,sigma_c,sigma_x,alpha,beta,alpha_c,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,m_cut,sigma_cut,a,b)



        self.res=res
        

        super(SkewNormalPlus3D, self).__init__(
            batch_shape=(m0.shape[0],),
            event_shape=(3,),
            validate_args=validate_args,
        )
    


    def sample(self,key, sample_shape=()):

        
        assert is_prng_key(key)

        n_sne = self.m0.shape[0]
        sample_shape =(1,n_sne)

        exp_x = self.x0

        exp_c = self.c0 + self.alpha_c*self.x0

        exp_m = self.m0+self.alpha*self.x0+self.beta*(self.alpha_c*self.x0+self.c0)
        

        
        var_m = self.sigma_int**2+(self.alpha+self.beta*self.alpha_c)**2*self.sigma_x**2+self.beta**2*self.sigma_c**2+self.W_mm
        var_c = self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2+self.W_cc
        var_x = self.sigma_x**2+self.W_xx
                               
        cov_m_c = self.alpha*self.alpha_c*self.sigma_x**2 +self.beta*(self.sigma_c**2+self.alpha_c**2*self.sigma_x**2) + self.W_mc
        
        cov_m_x = self.alpha*self.sigma_x**2 + self.beta*self.alpha_c*self.sigma_x**2 + self.W_mx
            
        cov_c_x = self.alpha_c * self.sigma_x**2 + self.W_cx
                               
                               



        var_c_giv_x = var_c - cov_c_x**2/var_x 
        var_m_giv_x = var_m - cov_m_x**2/var_x
        cov_mc_giv_x = cov_m_c -cov_m_x*cov_c_x/var_x

        var_m_giv_cx = var_m_giv_x - cov_mc_giv_x**2/var_c_giv_x



        x_div = cov_m_x/var_x + self.a +self.b*cov_c_x/var_x

        b_ = self.b+cov_mc_giv_x/var_c_giv_x

        a_ = self.a  +cov_m_x/var_x+  cov_c_x/var_x*self.b


        eval_denom =  self.sigma_cut*jnp.sqrt(1+(var_m_giv_cx/self.sigma_cut**2))

        eval_denom = eval_denom*jnp.sqrt(1+(var_c_giv_x*b_**2/eval_denom**2))/x_div


    
   
        snp=SkewNormalPlus(m_int=exp_x,sigma_int=jnp.sqrt(var_x),m_cut=(self.m_cut-(exp_m-cov_m_x/var_x*exp_x+self.b*(exp_c-cov_c_x/var_x*exp_x)))/x_div ,sigma_cut=eval_denom,res=self.res)
        key,_ = random.split(key)
        x_s=snp.sample(key,sample_shape).reshape(sample_shape[0]*sample_shape[1],order='F')

        exp_c_giv_x = exp_c + cov_c_x/var_x *(x_s-exp_x)
        exp_m_giv_x = exp_m + cov_m_x/var_x *(x_s-exp_x)

        c_div = cov_mc_giv_x/var_c_giv_x + self.b

  
        eval_denom =  self.sigma_cut*jnp.sqrt(1+(var_m_giv_cx/self.sigma_cut**2))/c_div

  
        snp=SkewNormalPlus(m_int=exp_c_giv_x,
        sigma_int=jnp.sqrt(var_c_giv_x),m_cut= (self.m_cut-(exp_m_giv_x-cov_mc_giv_x/var_c_giv_x*exp_c_giv_x+self.a*x_s))/c_div
                ,sigma_cut=eval_denom,res=self.res)


        key,_ = random.split(key)
        c_s=snp.sample(key,(1,sample_shape[0]*sample_shape[1])).reshape(sample_shape[0]*sample_shape[1],)

        snp=SkewNormalPlus(m_int=exp_m_giv_x+cov_mc_giv_x/var_c_giv_x*(c_s-exp_c_giv_x),
        sigma_int=jnp.sqrt(var_m_giv_cx),m_cut= self.m_cut-self.a*x_s-self.b*c_s,sigma_cut=self.sigma_cut,res=self.res)
        key,_ = random.split(key)
        m_s=snp.sample(key,(1,sample_shape[0]*sample_shape[1])).reshape(sample_shape[0]*sample_shape[1],)

  
        return jnp.column_stack((m_s,c_s,x_s)).reshape(sample_shape[0],sample_shape[1],3,order='F')




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

        


        var_c_giv_x = var_c - cov_c_x**2/var_x 
        var_c_giv_x = jnp.where(var_c_giv_x>0,var_c_giv_x,0)
        var_m_giv_x = var_m - cov_m_x**2/var_x
        var_m_giv_x = jnp.where(var_m_giv_x>0,var_m_giv_x,0)

        cov_mc_giv_x = cov_m_c -cov_m_x*cov_c_x/var_x

        var_m_giv_cx = var_m_giv_x - cov_mc_giv_x**2/var_c_giv_x
        var_m_giv_cx = jnp.where(var_m_giv_cx>0,var_m_giv_cx,0)



        b_ = self.b+cov_mc_giv_x/var_c_giv_x

        a_ = self.a  +cov_m_x/var_x+  cov_c_x/var_x*self.b

        eval_num = self.m_cut - (exp_m + exp_c*self.b +exp_x*self.a)


        eval_denom =  self.sigma_cut*jnp.sqrt(1+(var_m_giv_cx/self.sigma_cut**2))

        eval_denom = eval_denom*jnp.sqrt(1+(var_c_giv_x*b_**2/eval_denom**2))

        eval_denom =eval_denom*jnp.sqrt(1+(var_x*a_**2/eval_denom**2))

        denom =  js.norm.logcdf(eval_num/eval_denom)  
    
        prob = dist.MultivariateNormal(phi_s, W_s).log_prob(value) + js.norm.logcdf((self.m_cut-(value[:,0]+self.a*value[:,2]+self.b*value[:,1]))/self.sigma_cut) -denom

        return prob

