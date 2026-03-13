#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--rep", type=int,default=0)
parser.add_argument("--name", type=str,default='base_name')
parser.add_argument("--nn_width", type=int,default=32)
parser.add_argument("--nn_depth", type=int,default=2)
parser.add_argument("--no_flows", type=int,default=4)
parser.add_argument('--cmb', action='store_true', help='CMB prior')
parser.add_argument('--mismatch', action='store_true', help='different_cosm')



args = parser.parse_args()
rep_ = args.rep
name = args.name
cmb_bool = args.cmb
vary_cmb = True
mismatch = args.mismatch
sigma_Rcmb = 0.007

if mismatch:
    R_cmb_obs_default=1.743362735519523
else:
    R_cmb_obs_default=1.7579698042257326


std_norm = True
wCDM_bool = True
ZMIN = 0.01
ZMAX = 1.4

H0 = 70

import jax
jax.config.update('jax_enable_x64',True)

import jax.random as random
from jax import jit

import pickle

import numpyro
numpyro.set_host_device_count(4)
from SkewNormalPlus import SkewNormalPlus as snp
from SkewNormalPlus3DCov_FULL import SkewNormalPlus3D
from NaiveCov_FULL import Naive


from jax import random
import jax_cosmo as jc
from jax_cosmo import Cosmology, background
import numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.stats import norm

from jax.lib import xla_bridge


from tqdm.notebook import trange
import itertools
import numpy.random as npr
itercount = itertools.count()
import logging
itercount = itertools.count()
from matplotlib import pyplot as plt


import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow,masked_autoregressive_flow


"""Training loops."""

from collections.abc import Callable

import equinox as eqx
import optax
import paramax
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)




import wcosmo
import numpy.typing


if vary_cmb:
    key = jax.random.PRNGKey(rep_)  
    key_cmb, _= random.split(key)
    cmb_draw = jax.random.normal(key_cmb)
    R_cmb_obs=R_cmb_obs_default+cmb_draw*sigma_Rcmb
else:
    R_cmb_obs=R_cmb_obs_default


def set_backend(backend):
    from importlib import import_module
    np_modules = dict(
        numpy="numpy",
        jax="jax.numpy",
        cupy="cupy",
    )
    linalg_modules = dict(
        numpy="scipy.linalg",
        jax="jax.scipy.linalg",
        cupy="cupyx.scipy.linalg",
    )
    setattr(wcosmo.wcosmo, "xp", import_module(np_modules[backend]))
    setattr(wcosmo.utils, "xp", import_module(np_modules[backend]))
    toeplitz = getattr(import_module(linalg_modules[backend]), "toeplitz")
    setattr(wcosmo.utils, "toeplitz", toeplitz)


set_backend("jax")


@jit
def time_jax_comoving_distance(jdata,omega_m,w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).comoving_distance(jdata)


@jit
def jax_dvcdz(jdata,omega_m,w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).differential_comoving_volume(jdata)

from jax.scipy.integrate import trapezoid

import jax.numpy as jnp


@jit
def minmax_fit_and_scale(X):
  max= np.max(X,axis=0)
  min = np.min(X,axis=0)
  X_std = (X - min) / (max-min)
  return X_std,min,max


@jit
def minmax_scale(X,min,max):
  return (X - min) / (max - min)

@jit
def minmax_unscale(X,min,max):
  return X * (max - min) + min

@jit
def std_fit_and_scale(X):
  mu_ = np.mean(X,axis=0)
  std_= np.std(X,axis=0)
  X_std = (X - mu_)/ std_
  return X_std,mu_,std_


@jit
def std_scale(X,mu,std):
  return (X - mu)/std

@jit
def std_unscale(X,mu,std):
  return X*std + mu

if std_norm:
    file_=np.load(name+'_std.npz')
    mu_ = file_['mu']
    std_ = file_['std']
    add_ = jnp.sum(jnp.log(std_[:3]))
else:
    file_=np.load(name+'_minmax.npz')
    min_ = file_['min']
    max_ = file_['max']
    add_ = jnp.sum(jnp.log(max_-min_)[:3])



safe_log = lambda x: jnp.log(jnp.clip(x, a_min=1e-8, a_max=None))


from copy import deepcopy
def transform_cov(cov):
    #cov = deepcopy(cov)
    #cov[cov>0] = safe_log(1+np.absolute(cov[cov>0]))
    #cov[cov<0] = -safe_log(1+np.absolute(cov[cov<0]))
    return cov


from flowjax.experimental.numpyro import distribution_to_numpyro
key, subkey= jr.split(jr.key(2))


skel_flow =masked_autoregressive_flow(
    key=key,
    base_dist=Normal(jnp.zeros(3)),
    cond_dim=10,
nn_activation= jax.nn.gelu,flow_layers=args.no_flows,
nn_width=args.nn_width,
nn_depth=args.nn_depth,
)



flow_model = eqx.tree_deserialise_leaves(name+'.eqx',skel_flow)

from numpyro.distributions import (
    Distribution,
    constraints)
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

class FlowSNP3D(Distribution):
    arg_constraints = {"m0": constraints.real,
    "alpha": constraints.real,"beta": constraints.real, "W_mm": constraints.real,"W_cc": constraints.real,"W_xx": constraints.real,
     "W_mc": constraints.real,"W_mx": constraints.real,"W_cx": constraints.real,"z_hel":constraints.real}

    support = constraints.real
    reparametrized_params = ["m0", "alpha","beta","W_mm","W_cc","W_xx","W_mc","W_mx","W_cx","z_hel","m_cut", "sigma_cut","a","b"]

    def __init__(self,m0,alpha,beta,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,z_hel,*,validate_args=False,res=1000):

        self.m0,self.alpha,self.beta,self.W_mm,self.W_cc,self.W_xx,self.W_mc,self.W_mx,self.W_cx,self.z_hel=(m0,alpha,beta,W_mm,W_cc,W_xx,W_mc,W_mx,W_cx,z_hel)
        
  
        super(FlowSNP3D, self).__init__(
            batch_shape=(m0.shape[0],),
            event_shape=(3,),
            validate_args=validate_args,
        )
    


    def sample(self,key, sample_shape=()):
        assert is_prng_key(key)



                
        X=jnp.column_stack((self.m0,self.alpha,self.beta,safe_log(self.W_mm**0.5),safe_log(self.W_cc**0.5),safe_log(self.W_xx**0.5),self.W_mc,self.W_mx,self.W_cx,self.z_hel))

        X= std_scale(X,mu_[3:],std_[3:])

        samp = std_unscale(flow_model.sample(key,(1,),condition=X),mu_[:3],std_[:3])

        X= std_unscale(X,mu_[3:],std_[3:])

        samp = samp.at[:,:3].set(samp[:,:3]+X[:,:3])[:,:3]


        return samp.reshape(sample_shape[0],sample_shape[1],3,order='F')

    @jax.jit
    def log_prob(self, value):
        

        X=jnp.append(value,jnp.column_stack((self.m0,self.alpha,self.beta,safe_log(self.W_mm**0.5),safe_log(self.W_cc**0.5),safe_log(self.W_xx**0.5),self.W_mc,self.W_mx,self.W_cx,self.z_hel)),axis=1)

        X = X.at[:,0].set(X[:,0]-X[:,3])

        X = std_scale(X,mu_,std_)
    
        return flow_model.log_prob(X[:,:3],condition=X[:,3:]) - add_
        




# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.contrib.tfp.mcmc import RandomWalkMetropolis
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import jax

from jax.scipy.stats import norm
from jax.scipy.special import ndtri
#assert numpyro.__version__.startswith("0.11.0")

numpyro.set_host_device_count(4)
jax.config.update('jax_enable_x64',True)


import jax_cosmo as jc
from jax_cosmo import Cosmology, background


from numpyro import sample

import numpyro.distributions as dist

import jax.numpy as jnp

from numpyro.distributions.truncated import TruncatedDistribution 




def flow_model(z_s,z_s_err,z_hel,data_s=None,data_err_s=None,h=H0/100,m_cut=24,sigma_pec=300,wCDM=True, R_cmb_obs= R_cmb_obs,sigma_Rcmb=sigma_Rcmb):

 
    if wCDM:
        Om0=numpyro.sample('Om0',dist.Uniform(0.01,1))
        w = numpyro.sample('w',dist.Uniform(-2,0.))
        Omde= 1-Om0
    else:
        Om0=numpyro.sample('Om0',dist.Uniform(-2,2))
        Omde = numpyro.sample('Omde',dist.Uniform(-2,2))
        w=-1

    def R_calc():
        z = jnp.array(1089.0)
        d = wcosmo.FlatwCDM(1., Om0, w).comoving_distance(z)
        d=d/(1+z)
        return jnp.sqrt(Om0)*(1+z)*d/299792.458

    R_value = R_calc()   
    if cmb_bool:
        numpyro.sample("cmb_obs",dist.Normal(jnp.array([R_value]),sigma_Rcmb), obs=jnp.array([R_cmb_obs]))


    alpha = numpyro.sample('alpha', dist.Uniform(-0.2,-0.1))
    beta = numpyro.sample('beta', dist.Uniform(2.5,3.5))

    M0=numpyro.sample('M0', dist.ImproperUniform(dist.constraints.real,(),event_shape=()))

    cosmo_jax = Cosmology(Omega_c=Om0, h=h, w0=w, Omega_b=0, n_s= 0.96, sigma8=200000, Omega_k=1-(Om0+Omde), wa=0)

    n_sne=len(z_s)

    
    def mu_func(z,zpec,zhel):

        if wCDM:
            d__ = wcosmo.FlatwCDM(H0, Om0, w).comoving_distance(z)
            mu=5*jnp.log10(d__*(1+zpec)**2*(1+zhel)*(1+z)*1e6/10)
        else:
            d__=background.transverse_comoving_distance(cosmo_jax, 1/(1+z))
            mu=5*jnp.log10((1+zpec)**2*(1+zhel)*(1+z)/h*d__*1e6/10)
            mu = mu[0]
    
        return mu
    

    def mu_vmap(z,zpec,vhel):
    
    
        return jax.vmap(mu_func ,in_axes=(0,0,0))(z,zpec,vhel)
    
    


    mu_grad=jax.grad(mu_func,argnums=0)
    mu_vpec_grad=jax.grad(mu_func,argnums=1)  


    def mu_grad_vmap(z,zpec,vhel):
        
        
        return jax.vmap(mu_grad ,in_axes=(0,0,0))(z,zpec,vhel)


    def mu_vpec_grad_vmap(z,zpec,vhel):
        
        
        return jax.vmap(mu_vpec_grad ,in_axes=(0,0,0))(z,zpec,vhel)


    with numpyro.plate("plate_i",n_sne):

        mu_s=mu_vmap(z_s,jnp.zeros(n_sne),z_hel)



        err1 = mu_grad_vmap(z_s,jnp.zeros(n_sne),z_hel)*z_s_err
        err2 = mu_vpec_grad_vmap(z_s,jnp.zeros(n_sne),z_hel)*sigma_pec/299792.458

        cov =  mu_grad_vmap(z_s,jnp.zeros(n_sne),z_hel)*mu_vpec_grad_vmap(z_s,jnp.zeros(n_sne),z_hel)*(sigma_pec/299792.458)**2


        eps= numpyro.sample('eps',dist.Normal(jnp.zeros(n_sne),jnp.sqrt(err1**2+err2**2-2*cov)))
       
        alpha= jnp.repeat(alpha,n_sne)
        
        beta= jnp.repeat(beta,n_sne)


        numpyro.sample("obs", FlowSNP3D(mu_s+M0+eps,alpha,beta,data_err_s[:,0],data_err_s[:,1],data_err_s[:,2],data_err_s[:,3],data_err_s[:,4],data_err_s[:,5],(z_s+1.)*(z_hel+1.)-1.,m_cut,sigma_cut,a,b),sample_shape=(1,), obs=data_s)


seed=0

if cmb_bool:
    name+='_cmb'

if mismatch:
    name+='_mismatch'
import os
directory = 'SNANA_chains_'+name


files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
runs_arr = np.array([],dtype=np.int32)
for f_id,f in enumerate(files):
    runs_arr=np.append(runs_arr,int(f[11:][:-4]))

if len(runs_arr)>0:
    start_at= np.max(runs_arr)+1
else:
    start_at=0

full = range(100)
these =[]

for f in full:
    if f not in runs_arr:
        these+=[f]

for add in these:
    rep_ = add
    print('run no: ',rep_)
    if mismatch:
        X_load=np.load('test_sets_mismatch/SNANA_testing_set'+str(rep_)+'.npy')
    else:
        X_load=np.load('test_sets/SNANA_testing_set'+str(rep_)+'.npy')
    X_load = X_load[np.logical_and(X_load[:,1]>0.05,X_load[:,1]<1.1),:]
    print(X_load.shape)
    z_hel = X_load[:,0]
    z_hd = X_load[:,1]
    z_hd_err = X_load[:,2]

    d_hat_sim = X_load[:,3:6]


    x_err = X_load[:,8]
    c_err =  X_load[:,7]
    mag_err =X_load[:,6]


    cov_m_c =transform_cov(X_load[:,9])
    cov_m_x =  transform_cov(X_load[:,10])
    cov_c_x =  transform_cov(X_load[:,11])


    rng_key = jax.random.PRNGKey(seed)


    d_err_s= jnp.array([mag_err**2,c_err**2,x_err**2,cov_m_c,cov_m_x,cov_c_x]).T

    z_hel=jnp.array((z_hel+1)/(z_hd+1)-1)

    X_load =0

    mag_err =0
    c_err = 0
    x_err =0 
    cov_m_c =0
    cov_m_x = 0 
    cov_c_x =0


    from numpyro.infer import init_to_value

    init_dict={'w':-0.9,'Om0':0.3,'Omde':0.65,'M0':-19.36,'alpha':-0.12,'beta':3.,'eps':jnp.zeros(len(z_hel))}

    nuts_kernel = NUTS(flow_model,adapt_step_size=True,max_tree_depth=7,init_strategy=init_to_value(values=init_dict))                                                                                                   

    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500,num_chains=4)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, jnp.array(z_hd),jnp.array(z_hd_err),jnp.array(z_hel),data_s=jnp.array(d_hat_sim),data_err_s=jnp.array(d_err_s),wCDM=wCDM_bool)
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    if wCDM_bool:
        save_labels = ['w','Om0','M0','alpha','beta']
    else:
        save_labels = ['Om0','Omde','M0','alpha','beta']

    save_dict = {}
    for sl in save_labels:
        save_dict[sl]=posterior_samples[sl]


    if wCDM_bool:
        np.savez('SNANA_chains_'+name+'/wflow_SNANA'+str(rep_)+'.npz',**save_dict)
    else:
        np.savez('SNANA_chains_'+name+'/lflow_SNANA'+str(rep_)+'.npz',**save_dict)
