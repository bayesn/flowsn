#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--name", type=str,default='base_name')
parser.add_argument("--nn_width", type=int,default=32)
parser.add_argument("--nn_depth", type=int,default=2)
parser.add_argument("--no_flows", type=int,default=4)
parser.add_argument("--max_pat", type=int,default=10)
parser.add_argument("--val_prop", type=float,default=0.1)
parser.add_argument("--batch_size", type=int,default=8092)
parser.add_argument("--epochs", type=int,default=10)
parser.add_argument('--save_all', action='store_true', help='save old model versions')

args = parser.parse_args()
save_all = args.save_all
name = args.name
import numpy as onp

if name =='base_name':
    name = 'base_w'+str(args.nn_width)+'_d'+str(args.nn_depth)+'_f'+str(args.no_flows)+'_vp'+str(onp.around(args.val_prop,2))+'_bs'+str(args.batch_size)
    if save_all==False:
        name+='_p'+str(args.max_pat)


std_norm = True
restart=False
batch_samps = 10_000_000
import jax
jax.config.update('jax_enable_x64',True)
import jax.random as random

import numpyro
import numpyro.distributions as dist

from jax import random

import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.stats import norm

from jax.lib import xla_bridge


from tqdm.notebook import trange
import itertools
import numpy.random as npr
itercount = itertools.count()


itercount = itertools.count()

import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow,masked_autoregressive_flow

"""Training loops."""
import copy
from collections.abc import Callable

import equinox as eqx
import optax
import paramax
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm
import pickle
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)




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

_, opt_key= jr.split(jr.key(999))
opt_key,___= jr.split(opt_key)

        
key, rng= jr.split(jr.key(0))

from numpyro.infer import MCMC, NUTS

safe_log = lambda x: jnp.log(jnp.clip(x, a_min=1e-8, a_max=None))

X_load=onp.load('SNANA_training_set.npy')
z_hel = X_load[:,0]
z_hd = X_load[:,1]
z_hd_err = X_load[:,2]

d_hat_sim = X_load[:,3:6]



log_x_err = safe_log(X_load[:,8])
log_c_err =  safe_log(X_load[:,7])
log_mag_err = safe_log(X_load[:,6])




from copy import deepcopy
def transform_cov(cov):
    #cov = deepcopy(cov)
    #cov[cov>0] = safe_log(1+np.absolute(cov[cov>0]))
    #cov[cov<0] = -safe_log(1+np.absolute(cov[cov<0]))
    return cov
    

cov_m_c = transform_cov(X_load[:,9])
cov_m_x = transform_cov(X_load[:,10])
cov_c_x=transform_cov(X_load[:,11])


m0 = X_load[:,12]
alpha = X_load[:,13]
beta = X_load[:,14]


X_load =0

X= jnp.append(d_hat_sim, jnp.column_stack((m0,alpha,beta,log_mag_err, log_c_err,log_x_err,cov_m_c,cov_m_x,cov_c_x,z_hel)),axis=1)
#X= jnp.append(d_hat_sim, jnp.column_stack((m0,alpha,beta,log_mag_err, log_c_err,log_x_err)),axis=1)
X =jax.random.permutation(key,X, axis=0, independent=False)

X = X.at[:,0].set(X[:,0]-X[:,3])



z_hel = 0
z_hd = 0

d_hat_sim = 0



log_x_err = 0
log_c_err =  0
log_mag_err = 0

cov_m_c = 0
cov_m_x = 0
cov_c_x=0


m0 = 0
alpha =0
beta = 0


if restart:
    if std_norm:
        file_=np.load(name+'_std.npz')
        mu_ = file_['mu']
        std_ = file_['std']
        add_ = jnp.sum(safe_log((std_[:3])))
    else:
        file_=np.load(name+'_minmax.npz') 
        min_=file_['min']
        max_=file_['max']
        add_ = jnp.sum(safe_log((max_-min_)[:3]))
    
else:
    if std_norm:
       X,mu_,std_ = std_fit_and_scale(X)
       np.savez(name+'_std.npz',mu=mu_,std=std_)
       add_ = jnp.sum(safe_log(std_[:3]))
    else:
        X,min_,max_  = minmax_fit_and_scale(X)
        np.savez(name+'_minmax.npz',min=min_,max=max_)
        add_ = jnp.sum(safe_log((max_-min_))[:3])


   


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,  # Custom losses may support broader types than AbstractDistribution
    loss_fn: Callable | None = None,
    max_epochs: int = 100,
    max_patience: int = args.max_pat,
    batch_size: int = 100,
    val_prop: float = args.val_prop,
    return_best: bool = True,
    show_progress: bool = True,params=None,opt_state=None,opt_key=None,counter=None,lr_schedule=None,big_losses=None
):
    r"""Train a PyTree (e.g. a distribution) to samples from the target.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        x: Samples from target distribution.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        condition: Conditioning variables. Defaults to None.
        loss_fn: Loss function. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    if save_all:
        save_thresh = 1
    else:
        save_thresh = 100000

    optimizer = optax.adamw(learning_rate=lr_schedule(counter),weight_decay=1e-4)
    current_lr = float(lr_schedule(counter))
    epochs_counter =0
    for batch_count in range(1):

        key, rng= jr.split(key)
       
        X_batch=jax.random.permutation(jr.key(1),X[batch_count*batch_samps:(batch_count+1)*batch_samps], axis=0, independent=False)


        x = X_batch[:,:3]
        
        condition = X_batch[:,3:]
    
        data = (x,) if condition is None else (x, condition)
        data = tuple(jnp.asarray(a) for a in data)

        if loss_fn is None:
            loss_fn = MaximumLikelihoodLoss()


        best_params = params

        # train val split
        key, subkey = jr.split(key)
        train_data, val_data = train_val_split(jr.key(1), data, val_prop=val_prop)
        losses = {"train": [], "val": []}

        loop = tqdm(range(max_epochs), disable=not show_progress)

        for _ in loop:
            # Shuffle data
            key, *subkeys = jr.split(key, 3)
            train_data = [jr.permutation(subkeys[0], a) for a in train_data]
            val_data = [jr.permutation(subkeys[1], a) for a in val_data]
            # Train epoch
            batch_losses = []
            for batch in zip(*get_batches(train_data, batch_size), strict=True):
                opt_key,___= jr.split(opt_key)
                params, opt_state, loss_i = step(
                    params,
                    static,
                    *batch,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_fn=loss_fn,
                    key=opt_key,
                )
                counter+=1
                loss_i = loss_i + add_
                batch_losses.append(loss_i)
                if counter%2000==0:
                    current_lr = float(lr_schedule(counter))
                    optimizer = optax.adamw(learning_rate=lr_schedule(counter),weight_decay=1e-4)
            epochs_counter+=1
            losses["train"].append((sum(batch_losses) / len(batch_losses)).item())
            big_losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

            # Val epoch
            batch_losses = []
            for batch in zip(*get_batches(val_data, batch_size), strict=True):
                opt_key,___= jr.split(opt_key)
                loss_i = loss_fn(params, static, *batch, key=opt_key)
                loss_i = loss_i +add_
                batch_losses.append(loss_i)
            losses["val"].append((sum(batch_losses) / len(batch_losses)).item())
            big_losses["val"].append((sum(batch_losses) / len(batch_losses)).item())
            big_losses["lr"].append(current_lr)
            loop.set_postfix({k: v[-1] for k, v in losses.items()})
            if big_losses["val"][-1] == min(losses["val"]):
                best_params = params
                best_opt_state = copy.deepcopy(opt_state)

            
            if count_fruitless(big_losses["val"]) > save_thresh:
                params = best_params if return_best else params
                dist = eqx.combine(params, static)
                eqx.tree_serialise_leaves(name+'_p'+str(save_thresh)+'.eqx',dist)
                onp.savez(name+'_p'+str(save_thresh)+'_std.npz',mu=mu_,std=std_)
                if args.make_slurm:
                    cmd_ = "python3 make_slurm.py --name="+name+'_p'+str(save_thresh)+" --nn_width="+str(args.nn_width)+' --nn_depth='+str(args.nn_depth)+' --no_flows='+str(args.no_flows)
                    print('submitted ',name+'_p'+str(save_thresh))
                    subprocess.run(cmd_, shell=True)
                save_thresh=int(count_fruitless(big_losses["val"]))

            if count_fruitless(big_losses["val"]) > max_patience:
                params = best_params if return_best else params
                dist = eqx.combine(params, static)
                eqx.tree_serialise_leaves(name+'.eqx',dist)
                loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
                with open(name+".pkl", "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(lambda x: x, best_opt_state), f)

                break

            if epochs_counter%10 ==0:
                dist = eqx.combine(best_params, static)
                eqx.tree_serialise_leaves(name+'.eqx',dist)
                onp.savez(name+'_logs.npz',train=big_losses['train'],val=big_losses['val'],lr=big_losses['lr'])
                with open(name+".pkl", "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(lambda x: x, best_opt_state), f)

 
        params = best_params if return_best else params
        dist = eqx.combine(params, static)
        eqx.tree_serialise_leaves(name+'.eqx',dist)
        onp.savez(name+'_logs.npz',train=big_losses['train'],val=big_losses['val'],lr=big_losses['lr'])
        with open(name+".pkl", "wb") as f:
            pickle.dump(jax.tree_util.tree_map(lambda x: x, best_opt_state), f)

    return dist, big_losses, params,opt_state,counter


key, rng= jr.split(key)



flow = masked_autoregressive_flow(
    key=key,
    base_dist=Normal(jnp.zeros(3)),
    cond_dim=X.shape[1]-3,nn_activation= jax.nn.gelu,flow_layers=args.no_flows,
nn_width=args.nn_width,
nn_depth=args.nn_depth,
)

if restart:
    flow= eqx.tree_deserialise_leaves(name+'.eqx',flow)

key, rng= jr.split(key)
# Define schedule parameters
warmup_steps = 5000  # Number of warmup steps (~5% of total)
total_steps = 100000  # Approximate number of total training steps
lr_max = 2e-4  # Peak learning rate
lr_min = 5e-5  # Minimum learning rate (prevents zero update)

if restart:
    lr_schedule =optax.constant_schedule(lr_min)

else:    
    # Define warmup schedule (linear increase)
    warmup_fn = optax.linear_schedule(
        init_value=1e-6, end_value=lr_max, transition_steps=warmup_steps
    )

    # Define cosine decay schedule
    cosine_fn = optax.cosine_decay_schedule(
        init_value=lr_max, decay_steps=total_steps - warmup_steps, alpha=lr_min / lr_max
    )

    # Combine schedules
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )


optimizer = optax.adamw(learning_rate=lr_schedule(0), weight_decay=1e-4)
current_lr = float(lr_schedule(0))
params, static = eqx.partition(flow, eqx.is_inexact_array, is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),)
opt_state = optimizer.init(params)
if restart:
    with open(name+".pkl", "rb") as f:
        opt_state = pickle.load(f)

    

counter =0

train_logs =onp.array([])
val_logs = onp.array([])
lr_logs = onp.array([])
print(jnp.any(jnp.isnan(X)))
print(jnp.any(jnp.isinf(X)))

eval_score=-100.0
val_score = 1000
loss = []
val_mean = onp.array([])

if restart:
    log_file=onp.load(name+'_logs.npz')
    big_losses = {"train":list(log_file['train']),"val":list(log_file['val']),"lr":list(log_file['lr'])}
else:
    big_losses = {"train": [], "val": [],"lr":[]}



opt_key,___= jr.split(opt_key)
flow, losses,params,opt_state,counter = fit_to_data(
    key=key,
    dist=flow,
    max_patience=args.max_pat,
    max_epochs=args.epochs,
    batch_size=args.batch_size,params=params,opt_state=opt_state,opt_key=opt_key,lr_schedule=lr_schedule,counter=counter,big_losses=big_losses)

  

