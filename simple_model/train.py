#!/usr/bin/env python
import argparse
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import paramax
from tqdm import tqdm
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import get_batches, step, train_val_split
import os
current_dir = os.getcwd()
if 'simple_model' not in current_dir:
    current_dir+='/simple_model'
    
# Ensure 64-bit precision for physical consistency
jax.config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="sn_model")
    parser.add_argument("--data", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--nn_width", type=int, default=32)
    parser.add_argument("--nn_depth", type=int, default=2)
    parser.add_argument("--no_flows", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # 1. Load Data with robust pathing
    data_path = current_dir+'/training_data/'+ args.data if args.data.endswith('.npy') else current_dir+'/training_data/'+ args.data + '.npy'
    print(f"Loading {data_path}...")
    X = jnp.array(onp.load(data_path))
    
    # 2. Residual Calculation (Residual = Observation - Baseline)
    X = X.at[:, :3].set(X[:, :3] - X[:, 3:6])
    
    # 3. Standardization
    mu = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    X = (X - mu) / std
    onp.savez(f"{current_dir}/scaling/{args.name}_std.npz", mu=onp.array(mu), std=onp.array(std))
    
    # Jacobian correction for physical log-likelihood
    log_jac_adj = jnp.sum(jnp.log(std[:3]))
    
    # 4. Corrected Flow Initialization (Keyword arguments only)
    key = jr.PRNGKey(42)
    flow = masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(3)),
        cond_dim=15,
        nn_activation=jax.nn.gelu,
        flow_layers=args.no_flows,
        nn_width=args.nn_width,
        nn_depth=args.nn_depth,
    )
    
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate=2e-4, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    
    # 5. Training Loop
    data = (X[:, :3], X[:, 3:])
    train_data, val_data = train_val_split(jr.split(key)[0], data, val_prop=0.1)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        for batch in zip(*get_batches(train_data, args.batch_size), strict=True):
            params, opt_state, _ = step(params, static, *batch, 
                                        optimizer=optimizer, opt_state=opt_state, 
                                        loss_fn=MaximumLikelihoodLoss())
        
        # Validation loss including the Jacobian correction
        val_loss = MaximumLikelihoodLoss()(params, static, *val_data) + log_jac_adj
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

    # Final save
    eqx.tree_serialise_leaves(f"{current_dir}/weights/{args.name}.eqx", eqx.combine(params, static))
    print(f"Training complete. Model saved to {args.name}.eqx")

if __name__ == "__main__":
    main()