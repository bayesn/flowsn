import argparse
import pickle
import copy
from dataclasses import dataclass
from pathlib import Path

SNANA_DIR = Path(__file__).resolve().parent

import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import optax
import paramax

from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)



# =========================
# Config
# =========================

@dataclass
class Config:
    name: str
    nn_width: int
    nn_depth: int
    no_flows: int
    max_pat: int
    val_prop: float
    batch_size: int
    epochs: int
    save_all: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="base_name")
    parser.add_argument("--nn_width", type=int, default=32)
    parser.add_argument("--nn_depth", type=int, default=2)
    parser.add_argument("--no_flows", type=int, default=4)
    parser.add_argument("--max_pat", type=int, default=10)
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8092)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_all", action="store_true")

    args = parser.parse_args()

    name = args.name
    if name == "base_name":
        name = (
            f"base_w{args.nn_width}_d{args.nn_depth}_f{args.no_flows}"
            f"_vp{onp.around(args.val_prop,2)}_bs{args.batch_size}"
        )
        if not args.save_all:
            name += f"_p{args.max_pat}"

    return Config(
        name=name,
        nn_width=args.nn_width,
        nn_depth=args.nn_depth,
        no_flows=args.no_flows,
        max_pat=args.max_pat,
        val_prop=args.val_prop,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_all=args.save_all,
    )


# =========================
# Utils
# =========================

@jax.jit
def standardize(X):
    mu = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    return (X - mu) / std, mu, std


def safe_log(x):
    return jnp.log(jnp.clip(x, a_min=1e-8))


# =========================
# Data
# =========================

def load_data(path="SNANA_training_set.npy"):
    X = onp.load(path)

    d_hat = X[:, 3:6]
    m0, alpha, beta = X[:, 12], X[:, 13], X[:, 14]

    log_errs = jnp.column_stack(
        (
            safe_log(X[:, 6]),
            safe_log(X[:, 7]),
            safe_log(X[:, 8]),
        )
    )

    cov = jnp.column_stack((X[:, 9], X[:, 10], X[:, 11]))
    z_hel = X[:, 0]

    features = jnp.column_stack(
        (d_hat, m0, alpha, beta, log_errs, cov, z_hel)
    )

    return features


def preprocess_data(X, key):
    X = jr.permutation(key, X, axis=0)
    X = X.at[:, 0].set(X[:, 0] - X[:, 3])  # domain-specific shift

    X, mu, std = standardize(X)
    add_term = jnp.sum(safe_log(std[:3]))

    return X, mu, std, add_term


# =========================
# Model
# =========================

def build_flow(cfg: Config, key, cond_dim):
    return masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(3)),
        cond_dim=cond_dim,
        nn_activation=jax.nn.gelu,
        flow_layers=cfg.no_flows,
        nn_width=cfg.nn_width,
        nn_depth=cfg.nn_depth,
    )


# =========================
# LR schedule
# =========================

def make_schedule():
    warmup = optax.linear_schedule(1e-6, 2e-4, 5000)
    cosine = optax.cosine_decay_schedule(2e-4, 100000 - 5000, alpha=0.25)
    return optax.join_schedules([warmup, cosine], [5000])


# =========================
# Training
# =========================

def train(
    key,
    flow,
    X,
    cfg: Config,
    lr_schedule,
    add_term,
    save_path,
):

    params, static = eqx.partition(
        flow,
        eqx.is_inexact_array,
        is_leaf=lambda x: isinstance(x, paramax.NonTrainable),
    )

    opt = optax.adamw(lr_schedule(0), weight_decay=1e-4)
    opt_state = opt.init(params)

    logs = {"train": [], "val": [], "lr": []}
    best_params = params
    best_opt_state = opt_state

    x = X[:, :3]
    cond = X[:, 3:]

    key, split_key = jr.split(key)
    train_data, val_data = train_val_split(split_key, (x, cond), cfg.val_prop)

    n_train = train_data[0].shape[0]
    n_val = val_data[0].shape[0]
    n_train_batches = max(1, n_train // cfg.batch_size)
    print(f"\n{'='*60}")
    print(f"  Training samples : {n_train:,}  ({n_train_batches} batches)")
    print(f"  Val samples      : {n_val:,}")
    print(f"  Max epochs       : {cfg.epochs}  |  Patience: {cfg.max_pat}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'Best':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    loss_fn = MaximumLikelihoodLoss()
    counter = 0

    for epoch in range(cfg.epochs):
        key, k1, k2 = jr.split(key, 3)

        train_data = [jr.permutation(k1, a) for a in train_data]
        val_data = [jr.permutation(k2, a) for a in val_data]

        # ---- Train ----
        train_losses = []
        for batch in zip(*get_batches(train_data, cfg.batch_size), strict=True):
            key, step_key = jr.split(key)

            params, opt_state, loss = step(
                params,
                static,
                *batch,
                optimizer=opt,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=step_key,
            )

            loss = loss + add_term
            train_losses.append(loss)

            counter += 1
            if counter % 2000 == 0:
                opt = optax.adamw(lr_schedule(counter), weight_decay=1e-4)

        train_loss = jnp.mean(jnp.array(train_losses)).item()

        # ---- Val ----
        val_losses = []
        for batch in zip(*get_batches(val_data, cfg.batch_size), strict=True):
            key, val_key = jr.split(key)
            loss = loss_fn(params, static, *batch, key=val_key)
            val_losses.append(loss + add_term)

        val_loss = jnp.mean(jnp.array(val_losses)).item()

        logs["train"].append(train_loss)
        logs["val"].append(val_loss)
        logs["lr"].append(float(lr_schedule(counter)))

        is_best = val_loss <= min(logs["val"])
        if is_best:
            best_params = params
            best_opt_state = copy.deepcopy(opt_state)

        fruitless = count_fruitless(logs["val"])
        best_marker = "  <--" if is_best else f"  [{fruitless}/{cfg.max_pat}]"

        print(
            f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}"
            f"  {lr_schedule(counter):>10.2e}{best_marker}",
            flush=True,
        )

        if fruitless > cfg.max_pat:
            print(f"\n  Early stopping at epoch {epoch} (patience {cfg.max_pat} exceeded)")
            break

        # periodic checkpoint
        if epoch % 10 == 0:
            save_model(best_params, static, save_path)


    print(f"\n{'='*60}")
    print(f"  Training complete. Best val loss: {min(logs['val']):.4f}")
    print(f"  Saving to: {save_path}")
    print(f"{'='*60}\n")

    save_model(best_params, static, save_path)


    return eqx.combine(best_params, static), logs


# =========================
# I/O
# =========================

def save_model(params, static, name):
    eqx.tree_serialise_leaves(str(SNANA_DIR / "flow_weights" / f"{name}.eqx"), eqx.combine(params, static))



# =========================
# Main
# =========================

def main():
    cfg = parse_args()

    jax.config.update("jax_enable_x64", True)

    print(f"\nJAX devices : {jax.devices()}")
    print(f"Run name    : {cfg.name}")

    key = jr.key(0)

    # ---- Data ----
    print("\nLoading data...")
    X_raw = load_data(path=str(SNANA_DIR / "SNANA_training_set.npy"))
    print(f"  Raw data shape: {X_raw.shape}")

    key, subkey = jr.split(key)
    X, mu, std, add_term = preprocess_data(X_raw, subkey)
    print(f"  Preprocessed shape: {X.shape}  |  add_term: {add_term:.4f}")

    weights_dir = SNANA_DIR / "flow_weights"
    weights_dir.mkdir(exist_ok=True)
    onp.savez(str(weights_dir / (cfg.name + "_std.npz")), mu=mu, std=std)

    # ---- Model ----
    print("\nBuilding flow...")
    key, subkey = jr.split(key)
    flow = build_flow(cfg, subkey, cond_dim=X.shape[1] - 3)
    print(f"  Flows: {cfg.no_flows}  |  Width: {cfg.nn_width}  |  Depth: {cfg.nn_depth}  |  cond_dim: {X.shape[1] - 3}")

    # ---- Training ----
    lr_schedule = make_schedule()

    trained_flow, logs = train(
        key,
        flow,
        X,
        cfg,
        lr_schedule,
        add_term,
        cfg.name,
    )


if __name__ == "__main__":
    main()