# Load module
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
import pandas as pd
import optax 

# Load data
data_loaded = pd.read_csv("complex_regression_data.csv")
X = data_loaded.iloc[:, :-1].values 
y = data_loaded.iloc[:, -1].values 

# Data splitting
N = n_rows
indices = np.random.permutation(N)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train+n_val]
test_idx  = indices[n_train+n_val:]
X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# Neural architecture
batch_size = 128

def data_loader(X, y, batch_size, key):
    N = X.shape[0]
    perm = random.permutation(key, N)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
        
def init_mlp(layer_sizes, key):
    params = []
    keys = random.split(key, len(layer_sizes))
    for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        W = random.normal(k, (m, n)) * jnp.sqrt(2/m)  # He init
        b = jnp.zeros(n)
        params.append((W, b))
    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last

def loss_fn(params, X, y):
    y_pred = mlp(params, X).squeeze()
    return jnp.mean((y_pred - y) ** 2)

key = random.PRNGKey(0)
params = init_mlp([3, 128, 128, 64, 1], key)

# Optax optimizer
optimizer = optax.adam(1e-5)
opt_state = optimizer.init(params)

@jit
def update(params, opt_state, X, y):
    grads = jax.grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

best_val = jnp.inf
best_params = params
patience = 500
wait = 0
epochs = 5000

key = random.PRNGKey(0)

for epoch in range(epochs):
    key, subkey = random.split(key)

    # 🔹 Mini-batch training
    for X_batch, y_batch in data_loader(X_train, y_train, batch_size, subkey):
        params, opt_state = update(params, opt_state, X_batch, y_batch)

    # 🔹 Validation (full batch)
    if epoch % 100 == 0:
        train_loss = loss_fn(params, X_train, y_train)
        val_loss   = loss_fn(params, X_val, y_val)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4e}, Val Loss = {val_loss:.4e}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_params = params
            wait = 0
        else:
            wait += 100

        if wait >= patience:
            print("Early stopping triggered")
            break
