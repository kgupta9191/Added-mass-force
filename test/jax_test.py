import pytest
import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import optax

# ---------------------------------------------------------------------------
# Functions mirroring src/jax_code.py — tested in isolation so that the
# training script's top-level I/O code does not need to execute.
# ---------------------------------------------------------------------------

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
        W = random.normal(k, (m, n)) * jnp.sqrt(2 / m)  # He init
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


def make_update_fn(optimizer):
    """Return a JIT-compiled update step for the given optimizer."""

    @jit
    def update(params, opt_state, X, y):
        grads = jax.grad(loss_fn)(params, X, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    return update


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

KEY = random.PRNGKey(42)
LAYER_SIZES = [3, 128, 128, 64, 1]


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_yields_correct_feature_dim(self):
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        key = random.PRNGKey(0)
        for xb, _ in data_loader(X, y, batch_size=32, key=key):
            assert xb.shape[1] == 3

    def test_batch_x_y_length_match(self):
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        key = random.PRNGKey(0)
        for xb, yb in data_loader(X, y, batch_size=32, key=key):
            assert xb.shape[0] == yb.shape[0]

    def test_full_data_covered(self):
        X = np.random.randn(64, 3).astype(np.float32)
        y = np.random.randn(64).astype(np.float32)
        key = random.PRNGKey(1)
        total = sum(len(xb) for xb, _ in data_loader(X, y, batch_size=16, key=key))
        assert total == 64

    def test_batch_count_with_remainder(self):
        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        key = random.PRNGKey(2)
        batches = list(data_loader(X, y, batch_size=4, key=key))
        # 10 samples / batch_size 4 → 3 batches (last has 2 samples)
        assert len(batches) == 3

    def test_different_keys_produce_different_order(self):
        X = np.arange(20, dtype=np.float32).reshape(20, 1)
        y = np.zeros(20, dtype=np.float32)
        batches_a = list(data_loader(X, y, batch_size=20, key=random.PRNGKey(0)))
        batches_b = list(data_loader(X, y, batch_size=20, key=random.PRNGKey(1)))
        xb_a, _ = batches_a[0]
        xb_b, _ = batches_b[0]
        assert not jnp.array_equal(xb_a, xb_b)


# ---------------------------------------------------------------------------
# init_mlp tests
# ---------------------------------------------------------------------------

class TestInitMLP:
    def test_param_count(self):
        params = init_mlp(LAYER_SIZES, KEY)
        assert len(params) == len(LAYER_SIZES) - 1

    def test_weight_shapes(self):
        params = init_mlp(LAYER_SIZES, KEY)
        expected = list(zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]))
        for (W, b), (m, n) in zip(params, expected):
            assert W.shape == (m, n), f"Expected W shape ({m},{n}), got {W.shape}"
            assert b.shape == (n,), f"Expected b shape ({n},), got {b.shape}"

    def test_biases_initialized_to_zero(self):
        params = init_mlp(LAYER_SIZES, KEY)
        for _, b in params:
            assert jnp.allclose(b, 0.0)

    def test_different_keys_give_different_weights(self):
        params1 = init_mlp(LAYER_SIZES, random.PRNGKey(0))
        params2 = init_mlp(LAYER_SIZES, random.PRNGKey(1))
        W1, _ = params1[0]
        W2, _ = params2[0]
        assert not jnp.allclose(W1, W2)

    def test_single_layer_network(self):
        params = init_mlp([4, 2], KEY)
        assert len(params) == 1
        W, b = params[0]
        assert W.shape == (4, 2)
        assert b.shape == (2,)


# ---------------------------------------------------------------------------
# mlp forward-pass tests
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape_single_sample(self):
        params = init_mlp(LAYER_SIZES, KEY)
        x = jnp.ones((1, 3))
        out = mlp(params, x)
        assert out.shape == (1, 1)

    def test_output_shape_batch(self):
        params = init_mlp(LAYER_SIZES, KEY)
        x = jnp.ones((32, 3))
        out = mlp(params, x)
        assert out.shape == (32, 1)

    def test_forward_pass_is_deterministic(self):
        params = init_mlp(LAYER_SIZES, KEY)
        x = jnp.ones((8, 3))
        out1 = mlp(params, x)
        out2 = mlp(params, x)
        assert jnp.allclose(out1, out2)

    def test_relu_hidden_activations_non_negative(self):
        """Hidden units after ReLU must be >= 0."""
        params = init_mlp([3, 8, 1], KEY)
        x = jnp.array([[-1.0, -2.0, -3.0]])
        W0, b0 = params[0]
        pre = jnp.dot(x, W0) + b0
        post_relu = jax.nn.relu(pre)
        assert jnp.all(post_relu >= 0)

    def test_zero_weights_give_zero_output(self):
        params = [(jnp.zeros((3, 8)), jnp.zeros(8)),
                  (jnp.zeros((8, 1)), jnp.zeros(1))]
        x = jnp.ones((5, 3))
        out = mlp(params, x)
        assert jnp.allclose(out, 0.0)


# ---------------------------------------------------------------------------
# loss_fn tests
# ---------------------------------------------------------------------------

class TestLossFn:
    def test_zero_loss_on_perfect_prediction(self):
        # Single-layer net with bias=5 and zero weights → always predicts 5
        params = [(jnp.zeros((3, 1)), jnp.array([5.0]))]
        X = jnp.ones((10, 3))
        y = jnp.full((10,), 5.0)
        loss = loss_fn(params, X, y)
        assert jnp.isclose(loss, 0.0, atol=1e-5)

    def test_loss_is_non_negative(self):
        params = init_mlp(LAYER_SIZES, KEY)
        X = jnp.ones((20, 3))
        y = jnp.zeros(20)
        assert loss_fn(params, X, y) >= 0.0

    def test_loss_is_scalar(self):
        params = init_mlp(LAYER_SIZES, KEY)
        X = jnp.ones((16, 3))
        y = jnp.zeros(16)
        assert loss_fn(params, X, y).shape == ()

    def test_larger_residual_gives_larger_loss(self):
        # Zero-output network: prediction is always 0
        params = [(jnp.zeros((3, 1)), jnp.array([0.0]))]
        X = jnp.ones((10, 3))
        loss_close = loss_fn(params, X, jnp.ones(10) * 0.1)
        loss_far = loss_fn(params, X, jnp.ones(10) * 10.0)
        assert loss_far > loss_close

    def test_gradient_is_finite(self):
        params = init_mlp(LAYER_SIZES, KEY)
        X = jnp.ones((8, 3))
        y = jnp.zeros(8)
        grads = jax.grad(loss_fn)(params, X, y)
        for dW, db in grads:
            assert jnp.all(jnp.isfinite(dW))
            assert jnp.all(jnp.isfinite(db))


# ---------------------------------------------------------------------------
# update (training step) tests
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_params_change_after_update(self):
        params = init_mlp(LAYER_SIZES, KEY)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        update = make_update_fn(optimizer)

        X = jnp.ones((16, 3))
        y = jnp.ones(16)
        new_params, _ = update(params, opt_state, X, y)

        W_before, _ = params[0]
        W_after, _ = new_params[0]
        assert not jnp.allclose(W_before, W_after)

    def test_opt_state_changes_after_update(self):
        params = init_mlp([3, 4, 1], KEY)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        update = make_update_fn(optimizer)

        X = jnp.ones((8, 3))
        y = jnp.ones(8)
        _, new_opt_state = update(params, opt_state, X, y)
        # Adam tracks first and second moments; they should be non-zero after 1 step
        mu = new_opt_state[0].mu  # first moment
        first_layer_mu_W = mu[0][0]
        assert not jnp.allclose(first_layer_mu_W, 0.0)

    def test_loss_decreases_over_multiple_updates(self):
        params = init_mlp(LAYER_SIZES, KEY)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        update = make_update_fn(optimizer)

        X = jnp.ones((32, 3))
        y = jnp.ones(32)
        initial_loss = loss_fn(params, X, y)

        for _ in range(50):
            params, opt_state = update(params, opt_state, X, y)

        final_loss = loss_fn(params, X, y)
        assert final_loss < initial_loss
