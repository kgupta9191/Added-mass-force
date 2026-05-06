import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------------------------------------------------------------------
# MLP class mirroring src/torch_code.py — tested in isolation so that the
# training script's top-level I/O code does not need to execute.
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# MLP architecture tests
# ---------------------------------------------------------------------------

class TestMLPArchitecture:
    def test_output_shape_single_sample(self):
        model = MLP()
        x = torch.randn(1, 3)
        assert model(x).shape == (1, 1)

    def test_output_shape_batch(self):
        model = MLP()
        x = torch.randn(32, 3)
        assert model(x).shape == (32, 1)

    def test_linear_layer_count(self):
        model = MLP()
        linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 4

    def test_linear_layer_dimensions(self):
        model = MLP()
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        expected = [(3, 128), (128, 128), (128, 64), (64, 1)]
        for layer, (in_f, out_f) in zip(linears, expected):
            assert layer.in_features == in_f, (
                f"Expected in_features={in_f}, got {layer.in_features}"
            )
            assert layer.out_features == out_f, (
                f"Expected out_features={out_f}, got {layer.out_features}"
            )

    def test_relu_activation_count(self):
        model = MLP()
        relu_layers = [m for m in model.net if isinstance(m, nn.ReLU)]
        assert len(relu_layers) == 3

    def test_forward_is_deterministic(self):
        model = MLP()
        model.eval()
        x = torch.randn(8, 3)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_parameter_count_is_positive(self):
        model = MLP()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# MLP training tests
# ---------------------------------------------------------------------------

class TestMLPTraining:
    def test_loss_decreases_with_training(self):
        torch.manual_seed(42)
        model = MLP().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X = torch.randn(64, 3)
        y = torch.randn(64, 1)

        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X), y).item()

        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < initial_loss

    def test_gradients_flow_through_all_layers(self):
        torch.manual_seed(42)
        model = MLP().to(DEVICE)
        criterion = nn.MSELoss()

        X = torch.randn(16, 3)
        y = torch.randn(16, 1)

        loss = criterion(model(X), y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_model_state_dict_save_and_load(self, tmp_path):
        model = MLP()
        save_path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), save_path)

        loaded_model = MLP()
        loaded_model.load_state_dict(
            torch.load(save_path, weights_only=True)
        )

        model.eval()
        loaded_model.eval()
        x = torch.randn(4, 3)
        assert torch.allclose(model(x), loaded_model(x))

    def test_train_eval_mode_toggle(self):
        model = MLP()
        model.train()
        assert model.training
        model.eval()
        assert not model.training

    def test_zero_grad_clears_gradients(self):
        torch.manual_seed(0)
        model = MLP()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X = torch.randn(8, 3)
        y = torch.randn(8, 1)
        loss = criterion(model(X), y)
        loss.backward()

        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


# ---------------------------------------------------------------------------
# Data pipeline tests (TensorDataset / DataLoader / random_split)
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_dataset_split_sizes(self):
        torch.manual_seed(42)
        N = 100
        X = torch.randn(N, 3)
        y = torch.randn(N, 1)
        dataset = TensorDataset(X, y)

        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        n_test = N - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        assert len(train_ds) == n_train
        assert len(val_ds) == n_val
        assert len(test_ds) == n_test
        assert len(train_ds) + len(val_ds) + len(test_ds) == N

    def test_dataloader_batch_feature_shape(self):
        X = torch.randn(64, 3)
        y = torch.randn(64, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
        for xb, yb in loader:
            assert xb.shape[1] == 3
            assert yb.shape[1] == 1
            assert xb.shape[0] == yb.shape[0]

    def test_dataloader_covers_all_samples(self):
        N = 100
        X = torch.randn(N, 3)
        y = torch.randn(N, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
        total = sum(xb.shape[0] for xb, _ in loader)
        assert total == N

    def test_dataloader_shuffle_changes_order(self):
        torch.manual_seed(0)
        N = 64
        X = torch.arange(N, dtype=torch.float32).unsqueeze(1).expand(-1, 3)
        y = torch.zeros(N, 1)
        loader_a = DataLoader(
            TensorDataset(X, y), batch_size=N, shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        loader_b = DataLoader(
            TensorDataset(X, y), batch_size=N, shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        xb_a, _ = next(iter(loader_a))
        xb_b, _ = next(iter(loader_b))
        assert not torch.equal(xb_a, xb_b)


# ---------------------------------------------------------------------------
# MSELoss tests
# ---------------------------------------------------------------------------

class TestMSELoss:
    def test_zero_loss_on_perfect_prediction(self):
        criterion = nn.MSELoss()
        y = torch.tensor([[1.0], [2.0], [3.0]])
        assert torch.isclose(criterion(y, y), torch.tensor(0.0))

    def test_loss_is_non_negative(self):
        criterion = nn.MSELoss()
        y_pred = torch.randn(10, 1)
        y_true = torch.randn(10, 1)
        assert criterion(y_pred, y_true).item() >= 0.0

    def test_loss_is_scalar(self):
        criterion = nn.MSELoss()
        y_pred = torch.randn(16, 1)
        y_true = torch.randn(16, 1)
        assert criterion(y_pred, y_true).shape == torch.Size([])

    def test_larger_residual_gives_larger_loss(self):
        criterion = nn.MSELoss()
        y_pred = torch.zeros(10, 1)
        y_close = torch.ones(10, 1) * 0.1
        y_far = torch.ones(10, 1) * 10.0
        assert criterion(y_pred, y_far) > criterion(y_pred, y_close)
