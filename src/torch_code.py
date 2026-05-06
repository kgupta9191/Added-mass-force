# Load module
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import numpy as np

# Load data
data = torch.tensor(
    pd.read_csv("complex_regression_data.csv").values,
    dtype=torch.float32
)
X = data[:, :-1]
y = data[:, -1].unsqueeze(1)
dataset = TensorDataset(X, y)

# Data split
N = len(dataset)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val
train_ds, val_ds, test_ds = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)  # reproducible
)

# Neural Architecture
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Model, Optimizer, Loss function
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.compile(model, backend="inductor")  # or backend="nvfuser" on GPU
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

epochs = 5000
patience = 500
best_val = float('inf')
wait = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                val_loss += criterion(model(xb), yb).item()
            val_loss /= len(val_loader)


        print(f"Epoch {epoch}: Test Loss = {loss:.4e} Val Loss = {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 100

        if wait >= patience:
            print("Early stopping triggered")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
