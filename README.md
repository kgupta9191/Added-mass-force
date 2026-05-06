# Added Mass Force Prediction

> A machine learning surrogate for predicting added mass forces in fluid–structure interaction problems. Two fully independent implementations are provided — one in **JAX** and one in **PyTorch** — so you can benchmark frameworks or build on whichever stack you already use.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)](https://pytorch.org/)

---

## Abstract

The prediction of added mass force during fluid–structure interaction (FSI) problems — such as water entry and slamming — is computationally expensive using traditional numerical methods. This project presents a data-driven machine learning approach that uses a deep neural network (multi-layer perceptron) to model and predict added mass forces from input flow and geometric parameters. The model is trained on simulated or experimental data to learn the nonlinear mapping between input features and resulting hydrodynamic forces. High accuracy is achieved at a fraction of the cost of a full CFD solve, demonstrating the potential of surrogate modelling for fast, real-time prediction in complex fluid dynamics problems.

---

## Motivation

In FSI problems such as water entry, slamming, and wave impact, **added mass** plays a crucial role. It represents the extra inertia a body experiences due to the accelerating surrounding fluid — and computing it accurately is essential for structural design and dynamic response analysis.

**Traditional approaches are bottlenecked:**

| Method | Problem |
|---|---|
| CFD (Navier–Stokes solvers) | Computationally expensive; hours to days per run |
| Parametric CFD studies | Infeasible at scale |
| Experimental measurement | Difficult for transient, high-speed events; requires sophisticated rigs |

**This project addresses those bottlenecks by providing:**

1. A fast surrogate model trained once on existing data
2. Real-time prediction capability (milliseconds per query)
3. A dramatically reduced computational cost for parametric sweeps

---

## What Is It?

This repository contains two complete, independent implementations of the same MLP surrogate:

| File | Framework | Optimizer | Notes |
|---|---|---|---|
| `torch_code.py` | PyTorch | Adam | `torch.compile` + CUDA support, DataLoader with multi-worker prefetch |
| `jax_code.py` | JAX + Optax | Adam | JIT-compiled update step, functional-style mini-batch loop |

Both implementations share the same:
- **Network architecture** — `[3 → 128 → 128 → 64 → 1]` MLP with ReLU activations
- **Data split** — 80 % train / 10 % validation / 10 % test
- **Training schedule** — up to 5 000 epochs, early stopping with patience 500, logging every 100 epochs
- **Learning rate** — `1e-5` with Adam

---

## Repository Structure

```
Added-mass-force/
├── torch_code.py          # PyTorch implementation
├── jax_code.py            # JAX + Optax implementation
├── pytorch_report.pdf     # Training results & analysis (PyTorch)
├── jax_report.pdf         # Training results & analysis (JAX)
└── LICENSE
```

> **Data file** — both scripts expect `complex_regression_data.csv` in the working directory (3 input feature columns + 1 target column). This file is not included in the repository; supply your own FSI/added-mass dataset.

---

## Model Architecture

Both implementations use the same feed-forward MLP:

```
Input (3)  →  Linear(128) → ReLU
           →  Linear(128) → ReLU
           →  Linear(64)  → ReLU
           →  Linear(1)         ← scalar added-mass force prediction
```

- **He (Kaiming) weight initialisation** for stable ReLU training
- **MSE loss** on the scalar output
- **Early stopping** on validation loss (patience = 500 epochs)

---

## Installation

### Prerequisites

- Python 3.9+
- A CSV dataset (`complex_regression_data.csv`) with 3 feature columns and 1 target column

### PyTorch

```bash
pip install torch pandas numpy
```

GPU acceleration is used automatically when a CUDA-capable device is available.

### JAX

```bash
pip install jax jaxlib optax pandas numpy
```

For GPU/TPU support follow the [JAX installation guide](https://github.com/google/jax#installation).

---

## Usage

### PyTorch

```bash
python torch_code.py
```

The script will:
1. Load and split `complex_regression_data.csv` (80/10/10)
2. Build and compile the MLP (`torch.compile` with the `inductor` backend)
3. Train for up to 5 000 epochs with early stopping
4. Save the best checkpoint to `best_model.pth`

```
Epoch 0:   Test Loss = 1.2345e-01  Val Loss = 1.2300e-01
Epoch 100: Test Loss = 8.4321e-03  Val Loss = 8.1234e-03
...
Early stopping triggered
```

### JAX

```bash
python jax_code.py
```

The script will:
1. Load and split `complex_regression_data.csv` (80/10/10)
2. Initialise the MLP parameters with He initialisation
3. Run JIT-compiled mini-batch updates (batch size 128) for up to 5 000 epochs
4. Apply early stopping on validation MSE

```
Epoch 0:    Train Loss = 1.3456e-01, Val Loss = 1.3200e-01
Epoch 100:  Train Loss = 9.1234e-03, Val Loss = 9.0012e-03
...
Early stopping triggered
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Architecture | `[3, 128, 128, 64, 1]` MLP |
| Activation | ReLU |
| Weight init | He (Kaiming) |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Batch size | 64 (PyTorch) / 128 (JAX) |
| Max epochs | 5 000 |
| Early-stopping patience | 500 epochs |
| Train / Val / Test split | 80 % / 10 % / 10 % |
| Loss function | Mean Squared Error (MSE) |

---

## Results

Full training curves, loss plots, and analysis are available in the companion PDF reports:

- [`pytorch_report.pdf`](pytorch_report.pdf) — PyTorch training results
- [`jax_report.pdf`](jax_report.pdf) — JAX training results

Both frameworks converge to comparable validation loss, confirming framework-agnostic reproducibility of the surrogate model.

---

## Framework Comparison

| Feature | PyTorch | JAX |
|---|---|---|
| GPU acceleration | ✅ CUDA via `torch.device` | ✅ via `jaxlib` |
| Compiler optimisation | ✅ `torch.compile` (Inductor) | ✅ `@jit` |
| Functional API | Partial | ✅ Native |
| Ecosystem maturity | ✅ Very mature | Growing |
| Preferred for | Production / fine-tuning | Research / custom gradients |

---

## Extending the Model

- **More input features** — update the first `Linear` layer size in `torch_code.py` or the `layer_sizes` list in `jax_code.py`
- **Deeper/wider network** — add layers to `nn.Sequential` (PyTorch) or the `layer_sizes` list (JAX)
- **Different loss** — swap `nn.MSELoss()` / `jnp.mean((y_pred - y)**2)` for MAE, Huber, etc.
- **Hyperparameter search** — wrap training loops with Optuna or Ray Tune

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

---

## License

[MIT](LICENSE)

