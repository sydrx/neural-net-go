# 🧠 neural-net-go

A neural network built **from scratch** in pure Go — no ML libraries, no external dependencies.

Implements a fully-connected feedforward network with backpropagation trained via stochastic gradient descent.

## Features

- ✅ Matrix operations (dot product, transpose, Hadamard, …)
- ✅ Dense (fully-connected) layers
- ✅ Activations: **Sigmoid**, **ReLU**, **Tanh**, **Linear**
- ✅ Losses: **MSE**, **Binary Cross-Entropy**
- ✅ Backpropagation with SGD
- ✅ Xavier weight initialisation
- ✅ Zero dependencies — only the Go standard library

## Quick start

```bash
git clone https://github.com/sydrx/neural-net-go
cd neural-net-go

# Run the XOR demo
go run ./examples/xor
```

Expected output:

```
Training (10 000 epochs)...

epoch  1000 | loss: 0.249821 [████░░░░░░░░░░░░░░░░]
epoch  2000 | loss: 0.164432 [██████░░░░░░░░░░░░░░]
...
epoch 10000 | loss: 0.001243 [████████████████████]

Results:
┌──────────┬──────────┬──────────┬──────────┐
│  Input   │ Expected │   Got    │  Status  │
├──────────┼──────────┼──────────┼──────────┤
│ [0, 0]   │    0     │  0.0231  │    ✓     │
│ [0, 1]   │    1     │  0.9812  │    ✓     │
│ [1, 0]   │    1     │  0.9798  │    ✓     │
│ [1, 1]   │    0     │  0.0187  │    ✓     │
└──────────┴──────────┴──────────┴──────────┘

Accuracy: 4/4 (100%)
```

## Architecture

```
nn/
├── matrix.go      # Matrix type: Dot, Transpose, Apply, …
├── activation.go  # Sigmoid, ReLU, Tanh, Linear
├── layer.go       # Dense layer (forward + backward)
├── loss.go        # MSE, BinaryCrossEntropy
└── network.go     # Sequential network, Train, Predict

examples/
└── xor/
    └── main.go    # XOR demo
```

## Usage

```go
rng := nn.RNG(42)

net := nn.NewNetwork(0.1, nn.MSE).
    Add(nn.NewDense(2, 16, nn.ReLU, rng)).
    Add(nn.NewDense(16, 8, nn.ReLU, rng)).
    Add(nn.NewDense(8, 1, nn.Sigmoid, rng))

net.Train(X, Y, epochs, printEvery)
pred := net.Predict(X)
```

## Math

Forward pass for each layer:

```
z = X · W + b
a = activation(z)
```

Backward pass (backpropagation):

```
δ = dLoss/dOut ⊙ activation'(z)
dW = Xᵀ · δ
db = Σ δ  (sum over batch)
dX = δ · Wᵀ
```

Weight update (SGD):

```
W -= lr · dW
b -= lr · db
```

## License

MIT
