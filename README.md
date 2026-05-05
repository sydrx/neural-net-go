# 🧠 neural-net-go

Neural network built **from scratch** in pure Go — no ML libraries, no external dependencies.

Feedforward network with backpropagation and mini-batch SGD. Works on XOR and MNIST (~97-98% accuracy).

## Quick start

```bash
git clone https://github.com/yourname/neural-net-go
cd neural-net-go

# XOR demo (instant)
go run ./examples/xor

# MNIST digit recognizer (~97% accuracy, ~2 min on CPU)
go run ./examples/mnist
```

## What's inside

```
nn/
├── matrix.go      # Matrix ops: Dot, Transpose, Apply, Hadamard…
├── activation.go  # Sigmoid, ReLU, Tanh, Linear
├── layer.go       # Dense layer — forward + backprop
├── loss.go        # MSE, BinaryCrossEntropy, SoftmaxCE
├── network.go     # Sequential net, Train, TrainMiniBatch, Accuracy
└── mnist.go       # MNIST downloader + parser

examples/
├── xor/main.go    # Classic XOR problem  → 100% accuracy
└── mnist/main.go  # Handwritten digits   → ~97-98% accuracy
```

## MNIST results

```
Training: 20 epochs, batch 64, lr 0.01

epoch  1 | loss: 0.4821 [████░░░░░░░░░░░░░░░░]
epoch  5 | loss: 0.2103 [█████████░░░░░░░░░░░]
epoch 10 | loss: 0.1247 [█████████████░░░░░░░]
epoch 20 | loss: 0.0731 [████████████████████]

Train accuracy: 98.31%
Test  accuracy: 97.42%
```

## How it works

**Forward pass** for each Dense layer:
```
z = X · W + b
a = activation(z)
```

**Backward pass** (backpropagation):
```
δ = dLoss/dOut ⊙ activation'(z)
dW = Xᵀ · δ
db = Σ δ
dX = δ · Wᵀ
```

**SoftmaxCrossEntropy** gradient shortcut (avoids Jacobian):
```
dLoss/dz = softmax(z) - y_one_hot   (per sample, / batch_size)
```

**Mini-batch SGD**: data is shuffled each epoch, split into batches of 64.

## Architecture

| Example | Layers           | Activation | Loss        | Accuracy |
|---------|------------------|------------|-------------|----------|
| XOR     | 2→8→4→1          | Tanh/Sigmoid | MSE       | 100%     |
| MNIST   | 784→256→128→10   | ReLU/Linear  | SoftmaxCE | ~97-98%  |

## License

MIT
