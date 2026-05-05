package nn

import "math"

// Activation defines a differentiable activation function.
type Activation struct {
	Name    string
	Forward func(float64) float64
	Deriv   func(float64) float64 // derivative w.r.t. the pre-activation input z
}

// Sigmoid: σ(x) = 1 / (1 + e^-x)
var Sigmoid = Activation{
	Name:    "sigmoid",
	Forward: func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
	Deriv:   func(x float64) float64 { s := 1 / (1 + math.Exp(-x)); return s * (1 - s) },
}

// ReLU: max(0, x)
var ReLU = Activation{
	Name:    "relu",
	Forward: func(x float64) float64 { return math.Max(0, x) },
	Deriv:   func(x float64) float64 { if x > 0 { return 1 }; return 0 },
}

// Tanh
var Tanh = Activation{
	Name:    "tanh",
	Forward: math.Tanh,
	Deriv:   func(x float64) float64 { t := math.Tanh(x); return 1 - t*t },
}

// Linear (identity) – useful for the output layer of regression networks.
var Linear = Activation{
	Name:    "linear",
	Forward: func(x float64) float64 { return x },
	Deriv:   func(_ float64) float64 { return 1 },
}
