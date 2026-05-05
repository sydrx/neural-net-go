package nn

import "math/rand"

// Dense is a fully-connected layer: out = activation(X·W + b)
type Dense struct {
	W    *Matrix     // weights  [in × out]
	B    []float64   // biases   [out]
	Act  Activation  // activation function

	// cache filled during Forward, used in Backward
	input *Matrix // pre-activation input  X        [batch × in]
	z     *Matrix // pre-activation output X·W + b  [batch × out]
}

// NewDense creates a Dense layer with Xavier-initialised weights.
func NewDense(in, out int, act Activation, rng *rand.Rand) *Dense {
	w := NewMatrix(in, out)
	w.Randomize(rng)
	return &Dense{
		W:   w,
		B:   make([]float64, out),
		Act: act,
	}
}

// Forward computes the layer output and caches intermediate values.
//   output shape: [batch × out]
func (d *Dense) Forward(x *Matrix) *Matrix {
	d.input = x
	z := Dot(x, d.W)   // X · W
	AddBias(z, d.B)     // + b
	d.z = z
	return Apply(z, d.Act.Forward) // activation(z)
}

// Backward performs backpropagation through this layer.
//   dOut – gradient from the next layer  [batch × out]
//   lr   – learning rate
//   returns gradient w.r.t. this layer's input [batch × in]
func (d *Dense) Backward(dOut *Matrix, lr float64) *Matrix {
	// δ = dOut ⊙ act'(z)   [batch × out]
	delta := Apply(d.z, d.Act.Deriv)
	HadamardMul(delta, dOut)

	// ∂L/∂W = X^T · δ      [in × out]
	dW := Dot(Transpose(d.input), delta)

	// ∂L/∂b = Σ_rows δ     [out]
	dB := SumRows(delta)

	// ∂L/∂X = δ · W^T      [batch × in]
	dInput := Dot(delta, Transpose(d.W))

	// SGD update
	for i := range d.W.Data {
		d.W.Data[i] -= lr * dW.Data[i]
	}
	for j := range d.B {
		d.B[j] -= lr * dB[j]
	}

	return dInput
}
