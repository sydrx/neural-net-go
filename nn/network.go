package nn

import (
	"math/rand"
)

// Network is a sequential stack of Dense layers.
type Network struct {
	layers []*Dense
	loss   Loss
	lr     float64
}

// NewNetwork creates a network with the given learning rate and loss.
func NewNetwork(lr float64, loss Loss) *Network {
	return &Network{lr: lr, loss: loss}
}

// Add appends a layer.
func (n *Network) Add(layer *Dense) *Network {
	n.layers = append(n.layers, layer)
	return n
}

// RNG returns a seeded random source for weight initialisation.
func RNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

// Forward runs input through every layer.
func (n *Network) Forward(x *Matrix) *Matrix {
	out := x
	for _, l := range n.layers {
		out = l.Forward(out)
	}
	return out
}

func (n *Network) backward(grad *Matrix) {
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad, n.lr)
	}
}

// TrainStep performs one forward + backward pass and returns the loss.
func (n *Network) TrainStep(x, y *Matrix) float64 {
	pred := n.Forward(x)
	loss, grad := n.loss.Compute(pred, y)
	n.backward(grad)
	return loss
}

// Predict returns the network output for input x.
func (n *Network) Predict(x *Matrix) *Matrix { return n.Forward(x) }

// Accuracy computes classification accuracy.
// pred and target are [N × classes] matrices (one-hot or logits).
func Accuracy(pred, target *Matrix) float64 {
	correct := 0
	for i := 0; i < pred.Rows; i++ {
		if argmax(pred, i) == argmax(target, i) {
			correct++
		}
	}
	return float64(correct) / float64(pred.Rows)
}

func argmax(m *Matrix, row int) int {
	best, bestV := 0, m.At(row, 0)
	for j := 1; j < m.Cols; j++ {
		if v := m.At(row, j); v > bestV {
			best, bestV = j, v
		}
	}
	return best
}

// SliceRows extracts rows at given indices into a new Matrix.
func SliceRows(m *Matrix, indices []int) *Matrix {
	out := NewMatrix(len(indices), m.Cols)
	for i, idx := range indices {
		copy(out.Data[i*m.Cols:], m.Data[idx*m.Cols:(idx+1)*m.Cols])
	}
	return out
}
