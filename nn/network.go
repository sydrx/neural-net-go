package nn

import (
	"fmt"
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

// Train runs full-batch gradient descent.
func (n *Network) Train(x, y *Matrix, epochs, printEvery int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		loss := n.TrainStep(x, y)
		if printEvery > 0 && epoch%printEvery == 0 {
			fmt.Printf("epoch %5d | loss: %.6f %s\n", epoch, loss, lossBar(loss, 0.5, 20))
		}
	}
}

// TrainMiniBatch runs mini-batch SGD — essential for large datasets like MNIST.
//   x, y      – full dataset matrices  [N × features]
//   epochs    – number of full passes over the data
//   batchSize – samples per gradient update
//   rng       – random source for shuffling
func (n *Network) TrainMiniBatch(x, y *Matrix, epochs, batchSize int, rng *rand.Rand, printEvery int) {
	N := x.Rows
	indices := make([]int, N)
	for i := range indices {
		indices[i] = i
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		// Shuffle row indices each epoch
		rng.Shuffle(N, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		totalLoss := 0.0
		batches := 0

		for start := 0; start < N; start += batchSize {
			end := start + batchSize
			if end > N {
				end = N
			}
			bx := sliceRows(x, indices[start:end])
			by := sliceRows(y, indices[start:end])

			totalLoss += n.TrainStep(bx, by)
			batches++
		}

		if printEvery > 0 && epoch%printEvery == 0 {
			avgLoss := totalLoss / float64(batches)
			fmt.Printf("epoch %3d | loss: %.4f %s\n", epoch, avgLoss, lossBar(avgLoss, 2.5, 20))
		}
	}
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

// sliceRows extracts rows at given indices into a new Matrix.
func sliceRows(m *Matrix, indices []int) *Matrix {
	out := NewMatrix(len(indices), m.Cols)
	for i, idx := range indices {
		copy(out.Data[i*m.Cols:], m.Data[idx*m.Cols:(idx+1)*m.Cols])
	}
	return out
}

func lossBar(loss, maxLoss float64, width int) string {
	fill := int(float64(width) * (1 - loss/maxLoss))
	if fill < 0 {
		fill = 0
	}
	if fill > width {
		fill = width
	}
	bar := "["
	for i := 0; i < width; i++ {
		if i < fill {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	return bar + "]"
}
