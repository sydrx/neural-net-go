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

// NewNetwork builds a network with the given learning rate and loss.
// Use Add() to stack layers.
func NewNetwork(lr float64, loss Loss) *Network {
	return &Network{lr: lr, loss: loss}
}

// Add appends a layer to the network.
func (n *Network) Add(layer *Dense) *Network {
	n.layers = append(n.layers, layer)
	return n
}

// RNG returns a deterministic random source for weight init.
func RNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

// Forward runs the input through every layer.
func (n *Network) Forward(x *Matrix) *Matrix {
	out := x
	for _, l := range n.layers {
		out = l.Forward(out)
	}
	return out
}

// backward propagates gradients from the loss through all layers.
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

// Train runs full-batch gradient descent for the given number of epochs.
// printEvery controls how often progress is printed (0 = never).
func (n *Network) Train(x, y *Matrix, epochs, printEvery int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		loss := n.TrainStep(x, y)
		if printEvery > 0 && epoch%printEvery == 0 {
			bar := progressBar(loss, 0.5, 20)
			fmt.Printf("epoch %5d | loss: %.6f %s\n", epoch, loss, bar)
		}
	}
}

// Predict runs forward pass and returns predictions.
func (n *Network) Predict(x *Matrix) *Matrix { return n.Forward(x) }

func progressBar(loss, maxLoss float64, width int) string {
	fill := int(float64(width) * (1 - loss/maxLoss))
	if fill < 0 { fill = 0 }
	if fill > width { fill = width }
	bar := "["
	for i := 0; i < width; i++ {
		if i < fill {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	bar += "]"
	return bar
}
