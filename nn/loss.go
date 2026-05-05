package nn

import "math"

// Loss computes a scalar loss and its gradient w.r.t. predictions.
type Loss struct {
	Name    string
	Compute func(pred, target *Matrix) (loss float64, grad *Matrix)
}

// MSE – Mean Squared Error: L = mean((pred - target)²)
var MSE = Loss{
	Name: "mse",
	Compute: func(pred, target *Matrix) (float64, *Matrix) {
		n := float64(len(pred.Data))
		grad := NewMatrix(pred.Rows, pred.Cols)
		loss := 0.0
		for i, p := range pred.Data {
			diff := p - target.Data[i]
			loss += diff * diff
			grad.Data[i] = 2 * diff / n
		}
		return loss / n, grad
	},
}

// BinaryCrossEntropy: L = -mean(y·log(p) + (1-y)·log(1-p))
var BinaryCrossEntropy = Loss{
	Name: "binary_cross_entropy",
	Compute: func(pred, target *Matrix) (float64, *Matrix) {
		const eps = 1e-12
		n := float64(len(pred.Data))
		grad := NewMatrix(pred.Rows, pred.Cols)
		loss := 0.0
		for i, p := range pred.Data {
			p = math.Max(eps, math.Min(1-eps, p))
			y := target.Data[i]
			loss += -(y*math.Log(p) + (1-y)*math.Log(1-p))
			grad.Data[i] = (p - y) / (p * (1 - p) * n)
		}
		return loss / n, grad
	},
}
