package nn

import "math"

// Loss computes a scalar loss and its gradient w.r.t. predictions.
type Loss struct {
	Name    string
	Compute func(pred, target *Matrix) (loss float64, grad *Matrix)
}

// MSE – Mean Squared Error.
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

// BinaryCrossEntropy for binary classification (sigmoid output).
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

// SoftmaxCE – Softmax + Categorical Cross-Entropy combined.
//
// Expects raw logits as pred and one-hot vectors as target.
// The combined gradient simplifies to: (softmax(pred) - target) / batch.
var SoftmaxCE = Loss{
	Name: "softmax_cross_entropy",
	Compute: func(pred, target *Matrix) (float64, *Matrix) {
		batch := pred.Rows
		classes := pred.Cols
		const eps = 1e-12

		// ── Softmax (numerically stable) ──────────────────────
		sm := NewMatrix(batch, classes)
		for i := 0; i < batch; i++ {
			// subtract row max to avoid exp overflow
			maxV := pred.At(i, 0)
			for j := 1; j < classes; j++ {
				if v := pred.At(i, j); v > maxV {
					maxV = v
				}
			}
			sum := 0.0
			for j := 0; j < classes; j++ {
				e := math.Exp(pred.At(i, j) - maxV)
				sm.Set(i, j, e)
				sum += e
			}
			for j := 0; j < classes; j++ {
				sm.Set(i, j, sm.At(i, j)/sum)
			}
		}

		// ── Cross-Entropy loss ─────────────────────────────────
		loss := 0.0
		for i := 0; i < batch; i++ {
			for j := 0; j < classes; j++ {
				if t := target.At(i, j); t > 0 {
					loss -= t * math.Log(sm.At(i, j)+eps)
				}
			}
		}
		loss /= float64(batch)

		// ── Gradient: (softmax - target) / batch ──────────────
		grad := NewMatrix(batch, classes)
		for i := range grad.Data {
			grad.Data[i] = (sm.Data[i] - target.Data[i]) / float64(batch)
		}

		return loss, grad
	},
}
