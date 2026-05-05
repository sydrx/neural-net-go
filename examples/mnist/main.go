package main

import (
	"fmt"
	"math/rand"
	"time"

	nn "neural-net-go/nn"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║   Neural Network — MNIST Digit Recognizer ║")
	fmt.Println("╚══════════════════════════════════════════╝")
	fmt.Println()

	// ── Load data ────────────────────────────────────────────
	fmt.Println("Loading MNIST dataset...")
	mnist, err := nn.LoadMNIST("data/mnist")
	if err != nil {
		panic(err)
	}
	fmt.Printf("  train: %d samples\n", mnist.TrainX.Rows)
	fmt.Printf("  test:  %d samples\n\n", mnist.TestX.Rows)

	// ── Build network: 784 → 256 → 128 → 10 ─────────────────
	rng := nn.RNG(42)
	net := nn.NewNetwork(0.01, nn.SoftmaxCE).
		Add(nn.NewDense(784, 256, nn.ReLU, rng)).
		Add(nn.NewDense(256, 128, nn.ReLU, rng)).
		Add(nn.NewDense(128, 10, nn.Linear, rng)) // softmax is inside SoftmaxCE loss

	// ── Train ────────────────────────────────────────────────
	const (
		epochs    = 20
		batchSize = 64
	)

	fmt.Printf("Training: %d epochs, batch size %d\n\n", epochs, batchSize)
	trainRNG := rand.New(rand.NewSource(time.Now().UnixNano()))

	start := time.Now()
	net.TrainMiniBatch(mnist.TrainX, mnist.TrainY, epochs, batchSize, trainRNG, 1)
	elapsed := time.Since(start)

	// ── Evaluate ─────────────────────────────────────────────
	fmt.Println()

	trainPred := net.Predict(mnist.TrainX)
	trainAcc := nn.Accuracy(trainPred, mnist.TrainY)

	testPred := net.Predict(mnist.TestX)
	testAcc := nn.Accuracy(testPred, mnist.TestY)

	fmt.Printf("Training time : %s\n", elapsed.Round(time.Second))
	fmt.Printf("Train accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Test  accuracy: %.2f%%\n", testAcc*100)

	// ── Confusion matrix (test set) ───────────────────────────
	fmt.Println()
	fmt.Println("Confusion matrix (rows=actual, cols=predicted):")
	printConfusion(testPred, mnist.TestY)

	// ── Show a few predictions ────────────────────────────────
	fmt.Println("\nSample predictions:")
	for i := 0; i < 10; i++ {
		row := nn.SliceRow(mnist.TestX, i)
		pred := net.Predict(row)
		actual := argmax(mnist.TestY, i)
		predicted := argmaxVec(pred.Data)
		status := "✓"
		if actual != predicted {
			status = "✗"
		}
		fmt.Printf("  sample %2d: actual=%d  predicted=%d  %s\n",
			i, actual, predicted, status)
	}
}

func printConfusion(pred, target *nn.Matrix) {
	cm := [10][10]int{}
	for i := 0; i < pred.Rows; i++ {
		actual := argmax(target, i)
		predicted := argmaxVec(pred.Data[i*pred.Cols : (i+1)*pred.Cols])
		cm[actual][predicted]++
	}
	fmt.Print("      ")
	for j := 0; j < 10; j++ {
		fmt.Printf(" %4d", j)
	}
	fmt.Println()
	fmt.Print("      ")
	for j := 0; j < 10; j++ {
		fmt.Print(" ────")
	}
	fmt.Println()
	for i := 0; i < 10; i++ {
		fmt.Printf("  %d │ ", i)
		for j := 0; j < 10; j++ {
			if i == j {
				fmt.Printf(" \033[32m%4d\033[0m", cm[i][j]) // green on diagonal
			} else if cm[i][j] > 0 {
				fmt.Printf(" \033[31m%4d\033[0m", cm[i][j]) // red off-diagonal
			} else {
				fmt.Printf("    .")
			}
		}
		fmt.Println()
	}
}

func argmax(m *nn.Matrix, row int) int {
	best, bestV := 0, m.At(row, 0)
	for j := 1; j < m.Cols; j++ {
		if v := m.At(row, j); v > bestV {
			best, bestV = j, v
		}
	}
	return best
}

func argmaxVec(v []float64) int {
	best, bestV := 0, v[0]
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}
