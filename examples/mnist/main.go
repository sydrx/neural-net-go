package main

import (
	"fmt"
	"math/rand"
	"time"

	nn "neural-net-go/nn"
)

const modelPath = "model.bin"

func main() {
	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║  Neural Network — MNIST Digit Recognizer ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	// ── Load data ─────────────────────────────────────────────
	fmt.Println("Loading MNIST dataset...")
	mnist, err := nn.LoadMNIST("data/mnist")
	if err != nil {
		panic(err)
	}
	fmt.Printf("  train: %d samples\n", mnist.TrainX.Rows)
	fmt.Printf("  test:  %d samples\n\n", mnist.TestX.Rows)

	// ── Build network: 784 → 256 → 128 → 10 ──────────────────
	rng := nn.RNG(42)
	net := nn.NewNetwork(0.01, nn.SoftmaxCE).
		Add(nn.NewDense(784, 256, nn.ReLU, rng)).
		Add(nn.NewDense(256, 128, nn.ReLU, rng)).
		Add(nn.NewDense(128, 10, nn.Linear, rng))

	// ── Train ─────────────────────────────────────────────────
	const (
		epochs    = 20
		batchSize = 64
	)
	fmt.Printf("Training: %d epochs, batch size %d\n\n", epochs, batchSize)
	trainRNG := rand.New(rand.NewSource(time.Now().UnixNano()))

	start := time.Now()
	net.TrainMiniBatch(mnist.TrainX, mnist.TrainY, epochs, batchSize, trainRNG, 1)
	elapsed := time.Since(start)

	// ── Evaluate ──────────────────────────────────────────────
	trainPred := net.Predict(mnist.TrainX)
	testPred := net.Predict(mnist.TestX)

	fmt.Printf("\nTraining time : %s\n", elapsed.Round(time.Second))
	fmt.Printf("Train accuracy: %.2f%%\n", nn.Accuracy(trainPred, mnist.TrainY)*100)
	fmt.Printf("Test  accuracy: %.2f%%\n", nn.Accuracy(testPred, mnist.TestY)*100)

	// ── Save model ────────────────────────────────────────────
	fmt.Printf("\nSaving model to %s...\n", modelPath)
	if err := net.Save(modelPath); err != nil {
		fmt.Printf("Warning: could not save model: %v\n", err)
	} else {
		fmt.Println("Model saved ✓")
		fmt.Println("\nNow you can recognize your own handwritten digits:")
		fmt.Println("  go run ./examples/predict your_photo.jpg")
	}
}
