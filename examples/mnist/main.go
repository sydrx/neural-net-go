package main

import (
	"fmt"
	"math/rand"
	"time"

	nn "github.com/sydrx/neural-net-go/nn"
)

const modelPath = "model.bin"

func main() {
	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║   Neural Network — MNIST Digit Recognizer ║")
	fmt.Println("╚══════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Loading MNIST dataset...")
	mnist, err := nn.LoadMNIST("data/mnist")
	if err != nil {
		panic(err)
	}
	fmt.Printf("  train: %d samples\n", mnist.TrainX.Rows)
	fmt.Printf("  test:  %d samples\n\n", mnist.TestX.Rows)

	// Network: 784 → 256 → 128 → 10
	rng := nn.RNG(42)
	net := nn.NewNetwork(0.01, nn.SoftmaxCE).
		Add(nn.NewDense(784, 256, nn.ReLU, rng)).
		Add(nn.NewDense(256, 128, nn.ReLU, rng)).
		Add(nn.NewDense(128, 10, nn.Linear, rng))

	trainer := nn.NewTrainer(net, true) // true = use augmentation
	fmt.Println()
	fmt.Println("Training: 25 epochs, batch 64, augmentation ON")
	start := time.Now()

	shuffleRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	trainer.Run(mnist.TrainX, mnist.TrainY, 25, 64, shuffleRng, 1)

	elapsed := time.Since(start)

	trainAcc := nn.Accuracy(net.Predict(mnist.TrainX), mnist.TrainY)
	testAcc := nn.Accuracy(net.Predict(mnist.TestX), mnist.TestY)

	fmt.Printf("\nTime          : %s\n", elapsed.Round(time.Second))
	fmt.Printf("Train accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Test  accuracy: %.2f%%\n", testAcc*100)

	fmt.Printf("\nSaving to %s... ", modelPath)
	if err := net.Save(modelPath); err != nil {
		fmt.Println("failed:", err)
	} else {
		fmt.Println("✓")
		fmt.Println("\nNow try your own digits:")
		fmt.Println("  go run ./examples/predict photo.jpg")
	}
}
