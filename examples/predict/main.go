package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	nn "neural-net-go/nn"
)

const modelPath = "model.bin"

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run ./examples/predict <image.jpg|image.png>")
		fmt.Println()
		fmt.Println("Train the model first:")
		fmt.Println("  go run ./examples/mnist")
		os.Exit(1)
	}

	// ── Load model ────────────────────────────────────────────
	fmt.Printf("Loading model from %s...\n", modelPath)
	rng := nn.RNG(42)
	net := nn.NewNetwork(0.01, nn.SoftmaxCE).
		Add(nn.NewDense(784, 256, nn.ReLU, rng)).
		Add(nn.NewDense(256, 128, nn.ReLU, rng)).
		Add(nn.NewDense(128, 10, nn.Linear, rng))

	if err := net.LoadWeights(modelPath); err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("Train first: go run ./examples/mnist")
		os.Exit(1)
	}
	fmt.Println("Model loaded ✓\n")

	// ── Preprocess image ──────────────────────────────────────
	imagePath := os.Args[1]
	fmt.Printf("Processing: %s\n", imagePath)
	input, err := nn.ImageToMNIST(imagePath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nWhat the network sees (28×28):")
	fmt.Println("┌" + rep("─", 56) + "┐")
	nn.PrintASCII(input)
	fmt.Println("└" + rep("─", 56) + "┘")

	// ── Predict ───────────────────────────────────────────────
	output := net.Predict(input)
	probs := softmax(output.Data)
	best := argmax(probs)

	fmt.Printf("\n┌─────────────────────────────┐\n")
	fmt.Printf("│  Predicted digit:   %d       │\n", best)
	fmt.Printf("│  Confidence:    %5.1f%%      │\n", probs[best]*100)
	fmt.Printf("└─────────────────────────────┘\n")

	fmt.Println("\nAll probabilities:")
	type dp struct {
		d int
		p float64
	}
	ranked := make([]dp, 10)
	for i, p := range probs {
		ranked[i] = dp{i, p}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].p > ranked[j].p })
	for _, x := range ranked {
		bar := int(x.p * 30)
		marker := " "
		if x.d == best {
			marker = "►"
		}
		fmt.Printf("  %s %d │%-30s│ %5.1f%%\n", marker, x.d, rep("█", bar), x.p*100)
	}
}

func softmax(logits []float64) []float64 {
	maxV := logits[0]
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxV)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func argmax(v []float64) int {
	best, bestV := 0, v[0]
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}

func rep(s string, n int) string {
	r := ""
	for i := 0; i < n; i++ {
		r += s
	}
	return r
}
