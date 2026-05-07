package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	nn "github.com/sydrx/neural-net-go/nn"
)

const modelPath = "model.bin"

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage:")
		fmt.Println("  go run ./examples/predict photo.jpg")
		fmt.Println()
		fmt.Println("Train the model first:")
		fmt.Println("  go run ./examples/mnist")
		os.Exit(1)
	}

	imagePath := os.Args[1]

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
	fmt.Println("Model loaded ✓")
	fmt.Println()

	// ── Segment digits ────────────────────────────────────────
	fmt.Printf("Segmenting digits in: %s\n", imagePath)
	digits, rects, err := nn.SegmentDigits(imagePath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Found %d digit(s)\n\n", len(digits))

	// ── Predict each digit ────────────────────────────────────
	results := make([]int, len(digits))
	confidences := make([]float64, len(digits))

	for i, input := range digits {
		_ = rects[i] // bounding box (available if you want to draw it)

		output := net.Predict(input)
		probs := softmax(output.Data)
		best := argmax(probs)
		results[i] = best
		confidences[i] = probs[best]

		// Show ASCII preview of each digit
		fmt.Printf("── Digit %d ──────────────────────────────────────────────\n", i+1)
		fmt.Println("┌" + rep("─", 56) + "┐")
		nn.PrintASCII(input)
		fmt.Println("└" + rep("─", 56) + "┘")

		// Top-3 predictions
		type dp struct {
			d int
			p float64
		}
		ranked := make([]dp, 10)
		for j, p := range probs {
			ranked[j] = dp{j, p}
		}
		sort.Slice(ranked, func(a, b int) bool { return ranked[a].p > ranked[b].p })
		fmt.Printf("  Prediction: %d  (%.1f%%)\n", best, probs[best]*100)
		fmt.Printf("  Runners-up: %d (%.1f%%)   %d (%.1f%%)\n\n",
			ranked[1].d, ranked[1].p*100,
			ranked[2].d, ranked[2].p*100)
	}

	// ── Final answer ──────────────────────────────────────────
	number := ""
	for _, d := range results {
		number += fmt.Sprintf("%d", d)
	}

	fmt.Println(strings.Repeat("═", 44))
	fmt.Printf("  Recognized number: %s\n", number)

	// confidence bar per digit
	fmt.Println()
	for i, d := range results {
		bar := int(confidences[i] * 20)
		fmt.Printf("  digit %d: %d  %s  %.1f%%\n",
			i+1, d, "["+rep("█", bar)+rep("░", 20-bar)+"]", confidences[i]*100)
	}
	fmt.Println(strings.Repeat("═", 44))
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
