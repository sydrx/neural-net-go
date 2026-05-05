package main

import (
	"fmt"

	nn "neural-net-go/nn"
)

func main() {
	fmt.Println("╔══════════════════════════════════════╗")
	fmt.Println("║   Neural Network from Scratch in Go  ║")
	fmt.Println("║          XOR Problem Demo             ║")
	fmt.Println("╚══════════════════════════════════════╝")
	fmt.Println()

	// XOR truth table
	X := &nn.Matrix{
		Rows: 4, Cols: 2,
		Data: []float64{0, 0, 0, 1, 1, 0, 1, 1},
	}
	Y := &nn.Matrix{
		Rows: 4, Cols: 1,
		Data: []float64{0, 1, 1, 0},
	}

	// Architecture: 2 → 8 → 4 → 1
	rng := nn.RNG(42)
	net := nn.NewNetwork(0.5, nn.MSE).
		Add(nn.NewDense(2, 8, nn.Tanh, rng)).
		Add(nn.NewDense(8, 4, nn.Tanh, rng)).
		Add(nn.NewDense(4, 1, nn.Sigmoid, rng))

	fmt.Println("Training (10 000 epochs)...")
	fmt.Println()

	net.Train(X, Y, 10_000, 1_000)

	fmt.Println()
	fmt.Println("Results:")
	fmt.Println("┌──────────┬──────────┬──────────┬──────────┐")
	fmt.Println("│  Input   │ Expected │   Got    │  Status  │")
	fmt.Println("├──────────┼──────────┼──────────┼──────────┤")

	pred := net.Predict(X)
	inputs := [][2]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	expected := []float64{0, 1, 1, 0}
	correct := 0
	for i, inp := range inputs {
		p := pred.Data[i]
		rounded := 0.0
		if p >= 0.5 {
			rounded = 1.0
		}
		status := "✗"
		if rounded == expected[i] {
			status = "✓"
			correct++
		}
		fmt.Printf("│ [%v, %v]   │    %.0f     │  %.4f  │    %s     │\n",
			int(inp[0]), int(inp[1]), expected[i], p, status)
	}
	fmt.Println("└──────────┴──────────┴──────────┴──────────┘")
	fmt.Printf("\nAccuracy: %d/4 (%.0f%%)\n", correct, float64(correct)/4*100)
}
