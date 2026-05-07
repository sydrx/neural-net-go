package nn

import (
	"fmt"
	"math/rand"
)

type Trainer struct {
	Net     *Network
	Augment bool
}

func NewTrainer(net *Network, augment bool) *Trainer {
	return &Trainer{
		Net:     net,
		Augment: augment,
	}
}

func (t *Trainer) Run(x, y *Matrix, epochs, batchSize int, rng *rand.Rand, printEvery int) {
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
			bx := SliceRows(x, indices[start:end])
			by := SliceRows(y, indices[start:end])

			// Apply augmentation if enabled
			if t.Augment {
				bx = AugmentBatch(bx, rng)
			}

			totalLoss += t.Net.TrainStep(bx, by)
			batches++
		}

		if printEvery > 0 && epoch%printEvery == 0 {
			avgLoss := totalLoss / float64(batches)
			fmt.Printf("epoch %3d | loss: %.4f %s\n", epoch, avgLoss, t.trainBar(avgLoss, 2.5, 20))
		}
	}
}

func (t *Trainer) trainBar(loss, maxLoss float64, width int) string {
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
