package nn

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg" // register JPEG decoder
	_ "image/png"  // register PNG decoder
	"math"
	"os"
)

// ImageToMNIST loads an image file and converts it to a [1 × 784] matrix
// ready to feed into a network trained on MNIST.
//
// Steps:
//  1. Decode image (PNG or JPEG)
//  2. Convert to grayscale
//  3. Find bounding box of the digit and crop with padding
//  4. Resize to 28 × 28 using bilinear interpolation
//  5. Invert colours  (MNIST = white digit on black bg)
//  6. Normalise to [0, 1]
func ImageToMNIST(path string) (*Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, format, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("cannot decode image (%s): %w", format, err)
	}

	gray := toGray(img)
	gray = cropDigit(gray)
	small := resize28(gray)
	return grayToMatrix(small), nil
}

// ── Step 1: grayscale ─────────────────────────────────────────────────────────

func toGray(img image.Image) *image.Gray {
	b := img.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			g.SetGray(x, y, c)
		}
	}
	return g
}

// ── Step 2: crop to digit + 10% padding ───────────────────────────────────────

func cropDigit(g *image.Gray) *image.Gray {
	b := g.Bounds()
	threshold := uint8(200) // pixels darker than this belong to the digit

	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X, b.Min.Y

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if g.GrayAt(x, y).Y < threshold {
				if x < minX { minX = x }
				if x > maxX { maxX = x }
				if y < minY { minY = y }
				if y > maxY { maxY = y }
			}
		}
	}

	// nothing found — return original
	if minX > maxX || minY > maxY {
		return g
	}

	// add 10% padding
	w, h := maxX-minX, maxY-minY
	pad := int(math.Max(float64(w), float64(h))) / 10
	minX -= pad; minY -= pad; maxX += pad; maxY += pad
	// clamp to image bounds
	if minX < b.Min.X { minX = b.Min.X }
	if minY < b.Min.Y { minY = b.Min.Y }
	if maxX > b.Max.X { maxX = b.Max.X }
	if maxY > b.Max.Y { maxY = b.Max.Y }

	// make square
	cw, ch := maxX-minX, maxY-minY
	if cw > ch {
		diff := cw - ch
		minY -= diff / 2
		maxY += diff - diff/2
	} else {
		diff := ch - cw
		minX -= diff / 2
		maxX += diff - diff/2
	}
	// clamp again
	if minX < b.Min.X { minX = b.Min.X }
	if minY < b.Min.Y { minY = b.Min.Y }
	if maxX > b.Max.X { maxX = b.Max.X }
	if maxY > b.Max.Y { maxY = b.Max.Y }

	cropped := image.NewGray(image.Rect(0, 0, maxX-minX, maxY-minY))
	for y := minY; y < maxY; y++ {
		for x := minX; x < maxX; x++ {
			cropped.SetGray(x-minX, y-minY, g.GrayAt(x, y))
		}
	}
	return cropped
}

// ── Step 3: bilinear resize to 28×28 ──────────────────────────────────────────

func resize28(src *image.Gray) *image.Gray {
	const size = 28
	dst := image.NewGray(image.Rect(0, 0, size, size))
	sb := src.Bounds()
	sw, sh := float64(sb.Max.X-sb.Min.X), float64(sb.Max.Y-sb.Min.Y)

	for dy := 0; dy < size; dy++ {
		for dx := 0; dx < size; dx++ {
			// map destination pixel to source space
			sx := (float64(dx)+0.5)/size*sw - 0.5
			sy := (float64(dy)+0.5)/size*sh - 0.5

			x0, y0 := int(math.Floor(sx)), int(math.Floor(sy))
			x1, y1 := x0+1, y0+1

			// clamp
			clampInt := func(v, lo, hi int) int {
				if v < lo { return lo }
				if v > hi { return hi }
				return v
			}
			x0 = clampInt(x0, 0, sb.Max.X-1)
			x1 = clampInt(x1, 0, sb.Max.X-1)
			y0 = clampInt(y0, 0, sb.Max.Y-1)
			y1 = clampInt(y1, 0, sb.Max.Y-1)

			// bilinear weights
			fx, fy := sx-math.Floor(sx), sy-math.Floor(sy)
			v00 := float64(src.GrayAt(x0, y0).Y)
			v10 := float64(src.GrayAt(x1, y0).Y)
			v01 := float64(src.GrayAt(x0, y1).Y)
			v11 := float64(src.GrayAt(x1, y1).Y)

			v := v00*(1-fx)*(1-fy) + v10*fx*(1-fy) +
				v01*(1-fx)*fy + v11*fx*fy

			dst.SetGray(dx, dy, color.Gray{Y: uint8(v)})
		}
	}
	return dst
}

// ── Step 4: to matrix + invert + normalise ────────────────────────────────────

func grayToMatrix(g *image.Gray) *Matrix {
	b := g.Bounds()
	w, h := b.Max.X-b.Min.X, b.Max.Y-b.Min.Y
	m := NewMatrix(1, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			px := float64(g.GrayAt(x, y).Y)
			// Invert: MNIST has white digit on black background.
			// Photos typically have dark digit on white background.
			inverted := (255 - px) / 255.0
			m.Data[y*w+x] = inverted
		}
	}
	return m
}

// PrintASCII renders a [1×784] MNIST matrix as ASCII art in the terminal.
func PrintASCII(m *Matrix) {
	const chars = " ░▒▓█"
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			v := m.Data[y*28+x]
			idx := int(v * float64(len(chars)-1))
			fmt.Printf("%c%c", chars[idx], chars[idx])
		}
		fmt.Println()
	}
}
