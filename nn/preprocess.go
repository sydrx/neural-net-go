package nn

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
)

// ImageToMNIST loads an image and converts it to a single [1×784] matrix.
// Auto-detects stroke thickness and thins if needed.
func ImageToMNIST(path string) (*Matrix, error) {
	gray, err := loadGray(path)
	if err != nil {
		return nil, err
	}
	gray = autoThin(gray)
	gray = cropDigit(gray)
	small := resize28(gray)
	return grayToMatrix(small), nil
}

// SegmentDigits loads an image, finds individual digit regions by vertical
// projection, crops+resizes each to 28×28 and returns one matrix per digit.
func SegmentDigits(path string) ([]*Matrix, []image.Rectangle, error) {
	gray, err := loadGray(path)
	if err != nil {
		return nil, nil, err
	}

	// Auto-thin before segmentation so projection works on clean edges
	thinned := autoThin(gray)

	const threshold = uint8(128)
	bw := binarize(thinned, threshold)

	b := bw.Bounds()
	W, H := b.Max.X, b.Max.Y

	// Vertical projection
	proj := make([]int, W)
	for x := 0; x < W; x++ {
		for y := 0; y < H; y++ {
			if bw.GrayAt(x, y).Y == 0 {
				proj[x]++
			}
		}
	}

	// Find column spans (digit regions)
	type span struct{ lo, hi int }
	var spans []span
	inDigit := false
	var lo int
	lastInk := -4 - 1
	const gap = 4
	for x := 0; x < W; x++ {
		if proj[x] > 0 {
			if !inDigit {
				inDigit = true
				lo = x
			}
			lastInk = x
		} else if inDigit && x-lastInk > gap {
			spans = append(spans, span{lo, lastInk})
			inDigit = false
		}
	}
	if inDigit {
		spans = append(spans, span{lo, lastInk})
	}

	if len(spans) == 0 {
		m, err := ImageToMNIST(path)
		r := image.Rect(0, 0, W, H)
		return []*Matrix{m}, []image.Rectangle{r}, err
	}

	var matrices []*Matrix
	var rects []image.Rectangle

	for _, sp := range spans {
		minY, maxY := H, 0
		for x := sp.lo; x <= sp.hi; x++ {
			for y := 0; y < H; y++ {
				if bw.GrayAt(x, y).Y == 0 {
					if y < minY { minY = y }
					if y > maxY { maxY = y }
				}
			}
		}
		if minY > maxY {
			continue
		}

		padX := int(math.Max(1, float64(sp.hi-sp.lo)*0.15))
		padY := int(math.Max(1, float64(maxY-minY)*0.15))
		x0, x1 := sp.lo-padX, sp.hi+padX
		y0, y1 := minY-padY, maxY+padY
		if x0 < 0 { x0 = 0 }
		if y0 < 0 { y0 = 0 }
		if x1 >= W { x1 = W - 1 }
		if y1 >= H { y1 = H - 1 }

		// Make square
		cw, ch := x1-x0, y1-y0
		if cw > ch {
			diff := cw - ch
			y0 -= diff / 2
			y1 += diff - diff/2
		} else {
			diff := ch - cw
			x0 -= diff / 2
			x1 += diff - diff/2
		}
		if x0 < 0 { x0 = 0 }
		if y0 < 0 { y0 = 0 }
		if x1 >= W { x1 = W - 1 }
		if y1 >= H { y1 = H - 1 }

		// Crop from the THINNED image (not original) so network sees thin strokes
		cropped := image.NewGray(image.Rect(0, 0, x1-x0+1, y1-y0+1))
		for y := y0; y <= y1; y++ {
			for x := x0; x <= x1; x++ {
				cropped.SetGray(x-x0, y-y0, thinned.GrayAt(x, y))
			}
		}

		small := resize28(cropped)
		matrices = append(matrices, grayToMatrix(small))
		rects = append(rects, image.Rect(x0, y0, x1, y1))
	}

	return matrices, rects, nil
}

// ── auto-thinning ─────────────────────────────────────────────────────────────

// autoThin measures average stroke width and erodes if strokes are thick.
// MNIST strokes are ~2px wide; anything thicker gets eroded down.
func autoThin(g *image.Gray) *image.Gray {
	radius := measureStrokeRadius(g)
	if radius <= 1 {
		return g // already thin, skip
	}
	fmt.Printf("  [preprocess] thick strokes detected (r≈%d), thinning...\n", radius)
	return thinStrokes(g, 200, radius)
}

// measureStrokeRadius estimates half the average stroke width via horizontal runs.
func measureStrokeRadius(g *image.Gray) int {
	b := g.Bounds()
	const threshold = uint8(200)
	totalLen, count := 0, 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		inRun := false
		runLen := 0
		for x := b.Min.X; x < b.Max.X; x++ {
			if g.GrayAt(x, y).Y < threshold {
				inRun = true
				runLen++
			} else if inRun {
				totalLen += runLen
				count++
				inRun = false
				runLen = 0
			}
		}
	}
	if count == 0 {
		return 0
	}
	avg := totalLen / count
	// MNIST strokes are ~2px wide. If avg > 6px → thick stroke image
	if avg <= 6 {
		return 0
	}
	// erosion radius ≈ (avg - 2) / 2, capped at 6
	r := (avg - 2) / 2
	if r > 6 { r = 6 }
	return r
}

// ── morphological ops ─────────────────────────────────────────────────────────

// erode shrinks ink pixels inward by radius.
// ink=0, background=255 (after binarize).
func erode(g *image.Gray, radius int) *image.Gray {
	b := g.Bounds()
	W, H := b.Max.X, b.Max.Y
	out := image.NewGray(b)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			minVal := uint8(255)
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					nx, ny := x+dx, y+dy
					if nx < 0 || ny < 0 || nx >= W || ny >= H {
						continue
					}
					if v := g.GrayAt(nx, ny).Y; v < minVal {
						minVal = v
					}
				}
			}
			out.SetGray(x, y, color.Gray{Y: minVal})
		}
	}
	return out
}

// thinStrokes binarizes then erodes to reduce thick strokes.
func thinStrokes(g *image.Gray, threshold uint8, erosionRadius int) *image.Gray {
	b := g.Bounds()
	bin := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if g.GrayAt(x, y).Y < threshold {
				bin.SetGray(x, y, color.Gray{Y: 0})
			} else {
				bin.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	return erode(bin, erosionRadius)
}

// ── internal helpers ──────────────────────────────────────────────────────────

func loadGray(path string) (*image.Gray, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("cannot decode image (%s): %w", format, err)
	}
	return toGray(img), nil
}

func binarize(g *image.Gray, threshold uint8) *image.Gray {
	b := g.Bounds()
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if g.GrayAt(x, y).Y < threshold {
				out.SetGray(x, y, color.Gray{Y: 0})
			} else {
				out.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	return out
}

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

func cropDigit(g *image.Gray) *image.Gray {
	b := g.Bounds()
	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X, b.Min.Y
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if g.GrayAt(x, y).Y < 200 {
				if x < minX { minX = x }
				if x > maxX { maxX = x }
				if y < minY { minY = y }
				if y > maxY { maxY = y }
			}
		}
	}
	if minX > maxX || minY > maxY {
		return g
	}
	w, h := maxX-minX, maxY-minY
	pad := int(math.Max(float64(w), float64(h))) / 10
	minX -= pad; minY -= pad; maxX += pad; maxY += pad
	if minX < b.Min.X { minX = b.Min.X }
	if minY < b.Min.Y { minY = b.Min.Y }
	if maxX > b.Max.X { maxX = b.Max.X }
	if maxY > b.Max.Y { maxY = b.Max.Y }
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

func resize28(src *image.Gray) *image.Gray {
	const size = 28
	dst := image.NewGray(image.Rect(0, 0, size, size))
	sb := src.Bounds()
	sw, sh := float64(sb.Max.X-sb.Min.X), float64(sb.Max.Y-sb.Min.Y)
	for dy := 0; dy < size; dy++ {
		for dx := 0; dx < size; dx++ {
			sx := (float64(dx)+0.5)/size*sw - 0.5
			sy := (float64(dy)+0.5)/size*sh - 0.5
			x0, y0 := int(math.Floor(sx)), int(math.Floor(sy))
			x1, y1 := x0+1, y0+1
			ci := func(v, lo, hi int) int {
				if v < lo { return lo }
				if v > hi { return hi }
				return v
			}
			x0 = ci(x0, 0, sb.Max.X-1); x1 = ci(x1, 0, sb.Max.X-1)
			y0 = ci(y0, 0, sb.Max.Y-1); y1 = ci(y1, 0, sb.Max.Y-1)
			fx, fy := sx-math.Floor(sx), sy-math.Floor(sy)
			v := float64(src.GrayAt(x0, y0).Y)*(1-fx)*(1-fy) +
				float64(src.GrayAt(x1, y0).Y)*fx*(1-fy) +
				float64(src.GrayAt(x0, y1).Y)*(1-fx)*fy +
				float64(src.GrayAt(x1, y1).Y)*fx*fy
			dst.SetGray(dx, dy, color.Gray{Y: uint8(v)})
		}
	}
	return dst
}

func grayToMatrix(g *image.Gray) *Matrix {
	b := g.Bounds()
	w, h := b.Max.X-b.Min.X, b.Max.Y-b.Min.Y
	m := NewMatrix(1, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Data[y*w+x] = (255 - float64(g.GrayAt(x, y).Y)) / 255.0
		}
	}
	return m
}

// PrintASCII renders a [1×784] matrix as ASCII art in the terminal.
func PrintASCII(m *Matrix) {
	const chars = " ░▒▓█"
	for y := 0; y < 28; y++ {
		fmt.Print("│")
		for x := 0; x < 28; x++ {
			v := m.Data[y*28+x]
			idx := int(v * float64(len(chars)-1))
			fmt.Printf("%c%c", chars[idx], chars[idx])
		}
		fmt.Println("│")
	}
}