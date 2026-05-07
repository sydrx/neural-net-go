package nn

import (
	"image"
	"image/color"
	"math"
	"math/rand"
)

func AugmentBatch(x *Matrix, rng *rand.Rand) *Matrix {
	result := NewMatrix(x.Rows, x.Cols)
	for i := 0; i < x.Rows; i++ {
		src := x.Data[i*x.Cols : (i+1)*x.Cols]
		dst := result.Data[i*x.Cols : (i+1)*x.Cols]
		augmentSingle(src, dst, rng)
	}
	return result
}

func augmentSingle(src, dst []float64, rng *rand.Rand) {
	// Convert to 2D image
	var img [28][28]float64
	vecToImg(src, &img)

	// Apply random transformations
	if rng.Float64() < 0.7 {
		angle := (rng.Float64() - 0.5) * 30.0 * math.Pi / 180.0 // ±15 degrees
		img = rotateImg(img, angle)
	}

	if rng.Float64() < 0.6 {
		scale := 0.85 + rng.Float64()*0.30 // 0.85-1.15
		img = scaleImg(img, scale)
	}

	if rng.Float64() < 0.6 {
		tx := (rng.Float64() - 0.5) * 6.0 // ±3px
		ty := (rng.Float64() - 0.5) * 6.0 // ±3px
		img = translateImg(img, tx, ty)
	}

	if rng.Float64() < 0.5 {
		radius := 1 + rng.Intn(2) // radius 1-2
		img = dilateImg(img, radius)
	}

	if rng.Float64() < 0.4 {
		addNoise(&img, 0.08)
	}

	// Convert back to vector
	imgToVec(&img, dst)
}

func vecToImg(vec []float64, img *[28][28]float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			img[i][j] = vec[i*28+j]
		}
	}
}

func imgToVec(img *[28][28]float64, vec []float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			vec[i*28+j] = img[i][j]
		}
	}
}

func rotateImg(src [28][28]float64, angle float64) [28][28]float64 {
	var dst [28][28]float64
	cx, cy := 13.5, 13.5 // center of 28x28 image

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			// Translate to origin
			x := float64(j) - cx
			y := float64(i) - cy

			// Rotate
			newX := x*math.Cos(angle) - y*math.Sin(angle)
			newY := x*math.Sin(angle) + y*math.Cos(angle)

			// Translate back and sample
			newX += cx
			newY += cy

			dst[i][j] = bilinear(src, newX, newY)
		}
	}
	return dst
}

func scaleImg(src [28][28]float64, scale float64) [28][28]float64 {
	var dst [28][28]float64
	cx, cy := 13.5, 13.5

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			// Translate to origin
			x := float64(j) - cx
			y := float64(i) - cy

			// Scale
			newX := x / scale
			newY := y / scale

			// Translate back and sample
			newX += cx
			newY += cy

			dst[i][j] = bilinear(src, newX, newY)
		}
	}
	return dst
}

func translateImg(src [28][28]float64, tx, ty float64) [28][28]float64 {
	var dst [28][28]float64

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			newX := float64(j) + tx
			newY := float64(i) + ty
			dst[i][j] = bilinear(src, newX, newY)
		}
	}
	return dst
}

func dilateImg(src [28][28]float64, radius int) [28][28]float64 {
	var dst [28][28]float64

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			maxVal := 0.0
			for di := -radius; di <= radius; di++ {
				for dj := -radius; dj <= radius; dj++ {
					ni, nj := i+di, j+dj
					if ni >= 0 && ni < 28 && nj >= 0 && nj < 28 {
						if src[ni][nj] > maxVal {
							maxVal = src[ni][nj]
						}
					}
				}
			}
			dst[i][j] = maxVal
		}
	}
	return dst
}

func addNoise(img *[28][28]float64, strength float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			noise := (rand.Float64() - 0.5) * 2 * strength
			img[i][j] += noise
			if img[i][j] < 0 {
				img[i][j] = 0
			} else if img[i][j] > 1 {
				img[i][j] = 1
			}
		}
	}
}

func bilinear(src [28][28]float64, fx, fy float64) float64 {
	x := math.Floor(fx)
	y := math.Floor(fy)
	dx := fx - x
	dy := fy - y

	ix, iy := int(x), int(y)
	ix1, iy1 := ix+1, iy+1

	// Zero padding for out-of-bounds
	if ix < 0 || ix >= 28 || iy < 0 || iy >= 28 ||
		ix1 < 0 || ix1 >= 28 || iy1 < 0 || iy1 >= 28 {
		return 0.0
	}

	v00 := src[iy][ix]
	v10 := src[iy][ix1]
	v01 := src[iy1][ix]
	v11 := src[iy1][ix1]

	v0 := v00*(1-dx) + v10*dx
	v1 := v01*(1-dx) + v11*dx

	return v0*(1-dy) + v1*dy
}

func DilateGray(g *image.Gray, r int) *image.Gray {
	bounds := g.Bounds()
	result := image.NewGray(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			maxVal := uint8(0)
			for dy := -r; dy <= r; dy++ {
				for dx := -r; dx <= r; dx++ {
					ny, nx := y+dy, x+dx
					if ny >= bounds.Min.Y && ny < bounds.Max.Y &&
						nx >= bounds.Min.X && nx < bounds.Max.X {
						val := g.GrayAt(nx, ny).Y
						if val > maxVal {
							maxVal = val
						}
					}
				}
			}
			result.SetGray(x, y, color.Gray{Y: maxVal})
		}
	}
	return result
}
