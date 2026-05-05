package nn

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Matrix is a row-major 2D matrix of float64.
type Matrix struct {
	Rows, Cols int
	Data       []float64
}

// NewMatrix allocates a zero-filled matrix.
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

func (m *Matrix) At(i, j int) float64     { return m.Data[i*m.Cols+j] }
func (m *Matrix) Set(i, j int, v float64) { m.Data[i*m.Cols+j] = v }

// Randomize fills the matrix using Xavier/Glorot initialization.
func (m *Matrix) Randomize(rng *rand.Rand) {
	scale := math.Sqrt(2.0 / float64(m.Cols))
	for i := range m.Data {
		m.Data[i] = rng.NormFloat64() * scale
	}
}

// Dot computes (a × b): shapes (m×k) · (k×n) = (m×n).
func Dot(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("dot: dimension mismatch %dx%d · %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	out := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for k := 0; k < a.Cols; k++ {
			aik := a.At(i, k)
			for j := 0; j < b.Cols; j++ {
				out.Data[i*out.Cols+j] += aik * b.At(k, j)
			}
		}
	}
	return out
}

// Transpose returns a^T.
func Transpose(m *Matrix) *Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Set(j, i, m.At(i, j))
		}
	}
	return out
}

// Apply returns a new matrix with f applied element-wise.
func Apply(m *Matrix, f func(float64) float64) *Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	for i, v := range m.Data {
		out.Data[i] = f(v)
	}
	return out
}

// AddBias adds bias vector b (length == m.Cols) to every row of m in-place.
func AddBias(m *Matrix, b []float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i*m.Cols+j] += b[j]
		}
	}
}

// SumRows sums each column across all rows → slice of length m.Cols.
func SumRows(m *Matrix) []float64 {
	out := make([]float64, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out[j] += m.At(i, j)
		}
	}
	return out
}

// HadamardMul multiplies two same-shape matrices element-wise in-place (a *= b).
func HadamardMul(a, b *Matrix) {
	for i := range a.Data {
		a.Data[i] *= b.Data[i]
	}
}

// SliceRow extracts a single row as a new 1×cols matrix.
func SliceRow(m *Matrix, row int) *Matrix {
	out := NewMatrix(1, m.Cols)
	copy(out.Data, m.Data[row*m.Cols:(row+1)*m.Cols])
	return out
}

func (m *Matrix) String() string {
	var sb strings.Builder
	for i := 0; i < m.Rows; i++ {
		sb.WriteString("[")
		for j := 0; j < m.Cols; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			fmt.Fprintf(&sb, "%.4f", m.At(i, j))
		}
		sb.WriteString("]\n")
	}
	return sb.String()
}
