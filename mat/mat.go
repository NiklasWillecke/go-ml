package mat

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
)

type Matrix struct {
	Rows, Cols int
	Data       []float64
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

func (m *Matrix) Set(row int, col int, value float64) {
	m.Data[row*m.Cols+col] = value
}

func (m *Matrix) FillRandom() {

	for i := range m.Data {

		m.Data[i] = rand.Float64()
	}
}

func Add(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		return nil, errors.New("incompatible dimensions")
	}

	result := &Matrix{
		Rows: a.Rows,
		Cols: a.Cols,
		Data: make([]float64, a.Rows*a.Cols),
	}
	for i := range a.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return result, nil
}

func Sub(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		return nil, errors.New("incompatible dimensions")
	}

	result := &Matrix{
		Rows: a.Rows,
		Cols: a.Cols,
		Data: make([]float64, a.Rows*a.Cols),
	}
	for i := range a.Data {
		result.Data[i] = a.Data[i] - b.Data[i]
	}

	return result, nil
}

func Mul(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, errors.New("incompatible dimensions")
	}

	result := &Matrix{
		Rows: a.Rows,
		Cols: b.Cols,
		Data: make([]float64, a.Rows*b.Cols),
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i*a.Cols+k] * b.Data[k*b.Cols+j]
			}
			result.Set(i, j, sum)
		}
	}

	return result, nil
}

func Scale(a *Matrix, s float64) *Matrix {

	result := &Matrix{
		Rows: a.Rows,
		Cols: a.Cols,
		Data: make([]float64, a.Rows*a.Cols),
	}

	for i := range a.Data {
		result.Data[i] = a.Data[i] * s
	}

	return result
}

func (m *Matrix) Transpose() *Matrix {
	transpose := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			transpose.Data[i*m.Rows+j] = m.Data[j*m.Cols+i]
		}
	}
	return transpose
}

func (m *Matrix) Print() {
	for i := range m.Rows {
		for j := range m.Cols {
			fmt.Printf("%6.2f ", m.Data[i*m.Cols+j])
		}
		fmt.Println()
	}
	fmt.Println()
}

func ReLU(a *Matrix) *Matrix {
	result := &Matrix{
		Rows: a.Rows,
		Cols: a.Cols,
		Data: make([]float64, a.Rows*a.Cols),
	}
	for i := range a.Data {
		result.Data[i] = math.Max(0, a.Data[i])
	}

	return result
}

func (m *Matrix) SoftMax() {
	if m.Rows != 1 && m.Cols != 1 {
		panic("SoftMax nur für Vektoren implementiert")
	}

	maxVal := m.Data[0]
	for _, v := range m.Data {
		maxVal = math.Max(maxVal, v)
	}

	sum := 0.0
	for i := range m.Data {
		m.Data[i] = math.Exp(m.Data[i] - maxVal)
		sum += m.Data[i]
	}

	for i := range m.Data {
		m.Data[i] /= sum
	}
}

func CrossEntropieLoss(pred, target *Matrix) (*Matrix, error) {

	if pred.Rows != target.Rows || pred.Cols != target.Cols {
		return nil, errors.New("incompatible matrix dimensions")
	}

	size := pred.Rows * pred.Cols
	result := &Matrix{
		Rows: pred.Rows,
		Cols: pred.Cols,
		Data: make([]float64, size),
	}

	for i := 0; i < size; i++ {
		p := target.Data[i]
		q := pred.Data[i]

		if p == 0.0 {
			result.Data[i] = 0.0
		} else {
			result.Data[i] = -p * math.Log(q)
		}
	}

	return result, nil
}

func (m *Matrix) XavierInitFill() {
	bound := math.Sqrt(6.0 / float64(m.Cols+m.Rows))
	for i := range m.Data {
		m.Data[i] = rand.Float64()*(2*bound) - bound
	}
}

// Returned Max Value in Matrix
func (m *Matrix) Argmax() int {

	max := 0
	for i := range m.Data {
		if m.Data[i] > m.Data[max] {
			max = i
		}
	}
	return max
}
