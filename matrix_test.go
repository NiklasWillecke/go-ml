package main

import (
	"niklaswillecke/go-ml/mat"
	"testing"
)

func BenchmarkMatMul(t *testing.B) {
	a := mat.NewMatrix(1024, 1024)
	b := mat.NewMatrix(1024, 1024)

	// Setup: Matrix mit Werten füllen
	a.FillRandom()
	b.FillRandom()

	t.ResetTimer() // Timer nach Setup zurücksetzen

	for i := 0; i < t.N; i++ {
		mat.Mul(a, b)
	}
}

func Test_Transpose(t *testing.T) {
	a := mat.NewMatrix(4, 4)

	for i := range a.Data {

		a.Data[i] = float64(i)
	}
	a.Print()

}
