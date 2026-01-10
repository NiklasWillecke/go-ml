package nn

import "niklaswillecke/go-ml/mat"

type Layer struct {
	Weights *mat.Matrix
	Biases  *mat.Matrix
}
