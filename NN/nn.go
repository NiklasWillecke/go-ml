package nn

import (
	"math"
	"niklaswillecke/go-ml/mat"
)

type Node struct {
	Data     *mat.Matrix
	Grad     *mat.Matrix
	prev     []*Node
	op       string
	backward func()
}

type ModelTrainingDesc struct {
	TrainImages []*mat.Matrix
	TrainLabels []*mat.Matrix
	TestImages  []*mat.Matrix
	TestLabels  []*mat.Matrix

	Epochs       int
	BatchSize    int
	LearningRate float32
}

func CreateNewNode(f *mat.Matrix, op string) *Node {
	return &Node{
		Data: f,
		Grad: mat.NewMatrix(f.Rows, f.Cols),
		op:   op,
	}
}

func NewOpNode(op string, data *mat.Matrix, prev ...*Node) *Node {
	return &Node{
		Data: data,
		Grad: mat.NewMatrix(data.Rows, data.Cols),
		prev: prev,
		op:   op,
	}
}

func Add(a, b *Node) *Node {

	tmp, _ := mat.Add(a.Data, b.Data)
	out := NewOpNode("Add", tmp, a, b)
	out.backward = func() {
		a.Grad, _ = mat.Add(a.Grad, out.Grad)
		b.Grad, _ = mat.Add(b.Grad, out.Grad)
	}

	return out
}

func Mul(a, b *Node) *Node {

	tmp, _ := mat.Mul(a.Data, b.Data)
	out := NewOpNode("Mul", tmp, a, b)
	out.backward = func() {

		bT := b.Data.Transpose()
		a.Grad, _ = mat.Mul(out.Grad, bT)

		aT := a.Data.Transpose()
		b.Grad, _ = mat.Mul(aT, out.Grad)
	}

	return out
}

func Relu(a *Node) *Node {
	tmp := mat.ReLU(a.Data)
	out := NewOpNode("Relu", tmp, a)
	out.backward = func() {
		for i := range a.Data.Data {
			if a.Data.Data[i] > 0 {
				a.Grad.Data[i] += out.Grad.Data[i]
			} else {
				a.Grad.Data[i] += 0
			}
		}
	}
	return out
}

func Softmax(a *Node) *Node {
	// Kopie der Eingabematrix erzeugen, um keine Seiteneffekte zu haben
	tmp := mat.NewMatrix(a.Data.Rows, a.Data.Cols)
	copy(tmp.Data, a.Data.Data)

	tmp.SoftMax()

	out := NewOpNode("Softmax", tmp, a)

	out.backward = func() {
		// dL/dz = s * (dL/ds - sum_j (dL/ds_j * s_j))
		gradSum := float64(0)
		for i := range tmp.Data {
			gradSum += out.Grad.Data[i] * tmp.Data[i]
		}
		for i := range a.Grad.Data {
			a.Grad.Data[i] += tmp.Data[i] * (out.Grad.Data[i] - gradSum)
		}
	}
	return out
}

func startTopo(v *Node) []*Node {
	topo := []*Node{}
	visited := map[*Node]bool{}
	return buildTopo(v, topo, visited)
}

func buildTopo(v *Node, topo []*Node, visited map[*Node]bool) []*Node {
	if !visited[v] {
		visited[v] = true
		for _, prev := range v.prev {
			topo = buildTopo(prev, topo, visited)
		}
		topo = append(topo, v)
	}
	return topo
}

// TODO:
func CrossEntropyLoss(pred *Node, target *mat.Matrix) *Node {
	sum := 0.0
	for i := range pred.Data.Data {
		if target.Data[i] > 0 {
			sum += -target.Data[i] * math.Log(math.Max(pred.Data.Data[i], 1e-15))
		}
	}

	loss := mat.NewMatrix(1, 1)
	loss.Data[0] = sum

	out := NewOpNode("CrossEntropy", loss, pred)

	out.backward = func() {
		for i := range pred.Data.Data {
			pred.Grad.Data[i] += pred.Data.Data[i] - target.Data[i]
		}
	}
	return out
}
