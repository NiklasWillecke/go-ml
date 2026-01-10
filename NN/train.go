package nn

import (
	"fmt"
	"niklaswillecke/go-ml/mat"
)

type MNISTModel struct {
	w0, w1, w2 *Node
	b0, b1, b2 *Node
}

func NewMNISTModel() *MNISTModel {
	m := &MNISTModel{

		w0: CreateNewNode(mat.NewMatrix(16, 784), "Weight"),
		w1: CreateNewNode(mat.NewMatrix(16, 16), "Weight"),
		w2: CreateNewNode(mat.NewMatrix(10, 16), "Weight"),
		b0: CreateNewNode(mat.NewMatrix(16, 1), "Bias"),
		b1: CreateNewNode(mat.NewMatrix(16, 1), "Bias"),
		b2: CreateNewNode(mat.NewMatrix(10, 1), "Bias"),
	}

	m.w0.Data.XavierInitFill()
	m.w1.Data.XavierInitFill()
	m.w2.Data.XavierInitFill()

	return m
}

// Glue all Layers together and create the computational Graph
func (m *MNISTModel) Forward(input *Node) *Node {
	z0_a := Mul(m.w0, input)
	z0_b := Add(z0_a, m.b0)
	a0 := Relu(z0_b)

	z1_a := Mul(m.w1, a0)
	z1_b := Add(z1_a, m.b1)
	z1_c := Add(z1_b, a0)
	a1 := Relu(z1_c)

	z2_a := Mul(m.w2, a1)
	z2_b := Add(z2_a, m.b2)

	return Softmax(z2_b)
}

func (m *MNISTModel) ZeroGrad() {
	params := []*Node{m.w0, m.w1, m.w2, m.b0, m.b1, m.b2}
	for _, p := range params {
		for i := range p.Grad.Data {
			p.Grad.Data[i] = 0
		}
	}
}

func (m *MNISTModel) UpdateWeights(lr float32) {
	params := []*Node{m.w0, m.w1, m.w2, m.b0, m.b1, m.b2}
	for _, p := range params {
		for i := range p.Data.Data {
			p.Data.Data[i] -= float64(lr) * p.Grad.Data[i]
		}
	}
}

func Train(model *MNISTModel, desc *ModelTrainingDesc) {
	for epoch := 0; epoch < desc.Epochs; epoch++ {

		Test(model, desc)
		totalLoss := 0.0
		numSamples := len(desc.TrainImages)
		for i := range numSamples {
			model.ZeroGrad()

			image := desc.TrainImages[i]
			label := desc.TrainLabels[i]
			input := CreateNewNode(image, "Input")

			pred := model.Forward(input)

			loss := CrossEntropyLoss(pred, label)
			LossSum := 0.0
			for _, l := range loss.Data.Data {
				LossSum += l
			}

			totalLoss += LossSum

			topo := startTopo(loss)

			for j := range topo {
				node := topo[len(topo)-1-j]
				if node.backward != nil {
					node.backward()
				}
			}

			model.UpdateWeights(desc.LearningRate)

			progressPct := (i * 100) / numSamples
			fmt.Printf("\rLoading... %d%%", progressPct)
		}

		avgLoss := totalLoss / float64(numSamples)
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch+1, avgLoss)
	}
}
