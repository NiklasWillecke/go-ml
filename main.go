package main

import (
	nn "niklaswillecke/go-ml/NN"
	"niklaswillecke/go-ml/mnist"
)

func main() {

	trainImages, trainLabels := mnist.LoadData("./images/train-images.idx3-ubyte", "./images/train-labels.idx1-ubyte")
	testImages, testLabels := mnist.LoadData("./images/t10k-images.idx3-ubyte", "./images/t10k-labels.idx1-ubyte")

	// Model erstellen
	model := nn.NewMNISTModel()

	// Training Config
	desc := &nn.ModelTrainingDesc{
		TrainImages:  trainImages,
		TrainLabels:  trainLabels,
		TestImages:   testImages,
		TestLabels:   testLabels,
		Epochs:       4,
		LearningRate: 0.005,
	}

	// Training starten
	nn.Train(model, desc)
}
