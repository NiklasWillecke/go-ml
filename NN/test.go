package nn

import (
	"fmt"
)

func Test(model *MNISTModel, desc *ModelTrainingDesc) {
	numSamples := len(desc.TestImages)
	correct := 0

	for i := range numSamples {
		image := desc.TestImages[i]
		label := desc.TestLabels[i]

		input := CreateNewNode(image, "Input")
		pred := model.Forward(input)

		if pred.Data.Argmax() == label.Argmax() {
			correct++
		}

		progressPct := (i * 100) / numSamples
		fmt.Printf("\rLoading... %d%%", progressPct)
	}

	fmt.Println()

	accuracy := float64(correct) / float64(numSamples) * 100
	fmt.Printf("Accuracy: %d/%d (%.2f%%)\n", correct, numSamples, accuracy)
}
