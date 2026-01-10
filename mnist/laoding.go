package mnist

import (
	"encoding/binary"
	"fmt"
	"io"
	"niklaswillecke/go-ml/mat"
	"os"
)

// Liest Bilder aus MNIST-Datei
func LoadMNISTImages(path string) ([][]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numImages, numRows, numCols int32
	err = binary.Read(f, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}

	if magic != 2051 {
		return nil, fmt.Errorf("ungültige Magic Number für Images: %d", magic)
	}

	binary.Read(f, binary.BigEndian, &numImages)
	binary.Read(f, binary.BigEndian, &numRows)
	binary.Read(f, binary.BigEndian, &numCols)

	images := make([][]byte, numImages)
	for i := int32(0); i < numImages; i++ {
		img := make([]byte, numRows*numCols)
		_, err := io.ReadFull(f, img)
		if err != nil {
			return nil, err
		}
		images[i] = img
	}
	return images, nil
}

// Liest Labels aus MNIST-Datei
func LoadMNISTLabels(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numLabels int32
	err = binary.Read(f, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}

	if magic != 2049 {
		return nil, fmt.Errorf("ungültige Magic Number für Labels: %d", magic)
	}

	binary.Read(f, binary.BigEndian, &numLabels)

	labels := make([]byte, numLabels)
	_, err = io.ReadFull(f, labels)
	return labels, err
}

func ConvertImagesToMatrixList(images [][]byte) []*mat.Matrix {
	result := make([]*mat.Matrix, len(images))

	for i := 0; i < len(images); i++ {
		m := mat.NewMatrix(784, 1)
		for j := 0; j < 784; j++ {
			m.Data[j] = float64(images[i][j]) / 255.0
		}
		result[i] = m
	}

	return result
}

func ConvertLabelsToMatrixList(labels []byte) []*mat.Matrix {
	result := make([]*mat.Matrix, len(labels))

	for i := range labels {
		m := mat.NewMatrix(10, 1)
		label := int(labels[i])
		m.Data[label] = 1.0
		result[i] = m
	}

	return result
}

func LoadData(image_path, label_path string) ([]*mat.Matrix, []*mat.Matrix) {
	images, err := LoadMNISTImages(image_path)
	if err != nil {
		panic(err)
	}
	labels, err := LoadMNISTLabels(label_path)
	if err != nil {
		panic(err)
	}

	PrintMNISTImageColor(images[0], 28, 28)

	trainImages := ConvertImagesToMatrixList(images)
	trainLabels := ConvertLabelsToMatrixList(labels)

	return trainImages, trainLabels
}
