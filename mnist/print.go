package mnist

import "fmt"

func PrintMNISTImageColor(img []byte, rows, cols int) {
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			pixel := img[r*cols+c]
			gray := int(pixel)
			fmt.Printf("\x1b[48;2;%d;%d;%dm  \x1b[0m", gray, gray, gray)
		}
		fmt.Println()
	}
}
