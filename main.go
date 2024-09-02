package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {
	output()
}

func BiasesToTensor(biases []float64) t.Tensor {
	c := len(biases)
	r := c
	newBiases := []float64{}
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			newBiases = append(newBiases, biases[j])
		}
	}

	fmt.Println(c, r)
	fmt.Println(newBiases)

	return t.New(t.WithShape(r, c), t.WithBacking(newBiases))
}

// Testing function
func output() {
	inputs := t.New(t.WithShape(3, 4), t.WithBacking([]float64{
		1.0, 2.0, 3.0, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}))

	weights := t.New(t.WithShape(3, 4), t.WithBacking([]float64{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}))

	biases := []float64{2.0, 3.0, 0.5}
	bTensor := BiasesToTensor(biases)

	weights.T()

	product, _ := t.Dot(inputs, weights)
	productWithWeights, _ := t.Add(product, bTensor)

	fmt.Println("Inputs:")
	fmt.Println(inputs)

	fmt.Println("\n Weights:")
	fmt.Println(weights)

	fmt.Println("\n BiaseTensor:")
	fmt.Println(bTensor)

	fmt.Println("\n Products:")
	fmt.Println(product)

	fmt.Println("\n ProductWithWeights:")
	fmt.Println(productWithWeights)
}
