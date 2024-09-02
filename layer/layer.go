package layer

import (
	t "gorgonia.org/tensor"
)

type Layer struct {
	Weights t.Tensor
	Output  t.Tensor
	Biases  []float32
}

func NewLayer(numInputs int, numNuerons int) Layer {
	weights := t.New(
		t.WithShape(numInputs, numNuerons),
		t.WithBacking(t.Random(t.Float32, numInputs*numNuerons)),
	)

	var test float32 = 0.1
	weights, err := weights.MulScalar(test, false)
	if err != nil {
		panic(err)
	}

	return Layer{
		Weights: weights,
		Biases:  GenBiases(numNuerons),
		Output:  nil,
	}
}

// Weights = 2x3 (each input is size 2, and results in 3 output nums)
// Input = 100x3
func (l *Layer) Forward(input t.Tensor) {
	dotSum, err := t.Dot(input, l.Weights)
	if err != nil {
		panic(err)
	}
	outputShape := dotSum.Shape()

	bTensor := l.BiasesToTensor(outputShape[0])

	dotSumWithBias, err := t.Add(dotSum, bTensor)
	if err != nil {
		panic(err)
	}
	l.Output = dotSumWithBias
}

// Transforms our 1d biases to a tensor that fits the input for addition.
func (l *Layer) BiasesToTensor(cols int) t.Tensor {
	data := []float32{}
	rowSize := len(l.Biases)
	for i := 0; i < cols; i++ {
		for j := 0; j < rowSize; j++ {
			data = append(data, l.Biases[j])
		}
	}
	return t.New(t.WithShape(cols, rowSize), t.WithBacking(data))
}

func GenBiases(numNuerons int) []float32 {
	data := []float32{}
	for i := 0; i < numNuerons; i++ {
		data = append(data, 0.00)
	}
	return data
}
