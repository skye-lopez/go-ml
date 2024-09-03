package activators

import (
	"math"

	t "gorgonia.org/tensor"
)

func ReLU(input t.Tensor) t.Tensor {
	iter := input.Iterator()
	iter.SetForward()

	for !iter.Done() {
		cords := iter.Coord()
		value, err := input.At(cords...)
		if err != nil {
			panic(err)
		}

		// This is ensuring a 32 float dtype
		v := value.(float32)
		var zero float32 = 0.0
		if v < 0 {
			input.SetAt(zero, cords...)
		}
		iter.Next()
	}

	return input
}

func SoftMax(outputs t.Tensor) []float32 {
	e := 2.71828182846
	exponents := []float32{}
	var sum float32 = 0.0
	iter := outputs.Iterator()
	iter.SetForward()

	for !iter.Done() {
		cords := iter.Coord()
		value, err := outputs.At(cords...)
		if err != nil {
			panic(err)
		}
		v := value.(float32)
		exponent := math.Pow(e, float64(v))
		exponent32 := float32(exponent)
		exponents = append(exponents, exponent32)
		sum += exponent32

		iter.Next()
	}

	for i := 0; i < len(exponents); i++ {
		exponents[i] = exponents[i] / sum
	}

	return exponents
}
