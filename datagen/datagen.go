package datagen

import (
	"math/rand"

	t "gorgonia.org/tensor"
)

// Returns a 1D Tensor of random float32
func CreateData(classes int, sample int) t.Tensor {
	data := []float32{}
	for i := 0; i < classes; i++ {
		for j := 0; j < sample; j++ {
			data = append(data, rand.Float32())
		}
	}

	return t.New(t.WithShape(classes, sample), t.WithBacking(data))
}
