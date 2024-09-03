package main

import (
	"fmt"
	a "go-ml/activators"
	"go-ml/datagen"
	"go-ml/layer"
)

func main() {
	sampleData := datagen.CreateData(100)

	l := layer.NewLayer(2, 3)
	l.Forward(sampleData)
	reluOutput := a.ReLU(l.Output)

	l2 := layer.NewLayer(3, 3)
	l2.Forward(reluOutput)

	softMaxOutput := a.SoftMax(l2.Output)

	fmt.Println(softMaxOutput)
}
