package main

import (
	"fmt"
	"go-ml/datagen"
	"go-ml/layer"
)

func main() {
	sampleData := datagen.CreateData(100)

	l := layer.NewLayer(2, 3)

	l.Forward(sampleData)

	fmt.Println(l.Output)
}
