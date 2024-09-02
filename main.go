package main

import (
	"fmt"
	"go-ml/datagen"
)

func main() {
	sampleData := datagen.CreateData(100, 3)

	fmt.Println(sampleData)
}
