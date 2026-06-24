package tensor

// index element at row i, column j of a 2d tensor as Data[i*shape[1]+j]
type Tensor struct {
	Data  []float32
	Shape []int
}

// allocate zeroed memory
func New(shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: shape,
	}
}

// From will wrap the existing memory without copying
// use From when loading weights from the GGUF file, cause the weight data
// is already in memory and we dont want a second copy
func From(data []float32, shape ...int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}
