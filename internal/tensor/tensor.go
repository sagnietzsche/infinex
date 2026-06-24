package tensor

import "fmt"

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

// every single weight propjection in GPT-2 is going to be a call to this function.
// GPT-2 stoes the weight matrices as [out_features,in_features] and computes as x @ W ^ T

func MatmulTrans(a, b *Tensor) *Tensor {
	// a -> [m,k]
	// b -> [n,k] (using transposed layout)

	m, k := a.Shape[0], a.Shape[1]
	n, k2 := b.Shape[0], b.Shape[1]
	if k != k2 {
		panic(fmt.Sprintf(
			"ERROR MatmulTrans : inner dimensions must match, got [%d, %d] x [%d, %d]^T",
			m, k, n, k2))
	}

	out := New(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[j*k+p]
			}
			out.Data[i*n+j] = sum
		}
	}
	return out
}
