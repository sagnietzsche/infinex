package tensor

import (
	"fmt"
	"math"
)

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
	s := make([]int, len(shape))
	copy(s, shape)
	return &Tensor{
		Data:  make([]float32, size),
		Shape: s,
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
			//this loop allows the both a and b to be accessed sequentially
			// in the inner loop because we are reading along the row of b (column of W)
			// cache friendly
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[j*k+p]
			}
			out.Data[i*n+j] = sum
		}
	}
	return out
}

// naive matmul for the attention score computation where we multiply Q by K without transposing
func Matmul(a, b *Tensor) *Tensor {
	m, k := a.Shape[0], a.Shape[1]
	k2, n := b.Shape[0], b.Shape[1]

	if k != k2 {
		panic(fmt.Sprintf(
			"ERROR Matmul : inner dimensions must match, got [%d, %d] x [%d, %d]^T",
			m, k, n, k2))
	}

	out := New(m, n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			out.Data[i*n+j] = sum
		}
	}

	return out
}

// Bias Vector [out_features] added to every row of x.
// every linear projection in the model does MatMulTrans followed by AddBias
// making it mutate inplace , means we do not allocate a third tensor just to hold the sum
func AddBias(x, bias *Tensor) {
	cols := x.Shape[1]
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < cols; j++ {
			x.Data[i*cols+j] += bias.Data[j]
		}
	}
}

// Add used in for adding residula connections.

func Add(a, b *Tensor) *Tensor {
	out := New(a.Shape...)
	for i, v := range a.Data {
		out.Data[i] = v + b.Data[i]
	}
	return out
}

//softmax is called on the attention score matrix [seq_len, seq_len], row by row

func Softmax(t *Tensor) {
	rows, cols := t.Shape[0], t.Shape[1]

	for i := 0; i < rows; i++ {
		row := t.Data[i*cols : (i+1)*cols]

		//finding the max for numerical stability
		max := row[0]
		for _, v := range row {
			if v > max {
				max = v
			}
		}

		//exp and sum
		var sum float32
		for j, v := range row {
			e := float32(math.Exp(float64(v - max)))
			row[j] = e
			sum += e
		}

		//normalize
		for j := range row {
			row[j] /= sum
		}
	}
}

//GELU -> applied inside the MLP block between the up-projects and the down projection
// it is the only place in the model we can call this

func GELU(t *Tensor) *Tensor {
	out := New(t.Shape...)
	for i, v := range t.Data {
		x := float64(v)
		out.Data[i] = float32(
			0.5 * x *
				//hf trick to work with gpt-2
				(1.0 + math.Tanh(
					math.Sqrt(2.0/math.Pi)*
						(x*0.044715*x*x*x))))
	}
	return out
}
