package tensor

import (
	"math"
	"testing"
)

// --- MatMul ---

func TestMatmul(t *testing.T) {
	tests := []struct {
		name      string
		aData     []float32
		aShape    []int
		bData     []float32
		bShape    []int
		want      []float32
		wantShape []int
	}{
		{
			name:   "2x3 times 3x2",
			aData:  []float32{1, 2, 3, 4, 5, 6},
			aShape: []int{2, 3},
			bData:  []float32{7, 8, 9, 10, 11, 12},
			bShape: []int{3, 2},
			// row 0: [1,2,3] @ col 0 [7,9,11] = 58,  col 1 [8,10,12] = 64
			// row 1: [4,5,6] @ col 0 [7,9,11] = 139, col 1 [8,10,12] = 154
			want:      []float32{58, 64, 139, 154},
			wantShape: []int{2, 2},
		},
		{
			name:      "1x1 identity",
			aData:     []float32{3},
			aShape:    []int{1, 1},
			bData:     []float32{4},
			bShape:    []int{1, 1},
			want:      []float32{12},
			wantShape: []int{1, 1},
		},
		{
			name:      "1x4 times 4x1 dot product",
			aData:     []float32{1, 2, 3, 4},
			aShape:    []int{1, 4},
			bData:     []float32{1, 1, 1, 1},
			bShape:    []int{4, 1},
			want:      []float32{10},
			wantShape: []int{1, 1},
		},
		{
			name:   "3x3 square",
			aData:  []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
			aShape: []int{3, 3},
			bData:  []float32{5, 6, 7, 8, 9, 10, 11, 12, 13},
			bShape: []int{3, 3},
			// identity times anything = anything
			want:      []float32{5, 6, 7, 8, 9, 10, 11, 12, 13},
			wantShape: []int{3, 3},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := From(tc.aData, tc.aShape...)
			b := From(tc.bData, tc.bShape...)
			got := Matmul(a, b)

			// check shape
			if len(got.Shape) != len(tc.wantShape) {
				t.Fatalf("shape rank: got %v want %v", got.Shape, tc.wantShape)
			}
			for i, d := range tc.wantShape {
				if got.Shape[i] != d {
					t.Fatalf("shape[%d]: got %d want %d", i, got.Shape[i], d)
				}
			}

			// check values
			if len(got.Data) != len(tc.want) {
				t.Fatalf("data length: got %d want %d", len(got.Data), len(tc.want))
			}
			for i, v := range tc.want {
				if math.Abs(float64(got.Data[i]-v)) > 1e-4 {
					t.Errorf("[%d] got %f want %f", i, got.Data[i], v)
				}
			}
		})
	}
}

func TestMatmulPanicsOnShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on shape mismatch, got none")
		}
	}()
	a := From([]float32{1, 2, 3, 4}, 2, 2)
	b := From([]float32{1, 2, 3}, 1, 3)
	Matmul(a, b) // [2,2] x [1,3] -- should panic
}

// --- Softmax ---

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		data  []float32
		shape []int
	}{
		{
			name:  "single row uniform",
			data:  []float32{1, 1, 1, 1},
			shape: []int{1, 4},
		},
		{
			name:  "single row varied",
			data:  []float32{1, 2, 3, 4},
			shape: []int{1, 4},
		},
		{
			name:  "two rows",
			data:  []float32{1, 2, 3, 4, 3, 2, 1, 0},
			shape: []int{2, 4},
		},
		{
			name:  "large values that would overflow without stability fix",
			data:  []float32{800, 801, 802},
			shape: []int{1, 3},
		},
		{
			name:  "negative values",
			data:  []float32{-1, -2, -3, -4},
			shape: []int{1, 4},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := From(tc.data, tc.shape...)
			Softmax(x)

			rows, cols := tc.shape[0], tc.shape[1]

			for i := 0; i < rows; i++ {
				// all values must be in (0, 1)
				for j := 0; j < cols; j++ {
					v := x.Data[i*cols+j]
					if v <= 0 || v >= 1 {
						t.Errorf("row %d col %d: value %f not in (0,1)", i, j, v)
					}
				}

				// each row must sum to 1
				var sum float32
				for j := 0; j < cols; j++ {
					sum += x.Data[i*cols+j]
				}
				if math.Abs(float64(sum-1.0)) > 1e-5 {
					t.Errorf("row %d: sum = %f, want 1.0", i, sum)
				}
			}
		})
	}
}

// --- LayerNorm ---

func TestLayerNorm(t *testing.T) {
	tests := []struct {
		name     string
		data     []float32
		shape    []int
		wantMean float32 // expected mean of output (before weight/bias)
	}{
		{
			name:     "single row ascending",
			data:     []float32{1, 2, 3, 4},
			shape:    []int{1, 4},
			wantMean: 0.0,
		},
		{
			name:     "two rows",
			data:     []float32{1, 2, 3, 4, 10, 20, 30, 40},
			shape:    []int{2, 4},
			wantMean: 0.0,
		},
		{
			name:     "already zero mean",
			data:     []float32{-3, -1, 1, 3},
			shape:    []int{1, 4},
			wantMean: 0.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cols := tc.shape[1]
			x := From(tc.data, tc.shape...)

			// identity weight and zero bias so we can check normalization directly
			w := New(cols)
			b := New(cols)
			for i := range w.Data {
				w.Data[i] = 1.0
			}

			out := LayerNorm(x, w, b, 1e-5)

			rows := tc.shape[0]
			for i := 0; i < rows; i++ {
				var mean float32
				for j := 0; j < cols; j++ {
					mean += out.Data[i*cols+j]
				}
				mean /= float32(cols)
				if math.Abs(float64(mean)) > 1e-4 {
					t.Errorf("row %d: mean = %f, want ~0", i, mean)
				}
			}
		})
	}
}

// --- GELU ---

func TestGELU(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
		tol   float32
	}{
		// reference values from:
		// torch.nn.functional.gelu(torch.tensor(x), approximate='tanh')
		{name: "zero", input: 0.0, want: 0.0, tol: 1e-5},
		{name: "one", input: 1.0, want: 0.8413, tol: 1e-3},
		{name: "negative", input: -1.0, want: -0.1587, tol: 1e-3},
		{name: "large", input: 10.0, want: 10.0, tol: 1e-3},
		{name: "small", input: -10.0, want: 0.0, tol: 1e-3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := From([]float32{tc.input}, 1, 1)
			out := GELU(x)
			if math.Abs(float64(out.Data[0]-tc.want)) > float64(tc.tol) {
				t.Errorf("GELU(%f) = %f, want %f (tol %f)",
					tc.input, out.Data[0], tc.want, tc.tol)
			}
		})
	}
}

// --- AddBias ---

func TestAddBias(t *testing.T) {
	tests := []struct {
		name   string
		xData  []float32
		xShape []int
		bData  []float32
		want   []float32
	}{
		{
			name:   "single row",
			xData:  []float32{1, 2, 3},
			xShape: []int{1, 3},
			bData:  []float32{10, 20, 30},
			want:   []float32{11, 22, 33},
		},
		{
			name:   "two rows same bias",
			xData:  []float32{1, 2, 3, 4, 5, 6},
			xShape: []int{2, 3},
			bData:  []float32{1, 1, 1},
			want:   []float32{2, 3, 4, 5, 6, 7},
		},
		{
			name:   "zero bias is identity",
			xData:  []float32{5, 10, 15},
			xShape: []int{1, 3},
			bData:  []float32{0, 0, 0},
			want:   []float32{5, 10, 15},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := From(tc.xData, tc.xShape...)
			b := From(tc.bData, len(tc.bData))
			AddBias(x, b)
			for i, v := range tc.want {
				if math.Abs(float64(x.Data[i]-v)) > 1e-5 {
					t.Errorf("[%d] got %f want %f", i, x.Data[i], v)
				}
			}
		})
	}
}
