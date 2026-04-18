package onnx

// Float32Input describes a named float32 input tensor.
type Float32Input struct {
	Data  []float32
	Shape []int64
}

// Int64Input describes a named int64 input tensor.
type Int64Input struct {
	Data  []int64
	Shape []int64
}

// Float32Output contains the data and shape of a float32 output tensor.
type Float32Output struct {
	Data  []float32
	Shape []int64 // e.g. [batch, time, vocab] for logits
}

// Int32Input describes a named int32 input tensor.
type Int32Input struct {
	Data  []int32
	Shape []int64
}

// AnyInput is implemented by Float32Input, Int64Input, and Int32Input.
type AnyInput interface{ isInput() }

func (Float32Input) isInput() {}
func (Int64Input) isInput()   {}
func (Int32Input) isInput()   {}

// Elem returns the element at multi-dimensional index idx within out.
// Panics if idx is out of bounds.
func (o Float32Output) Elem(idx ...int) float32 {
	flat := 0
	stride := 1
	for i := len(idx) - 1; i >= 0; i-- {
		flat += idx[i] * stride
		stride *= int(o.Shape[i])
	}
	return o.Data[flat]
}

// Slice returns a sub-slice along the first dimension (batch or time).
// Returns a view into the underlying Data slice — no copy.
func (o Float32Output) Slice(i int) Float32Output {
	if len(o.Shape) < 2 {
		panic("onnx: Slice called on 1-D tensor")
	}
	innerSize := 1
	for _, s := range o.Shape[1:] {
		innerSize *= int(s)
	}
	sub := o.Data[i*innerSize : (i+1)*innerSize]
	newShape := make([]int64, len(o.Shape)-1)
	copy(newShape, o.Shape[1:])
	return Float32Output{Data: sub, Shape: newShape}
}

// TimeStep returns the vocabulary logits at time step t for a [batch=1, time, vocab] tensor.
func (o Float32Output) TimeStep(t int) []float32 {
	if len(o.Shape) != 3 {
		panic("onnx: TimeStep requires shape [1, time, vocab]")
	}
	vocab := int(o.Shape[2])
	return o.Data[t*vocab : (t+1)*vocab]
}

// PadBatch builds a batched [B, F, maxT] float32 slice from variable-length inputs.
// Each item in inputs is a flat [F × Ti] slice. Returns the padded tensor and
// the per-item frame counts (lengths).
func PadBatch(inputs [][]float32, nFeats int) (padded []float32, lengths []int64) {
	maxT := 0
	for _, inp := range inputs {
		t := len(inp) / nFeats
		if t > maxT {
			maxT = t
		}
	}
	B := len(inputs)
	padded = make([]float32, B*nFeats*maxT)
	lengths = make([]int64, B)
	for b, inp := range inputs {
		t := len(inp) / nFeats
		lengths[b] = int64(t)
		// Copy row by row: inp is [nFeats × t], padded is [nFeats × maxT] for this batch item.
		for f := 0; f < nFeats; f++ {
			dst := b*nFeats*maxT + f*maxT
			src := f * t
			copy(padded[dst:dst+t], inp[src:src+t])
		}
	}
	return
}
