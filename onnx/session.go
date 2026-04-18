// Package onnx wraps onnxruntime_go with a simpler interface that handles
// tensor memory lifecycle, dynamic shapes, and named I/O.
package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Session wraps an ORT dynamic session and manages tensor lifecycles.
// All input tensors are destroyed after Run returns.
type Session struct {
	inner       *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
}

// SessionOptions controls device placement and ORT execution settings.
type SessionOptions struct {
	// CUDADeviceID >= 0 enables CUDA on that device; -1 means CPU-only.
	CUDADeviceID int
}

// DefaultSessionOptions returns CPU-only options.
func DefaultSessionOptions() SessionOptions {
	return SessionOptions{CUDADeviceID: -1}
}

// NewSession loads an ONNX model from onnxPath and creates a session.
// inputNames and outputNames must match the names in the ONNX graph.
func NewSession(onnxPath string, inputNames, outputNames []string, opts SessionOptions) (*Session, error) {
	ortOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnx: create session options: %w", err)
	}
	defer ortOpts.Destroy()

	if opts.CUDADeviceID >= 0 {
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("onnx: create CUDA provider options: %w", err)
		}
		if err := cuda.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", opts.CUDADeviceID),
		}); err != nil {
			cuda.Destroy()
			return nil, fmt.Errorf("onnx: configure CUDA provider: %w", err)
		}
		if err := ortOpts.AppendExecutionProviderCUDA(cuda); err != nil {
			cuda.Destroy()
			return nil, fmt.Errorf("onnx: append CUDA provider: %w", err)
		}
		cuda.Destroy()
	}

	inner, err := ort.NewDynamicAdvancedSession(onnxPath, inputNames, outputNames, ortOpts)
	if err != nil {
		return nil, fmt.Errorf("onnx: load %s: %w", onnxPath, err)
	}
	return &Session{
		inner:       inner,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil
}

// RunFloat32 runs inference with named float32 inputs and returns named float32 outputs.
//
// inputs is a map from input name → (flat []float32 data, shape []int64).
// Returns a map from output name → (flat []float32 data, shape []int64).
//
// The caller owns the returned slices; the function handles all ORT tensor lifetimes.
func (s *Session) RunFloat32(inputs map[string]Float32Input) (map[string]Float32Output, error) {
	// Build ordered input tensors matching s.inputNames.
	inTensors := make([]ort.ArbitraryTensor, len(s.inputNames))
	for i, name := range s.inputNames {
		inp, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("onnx: missing input %q", name)
		}
		t, err := ort.NewTensor(ort.NewShape(inp.Shape...), inp.Data)
		if err != nil {
			// Destroy already-created tensors.
			for j := 0; j < i; j++ {
				inTensors[j].(*ort.Tensor[float32]).Destroy()
			}
			return nil, fmt.Errorf("onnx: create input tensor %q: %w", name, err)
		}
		inTensors[i] = t
	}
	defer func() {
		for _, t := range inTensors {
			t.(*ort.Tensor[float32]).Destroy()
		}
	}()

	// Build output tensors (nil = ORT allocates).
	outTensors := make([]ort.ArbitraryTensor, len(s.outputNames))
	defer func() {
		for _, t := range outTensors {
			if t != nil {
				t.(*ort.Tensor[float32]).Destroy()
			}
		}
	}()

	if err := s.inner.Run(inTensors, outTensors); err != nil {
		return nil, fmt.Errorf("onnx: run: %w", err)
	}

	// Copy output data into Go-owned slices before destroying ORT tensors.
	result := make(map[string]Float32Output, len(s.outputNames))
	for i, name := range s.outputNames {
		t := outTensors[i].(*ort.Tensor[float32])
		shape := t.GetShape()
		data := t.GetData()
		shapeCopy := make([]int64, len(shape))
		copy(shapeCopy, shape)
		dataCopy := make([]float32, len(data))
		copy(dataCopy, data)
		result[name] = Float32Output{Data: dataCopy, Shape: shapeCopy}
	}
	return result, nil
}

// RunMixed runs inference where inputs may be float32 or int64 tensors.
// Use this for encoder-decoder models where token IDs are int64.
func (s *Session) RunMixed(inputs map[string]AnyInput) (map[string]Float32Output, error) {
	inTensors := make([]ort.ArbitraryTensor, len(s.inputNames))
	var toDestroy []func()

	for i, name := range s.inputNames {
		inp, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("onnx: missing input %q", name)
		}
		switch v := inp.(type) {
		case Float32Input:
			t, err := ort.NewTensor(ort.NewShape(v.Shape...), v.Data)
			if err != nil {
				for _, f := range toDestroy {
					f()
				}
				return nil, fmt.Errorf("onnx: create float input %q: %w", name, err)
			}
			inTensors[i] = t
			tc := t
			toDestroy = append(toDestroy, func() { tc.Destroy() })
		case Int64Input:
			t, err := ort.NewTensor(ort.NewShape(v.Shape...), v.Data)
			if err != nil {
				for _, f := range toDestroy {
					f()
				}
				return nil, fmt.Errorf("onnx: create int64 input %q: %w", name, err)
			}
			inTensors[i] = t
			tc := t
			toDestroy = append(toDestroy, func() { tc.Destroy() })
		case Int32Input:
			t, err := ort.NewTensor(ort.NewShape(v.Shape...), v.Data)
			if err != nil {
				for _, f := range toDestroy {
					f()
				}
				return nil, fmt.Errorf("onnx: create int32 input %q: %w", name, err)
			}
			inTensors[i] = t
			tc := t
			toDestroy = append(toDestroy, func() { tc.Destroy() })
		default:
			for _, f := range toDestroy {
				f()
			}
			return nil, fmt.Errorf("onnx: unsupported input type for %q: %T", name, inp)
		}
	}
	defer func() {
		for _, f := range toDestroy {
			f()
		}
	}()

	outTensors := make([]ort.ArbitraryTensor, len(s.outputNames))
	defer func() {
		for _, t := range outTensors {
			if t == nil {
				continue
			}
			switch v := t.(type) {
			case *ort.Tensor[float32]:
				v.Destroy()
			case *ort.Tensor[int64]:
				v.Destroy()
			case *ort.Tensor[int32]:
				v.Destroy()
			}
		}
	}()

	if err := s.inner.Run(inTensors, outTensors); err != nil {
		return nil, fmt.Errorf("onnx: run: %w", err)
	}

	result := make(map[string]Float32Output, len(s.outputNames))
	for i, name := range s.outputNames {
		switch t := outTensors[i].(type) {
		case *ort.Tensor[float32]:
			shape := t.GetShape()
			data := t.GetData()
			shapeCopy := make([]int64, len(shape))
			copy(shapeCopy, shape)
			dataCopy := make([]float32, len(data))
			copy(dataCopy, data)
			result[name] = Float32Output{Data: dataCopy, Shape: shapeCopy}
		case *ort.Tensor[int64]:
			shape := t.GetShape()
			data := t.GetData()
			shapeCopy := make([]int64, len(shape))
			copy(shapeCopy, shape)
			dataCopy := make([]float32, len(data))
			for j, v := range data {
				dataCopy[j] = float32(v)
			}
			result[name] = Float32Output{Data: dataCopy, Shape: shapeCopy}
		case *ort.Tensor[int32]:
			shape := t.GetShape()
			data := t.GetData()
			shapeCopy := make([]int64, len(shape))
			copy(shapeCopy, shape)
			dataCopy := make([]float32, len(data))
			for j, v := range data {
				dataCopy[j] = float32(v)
			}
			result[name] = Float32Output{Data: dataCopy, Shape: shapeCopy}
		default:
			return nil, fmt.Errorf("onnx: unexpected output tensor type for %q: %T", name, outTensors[i])
		}
	}
	return result, nil
}

// Close releases the session.
func (s *Session) Close() error {
	return s.inner.Destroy()
}

// InputNames returns the session's expected input names in order.
func (s *Session) InputNames() []string { return s.inputNames }

// OutputNames returns the session's output names in order.
func (s *Session) OutputNames() []string { return s.outputNames }
