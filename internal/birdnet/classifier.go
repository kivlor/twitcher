// Package birdnet wraps the BirdNET v2.4 TFLite model: loading model + labels,
// running inference on 3 s × 48 kHz mono float32 chunks, and returning ranked
// species predictions with softmax confidences.
//
// The interpreter is single-threaded-use: Invoke is not safe for concurrent
// calls, so the caller (one inference worker goroutine) owns it exclusively.
package birdnet

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	tflite "github.com/tphakala/go-tflite"
)

// InputSamples is the BirdNET v2.4 input window: 3 s @ 48 kHz, mono.
const InputSamples = 144000

// SampleRate is the audio format the model expects.
const SampleRate = 48000

// Result is a single ranked species prediction for one 3-s window.
type Result struct {
	Index          int
	Confidence     float32
	ScientificName string
	CommonName     string
}

// Classifier is a loaded BirdNET v2.4 interpreter + labels.
type Classifier struct {
	model   *tflite.Model
	opts    *tflite.InterpreterOptions
	interp  *tflite.Interpreter
	in      *tflite.Tensor
	labels  []string
	nOut    int
	inferNs int64
	nInfer  int64
}

// Load loads the TFLite model and (optionally) the label list. labelsPath may
// be empty; predictions then carry index-only labels. threads sets the number
// of TFLite inference threads (2 measured well on the Pi 3B).
func Load(modelPath, labelsPath string, threads int) (*Classifier, error) {
	if threads <= 0 {
		threads = 2
	}
	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		return nil, fmt.Errorf("tflite: NewModelFromFile(%q) returned nil", modelPath)
	}
	opts := tflite.NewInterpreterOptions()
	if opts == nil {
		model.Delete()
		return nil, fmt.Errorf("tflite: NewInterpreterOptions returned nil")
	}
	opts.SetNumThread(threads)
	interp := tflite.NewInterpreter(model, opts)
	if interp == nil {
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("tflite: NewInterpreter returned nil")
	}
	if status := interp.AllocateTensors(); status != tflite.OK {
		interp.Delete()
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("tflite: AllocateTensors: %v", status)
	}
	in := interp.GetInputTensor(0)
	if in == nil {
		interp.Delete()
		opts.Delete()
		model.Delete()
		return nil, fmt.Errorf("tflite: model has no input tensor")
	}
	c := &Classifier{model: model, opts: opts, interp: interp, in: in}
	if d := in.Dim(in.NumDims() - 1); d != InputSamples {
		// v2.4 FP32 input is shape [1, 144000] float32 (last dim = samples).
		c.Close()
		return nil, fmt.Errorf("tflite: unexpected input shape (last dim=%d, want %d); wrong model file?",
			d, InputSamples)
	}
	out := interp.GetOutputTensor(0)
	if out == nil {
		c.Close()
		return nil, fmt.Errorf("tflite: model has no output tensor")
	}
	c.nOut = out.Dim(out.NumDims() - 1)

	if labelsPath != "" {
		f, err := os.Open(labelsPath)
		if err != nil {
			c.Close()
			return nil, fmt.Errorf("labels: %w", err)
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line != "" {
				c.labels = append(c.labels, line)
			}
		}
		if err := sc.Err(); err != nil {
			c.Close()
			return nil, fmt.Errorf("labels: %w", err)
		}
	}
	return c, nil
}

// NumLabels returns the number of loaded labels (0 if none were provided).
func (c *Classifier) NumLabels() int { return len(c.labels) }

// NumOutputs returns the model output size (number of classes).
func (c *Classifier) NumOutputs() int { return c.nOut }

// MeanInferenceSec returns the mean Invoke() time across all Classify calls.
func (c *Classifier) MeanInferenceSec() float64 {
	if c.nInfer == 0 {
		return 0
	}
	return float64(c.inferNs) / float64(c.nInfer) / 1e9
}

// Close releases the interpreter, options and model. Safe to call once.
func (c *Classifier) Close() {
	if c.interp != nil {
		c.interp.Delete()
		c.interp = nil
	}
	if c.opts != nil {
		c.opts.Delete()
		c.opts = nil
	}
	if c.model != nil {
		c.model.Delete()
		c.model = nil
	}
}

// Classify runs inference on one window of InputSamples float32 samples and
// returns the top-n predictions sorted by descending confidence. The input
// slice is copied into the tensor, so the caller may reuse its buffer.
func (c *Classifier) Classify(pcm []float32, topN int) ([]Result, time.Duration, error) {
	if len(pcm) != InputSamples {
		return nil, 0, fmt.Errorf("classify: got %d samples, want %d", len(pcm), InputSamples)
	}
	if topN <= 0 || topN > c.nOut {
		topN = c.nOut
	}
	if status := c.in.CopyFromBuffer(pcm); status != tflite.OK {
		return nil, 0, fmt.Errorf("tflite: CopyFromBuffer: %v", status)
	}
	t0 := time.Now()
	if status := c.interp.Invoke(); status != tflite.OK {
		return nil, 0, fmt.Errorf("tflite: Invoke: %v", status)
	}
	d := time.Since(t0)
	c.inferNs += int64(d)
	c.nInfer++

	out := c.interp.GetOutputTensor(0)
	raw := make([]float32, c.nOut)
	if status := out.CopyToBuffer(raw); status != tflite.OK {
		return nil, 0, fmt.Errorf("tflite: CopyToBuffer: %v", status)
	}

	// softmax
	maxv := raw[0]
	for _, v := range raw {
		if v > maxv {
			maxv = v
		}
	}
	probs := make([]float32, c.nOut)
	var sum float64
	for i, v := range raw {
		e := math.Exp(float64(v - maxv))
		sum += e
		probs[i] = float32(e)
	}
	for i := range probs {
		probs[i] = float32(float64(probs[i]) / sum)
	}

	idx := make([]int, c.nOut)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	results := make([]Result, 0, topN)
	for _, i := range idx[:topN] {
		r := Result{Index: i, Confidence: probs[i]}
		if i < len(c.labels) {
			r.ScientificName, r.CommonName = SplitLabel(c.labels[i])
		}
		results = append(results, r)
	}
	return results, d, nil
}

// SplitLabel splits a BirdNET label line "Scientific name_Common name" on the
// first underscore. Labels without an underscore return the whole string as
// the scientific name (upstream treats that as a non-species sound class).
func SplitLabel(label string) (scientific, common string) {
	before, after, found := strings.Cut(label, "_")
	if !found {
		return label, ""
	}
	return before, after
}
