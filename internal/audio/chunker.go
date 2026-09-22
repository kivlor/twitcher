// Package audio holds the capture-side audio utilities: the overlapping
// window chunker that turns a live 48 kHz mono stream into BirdNET-sized
// 3-second windows.
package audio

import "time"

// Frame is one callback delivery of mono 48 kHz float32 samples from the
// capture device. Begin is the wall-clock time of the first sample.
type Frame struct {
	Samples []float32
	Begin   time.Time
}

// Chunk is one 3-second window ready for BirdNET inference. PCM is exactly
// birdnet.InputSamples samples; Begin is the wall-clock time the window's
// first sample was captured.
type Chunk struct {
	PCM   []float32
	Begin time.Time
}

// BirdNET input window: 3 s @ 48 kHz mono float32.
const (
	windowSamples = 144000
	sampleRate    = 48000
)

// Chunker converts a stream of Frames into overlapping Chunks. It is not
// safe for concurrent use; the M2 pipeline runs it on a single goroutine.
type Chunker struct {
	win, step    int
	buf          []float32
	windowStart  time.Time
	haveStart    bool
	sampleOffset int // samples appended since last emitted window's start
}

// NewChunker returns a Chunker emitting windowSamples-sample (3 s @ 48 kHz)
// windows with the given overlap fraction in [0, 1): consecutive windows start
// (overlap × 3 s) apart. 0.33 ≈ 1 s hop on the Pi 3 (EXTRACTION_PLAN §S1).
func NewChunker(overlap float64) *Chunker {
	if overlap < 0 || overlap >= 1 {
		panic("overlap must be in [0, 1)")
	}
	win := windowSamples
	step := win - int(overlap*float64(win))
	if step < 1 {
		step = 1
	}
	return &Chunker{win: win, step: step, buf: make([]float32, 0, win+step)}
}

// Push appends a Frame and invokes emit once for every complete window.
// The chunker never blocks: emit is called synchronously and must either
// enqueue (bounded queue, drop-oldest) or discard.
func (c *Chunker) Push(f Frame, emit func(Chunk)) {
	if len(f.Samples) == 0 {
		return
	}
	if !c.haveStart {
		c.windowStart = f.Begin
		c.haveStart = true
	}
	c.buf = append(c.buf, f.Samples...)
	for len(c.buf) >= c.win {
		window := make([]float32, c.win)
		copy(window, c.buf[:c.win])
		emit(Chunk{PCM: window, Begin: c.windowStart})
		// Slide: drop step samples. Re-slice in place, with occasional
		// reallocation so append() cannot grow capacity without bound.
		buf := c.buf
		n := copy(buf, buf[c.step:])
		c.buf = buf[:n]
		if cap(c.buf) > 4*c.win {
			nb := make([]float32, len(c.buf), c.win+c.step)
			copy(nb, c.buf)
			c.buf = nb
		}
		c.windowStart = c.windowStart.Add(sampleDur(c.step))
	}
}

func sampleDur(n int) time.Duration {
	return time.Duration(n) * time.Second / sampleRate
}
