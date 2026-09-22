package audio

import (
	"testing"
	"time"
)

// TestChunkerWindowsNoOverlap feeds exactly three windows of audio in small
// frames and checks three chunks come out with correct start times.
func TestChunkerWindowsNoOverlap(t *testing.T) {
	c := NewChunker(0)
	t0 := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)

	const total = 3 * windowSamples
	// frames of 1000 samples (like a 20 ms ALSA callback would deliver 960)
	const frameLen = 1000
	var got []Chunk
	push := func(n int) {
		f := Frame{Samples: make([]float32, n), Begin: t0.Add(time.Duration(n) * time.Second / sampleRate)}
		c.Push(f, func(ch Chunk) { got = append(got, ch) })
	}
	_ = push

	var sent int
	for sent < total {
		n := frameLen
		if total-sent < n {
			n = total - sent
		}
		c.Push(Frame{
			Samples: make([]float32, n),
			Begin:   t0.Add(sampleDur(sent)),
		}, func(ch Chunk) { got = append(got, ch) })
		sent += n
	}

	if len(got) != 3 {
		t.Fatalf("got %d chunks, want 3", len(got))
	}
	for i, ch := range got {
		if len(ch.PCM) != windowSamples {
			t.Fatalf("chunk %d: len=%d, want %d", i, len(ch.PCM), windowSamples)
		}
		want := t0.Add(sampleDur(i * windowSamples))
		if !ch.Begin.Equal(want) {
			t.Errorf("chunk %d: Begin=%s, want %s", i, ch.Begin, want)
		}
	}
}

// TestChunkerOverlapSteps verifies that with overlap 0.5 consecutive emitted
// windows overlap by half and starts advance by the step.
func TestChunkerOverlapSteps(t *testing.T) {
	const overlap = 0.5
	c := NewChunker(overlap)
	win := windowSamples
	step := win - int(overlap*float64(win))
	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	var got []Chunk
	// send 2 windows worth in 2 large frames
	for i := 0; i < 2; i++ {
		c.Push(Frame{Samples: make([]float32, win), Begin: t0.Add(sampleDur(i * win))},
			func(ch Chunk) { got = append(got, ch) })
	}

	wantChunks := 1 + (2*win-win)/step // = 3 for overlap 0.5
	if len(got) != wantChunks {
		t.Fatalf("got %d chunks, want %d", len(got), wantChunks)
	}
	for i := 1; i < len(got); i++ {
		want := t0.Add(sampleDur(i * step))
		if !got[i].Begin.Equal(want) {
			t.Errorf("chunk %d: Begin=%s, want %s (step %d samples)", i, got[i].Begin, want, step)
		}
	}
}

// TestChunkerDataIntegrity checks that windows actually contain the samples
// they claim to, across the overlap boundary.
func TestChunkerDataIntegrity(t *testing.T) {
	c := NewChunker(0.33)
	win := windowSamples
	step := win - int(0.33*float64(win))

	// stream where sample i has value float32(i)
	total := win + step
	data := make([]float32, total)
	for i := range data {
		data[i] = float32(i)
	}
	var got []Chunk
	for s := 0; s < total; s += 480 {
		e := s + 480
		if e > total {
			e = total
		}
		c.Push(Frame{Samples: data[s:e], Begin: time.Now()}, func(ch Chunk) { got = append(got, ch) })
	}
	if len(got) != 2 {
		t.Fatalf("got %d chunks, want 2", len(got))
	}
	for ci, ch := range got {
		for k := 0; k < 10; k++ {
			pos := k * win / 10
			want := float32(ci*step + pos)
			if ch.PCM[pos] != want {
				t.Errorf("chunk %d pos %d: got %v, want %v", ci, pos, ch.PCM[pos], want)
			}
		}
	}
}

func TestNewChunkerRejectsBadOverlap(t *testing.T) {
	for _, o := range []float64{-0.1, 1.0, 1.5} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("overlap %v: expected panic", o)
				}
			}()
			NewChunker(o)
		}()
	}
}
