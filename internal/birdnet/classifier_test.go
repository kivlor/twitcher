package birdnet

import (
	"os"
	"testing"

	"github.com/kivlor/twitcher/internal/audio"
)

// Model/labels fixtures live in build/amd64 (dev host) or build/pi (Pi);
// the test is skipped when neither is present so `go test` stays hermetic
// on machines without the multi-MB artifacts.
func fixtures(t *testing.T) (modelPath, labelsPath, wavPath string, ok bool) {
	t.Helper()
	for _, dir := range []string{"../../build/amd64", "../../build/pi", "build/amd64", "build/pi"} {
		m := dir + "/model.tflite"
		l := dir + "/labels.txt"
		if _, err := os.Stat(m); err == nil {
			if _, err := os.Stat(l); err == nil {
				w := dir + "/tawnyowl.wav"
				_, wavErr := os.Stat(w)
				return m, l, w, wavErr == nil
			}
		}
	}
	return "", "", "", false
}

// TestClassifierLoads checks model+labels load and shapes match v2.4.
func TestClassifierLoads(t *testing.T) {
	modelPath, labelsPath, _, ok := fixtures(t)
	if !ok {
		t.Skip("model/labels fixtures not present")
	}
	cl, err := Load(modelPath, labelsPath, 2)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer cl.Close()
	if cl.NumOutputs() < 6000 {
		t.Errorf("NumOutputs=%d, want ~6522 (v2.4 6K model)", cl.NumOutputs())
	}
	if cl.NumLabels() != cl.NumOutputs() {
		t.Errorf("NumLabels=%d != NumOutputs=%d", cl.NumLabels(), cl.NumOutputs())
	}
}

// TestClassifyTawnyOwl runs the known-good fixture: the wav must yield
// Strix aluco (Tawny Owl) as a top-1 hit with high confidence on at least
// one 3-s window — matching the M1 spike results on Pi and qemu.
func TestClassifyTawnyOwl(t *testing.T) {
	modelPath, labelsPath, wavPath, ok := fixtures(t)
	if !ok {
		t.Skip("model/labels/wav fixtures not present")
	}
	cl, err := Load(modelPath, labelsPath, 2)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer cl.Close()

	pcm, err := audio.ReadWavMono48k(wavPath)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	best := Result{}
	for start := 0; start+InputSamples <= len(pcm); start += InputSamples {
		results, _, err := cl.Classify(pcm[start:start+InputSamples], 3)
		if err != nil {
			t.Fatalf("Classify: %v", err)
		}
		if results[0].Confidence > best.Confidence {
			best = results[0]
		}
	}
	if best.ScientificName != "Strix aluco" {
		t.Errorf("best = %s (%s) conf=%.3f, want Strix aluco (Tawny Owl)",
			best.ScientificName, best.CommonName, best.Confidence)
	}
	if best.Confidence < 0.9 {
		t.Errorf("best confidence %.3f < 0.9", best.Confidence)
	}
}
