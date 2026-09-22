// M0 spike: verify tphakala/go-tflite compiles and links for linux/arm (GOARM=7).
// M1 spike: load BirdNET v2.4 TFLite model, classify a WAV, print top-3 per chunk.
//
// Usage: m1-spike <model.tflite> <audio.wav> [labels.txt]
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"time"
	"unsafe"

	tflite "github.com/tphakala/go-tflite"
)

const inputSamples = 144000 // 3 s @ 48 kHz, BirdNET v2.4 input window

type topHit struct {
	idx   int
	prob  float32
	label string
}

func main() {
	if len(os.Args) < 3 {
		fmt.Println("usage: m1-spike <model.tflite> <audio.wav> [labels.txt]")
		os.Exit(1)
	}
	modelPath, wavPath := os.Args[1], os.Args[2]
	var labels []string
	if len(os.Args) > 3 {
		f, err := os.Open(os.Args[3])
		if err != nil {
			fmt.Println("WARN: labels:", err)
		} else {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				labels = append(labels, sc.Text())
			}
			f.Close()
			fmt.Printf("loaded %d labels\n", len(labels))
		}
	}

	pcm, err := readWavMono48k(wavPath)
	if err != nil {
		fmt.Println("ERROR reading wav:", err)
		os.Exit(1)
	}
	fmt.Printf("wav: %d samples (%.1f s)\n", len(pcm), float64(len(pcm))/48000)

	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		fmt.Println("ERROR: NewModelFromFile returned nil")
		os.Exit(1)
	}
	defer model.Delete()

	opts := tflite.NewInterpreterOptions()
	if opts == nil {
		fmt.Println("ERROR: NewInterpreterOptions returned nil")
		os.Exit(1)
	}
	defer opts.Delete()
	opts.SetNumThread(2)

	interp := tflite.NewInterpreter(model, opts)
	if interp == nil {
		fmt.Println("ERROR: NewInterpreter returned nil")
		os.Exit(1)
	}
	defer interp.Delete()
	if status := interp.AllocateTensors(); status != tflite.OK {
		fmt.Println("ERROR: AllocateTensors:", status)
		os.Exit(1)
	}

	in := interp.GetInputTensor(0)
	if in == nil {
		fmt.Println("ERROR: no input tensor")
		os.Exit(1)
	}
	fmt.Printf("input: type=%d shape", in.Type())
	for d := 0; d < in.NumDims(); d++ {
		fmt.Printf(" %d", in.Dim(d))
	}
	fmt.Println()

	buf := make([]float32, inputSamples)
	var totalInfer time.Duration
	nChunks := 0

	for off := 0; off+inputSamples <= len(pcm); off += inputSamples {
		copy(buf, pcm[off:off+inputSamples])
		if status := in.CopyFromBuffer(buf); status != tflite.OK {
			fmt.Println("ERROR: CopyFromBuffer:", status)
			os.Exit(1)
		}
		t0 := time.Now()
		if status := interp.Invoke(); status != tflite.OK {
			fmt.Println("ERROR: Invoke:", status)
			os.Exit(1)
		}
		d := time.Since(t0)
		totalInfer += d
		nChunks++

		out := interp.GetOutputTensor(0)
		n := out.Dim(out.NumDims() - 1)
		outBuf := make([]float32, n)
		if status := out.CopyToBuffer(outBuf); status != tflite.OK {
			fmt.Println("ERROR: CopyToBuffer:", status)
			os.Exit(1)
		}

		// softmax + top-3
		maxv := outBuf[0]
		for _, v := range outBuf {
			if v > maxv {
				maxv = v
			}
		}
		hits := make([]topHit, n)
		var sum float64
		for i, v := range outBuf {
			e := math.Exp(float64(v - maxv))
			sum += e
			hits[i] = topHit{idx: i, prob: float32(e)}
		}
		for i := range hits {
			hits[i].prob = float32(float64(hits[i].prob) / sum)
			if hits[i].idx < len(labels) {
				hits[i].label = labels[hits[i].idx]
			}
		}
		sort.Slice(hits, func(a, b int) bool { return hits[a].prob > hits[b].prob })

		ts := float64(off) / 48000.0
		fmt.Printf("[%.0fs] %.3fs infer | ", ts, d.Seconds())
		for i := 0; i < 3; i++ {
			l := hits[i].label
			if l == "" {
				l = fmt.Sprintf("#%d", hits[i].idx)
			}
			fmt.Printf("%s %.3f", l, hits[i].prob)
			if i < 2 {
				fmt.Print(", ")
			}
		}
		fmt.Println()
	}

	if nChunks > 0 {
		fmt.Printf("\nmean inference: %.3fs/chunk (window is 3.000s of audio)\n",
			totalInfer.Seconds()/float64(nChunks))
	}
}

// readWavMono48k reads a WAV file and returns mono 48 kHz samples as float32 in [-1,1].
func readWavMono48k(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a RIFF/WAVE file")
	}
	var fmtChunk, dataChunk []byte
	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := data[pos+8 : pos+8+size]
		switch id {
		case "fmt ":
			fmtChunk = body
		case "data":
			dataChunk = body
		}
		pos += 8 + size
		if size%2 == 1 {
			pos++ // chunks are word-aligned
		}
	}
	if fmtChunk == nil || dataChunk == nil {
		return nil, fmt.Errorf("missing fmt/data chunk")
	}
	audioFmt := binary.LittleEndian.Uint16(fmtChunk[0:2])
	if audioFmt == 0xFFFE && len(fmtChunk) >= 26 { // WAVE_FORMAT_EXTENSIBLE
		audioFmt = binary.LittleEndian.Uint16(fmtChunk[24:26]) // subformat GUID first 2 bytes
	}
	channels := int(binary.LittleEndian.Uint16(fmtChunk[2:4]))
	rate := int(binary.LittleEndian.Uint32(fmtChunk[4:8]))
	bits := int(binary.LittleEndian.Uint16(fmtChunk[14:16]))
	if channels != 1 || rate != 48000 {
		return nil, fmt.Errorf("unsupported wav: ch=%d rate=%d (want mono 48 kHz)", channels, rate)
	}
	switch {
	case audioFmt == 1 && bits == 16: // PCM16
		n := len(dataChunk) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(int16(binary.LittleEndian.Uint16(dataChunk[i*2:i*2+2]))) / 32768.0
		}
		return out, nil
	case audioFmt == 1 && bits == 32: // PCM32
		n := len(dataChunk) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(int32(binary.LittleEndian.Uint32(dataChunk[i*4:i*4+4]))) / 2147483648.0
		}
		return out, nil
	case audioFmt == 3 && bits == 32: // IEEE float32
		n := len(dataChunk) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32FromBits(binary.LittleEndian.Uint32(dataChunk[i*4 : i*4+4]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported wav: fmt=%d bits=%d (want PCM16/32 or float32)", audioFmt, bits)
	}
}

func float32FromBits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
