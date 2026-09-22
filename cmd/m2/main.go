// M2 — Stream + detect: live ALSA capture → 3 s chunker with overlap →
// BirdNET v2.4 inference → log detections above the confidence threshold.
//
// Usage:
//
//	m2 [-model model.tflite] [-labels labels.txt] [-device default]
//	    [-threshold 0.80] [-overlap 0.33] [-duration 0] [-list-devices]
//
// Defaults assume the model/labels sit in the working directory (same layout
// as build/arm64). -duration 0 runs until SIGINT/SIGTERM.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/gen2brain/malgo"

	"github.com/kivlor/twitcher/internal/audio"
	"github.com/kivlor/twitcher/internal/birdnet"
	"github.com/kivlor/twitcher/internal/labels/nonbird"
)

func main() {
	var (
		modelPath   = flag.String("model", "model.tflite", "BirdNET v2.4 TFLite model path")
		labelsPath  = flag.String("labels", "labels.txt", "labels file path")
		deviceName  = flag.String("device", "default", "ALSA capture device (name or ID substring)")
		threshold   = flag.Float64("threshold", 0.80, "minimum confidence to log a detection")
		overlap     = flag.Float64("overlap", 0.33, "chunk overlap fraction [0,1)")
		duration    = flag.Duration("duration", 0, "stop after this long (0 = run until signal)")
		threads     = flag.Int("threads", 2, "TFLite inference threads")
		verbose     = flag.Bool("v", false, "log every chunk's top-3, even below threshold")
		listDevices = flag.Bool("list-devices", false, "list capture devices and exit")
	)
	flag.Parse()

	if *overlap < 0 || *overlap >= 1 {
		log.Fatalf("overlap must be in [0, 1), got %v", *overlap)
	}

	// --- classifier -------------------------------------------------------
	cl, err := birdnet.Load(*modelPath, *labelsPath, *threads)
	if err != nil {
		log.Fatalf("load classifier: %v", err)
	}
	defer cl.Close()
	log.Printf("classifier ready: %d outputs, %d labels", cl.NumOutputs(), cl.NumLabels())

	if *listDevices {
		listCaptureDevices()
		return
	}

	// --- pipeline --------------------------------------------------------
	// capture callback → frames channel → chunker → chunks queue → worker.
	// Both queues are bounded; under backlog we drop oldest so the pipeline
	// never falls behind real time (N2 defensive design).
	frames := make(chan audio.Frame, 64)
	chunks := make(chan audio.Chunk, 8)

	var (
		droppedChunks int64
		droppedFrames int64
	)

	chunksDone := make(chan struct{})
	go func() {
		defer close(chunksDone)
		for ch := range chunks {
			results, inferDur, err := cl.Classify(ch.PCM, 3)
			if err != nil {
				log.Printf("classify error: %v", err)
				continue
			}
			if *verbose {
				log.Printf("chunk %s infer=%.3fs top3: %s",
					ch.Begin.Format("15:04:05"), inferDur.Seconds(), formatTop(results))
			}
			top := results[0]
			if top.Confidence >= float32(*threshold) {
				if nonbird.IsNonSpeciesLabel(rawLabel(top)) {
					log.Printf("[%s] filtered non-bird label: %q %.3f",
						ch.Begin.Format("15:04:05.000"), top.ScientificName, top.Confidence)
				} else {
					log.Printf("DETECTION [%s] %s (%s) confidence=%.3f infer=%.3fs",
						ch.Begin.Format("15:04:05.000"), top.CommonName, top.ScientificName, top.Confidence,
						inferDur.Seconds())
				}
			}
		}
	}()

	chunkerDone := make(chan struct{})
	go func() {
		defer close(chunkerDone)
		chunker := audio.NewChunker(*overlap)
		for f := range frames {
			chunker.Push(f, func(ch audio.Chunk) {
				select {
				case chunks <- ch:
				default:
					// drop-oldest: freshest audio wins (N2 defensive design)
					droppedChunks++
					select {
					case <-chunks:
					default:
					}
					chunks <- ch
				}
			})
		}
		close(chunks)
	}()

	// --- capture ---------------------------------------------------------
	stopCapture, err := startCapture(*deviceName, frames, &droppedFrames)
	if err != nil {
		log.Fatalf("capture: %v", err)
	}

	// --- run until signal or duration ------------------------------------
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	var timeout <-chan time.Time
	if *duration > 0 {
		timeout = time.After(*duration)
	}

	started := time.Now()
	var stopReason string
	select {
	case s := <-sig:
		stopReason = fmt.Sprintf("signal %v", s)
	case <-timeout:
		stopReason = fmt.Sprintf("duration %s elapsed", *duration)
	}

	log.Printf("stopping (%s)…", stopReason)
	stopCapture()
	close(frames)
	<-chunkerDone
	<-chunksDone

	log.Printf("ran %.1fs | mean inference %.3fs/chunk | chunks dropped: %d | frames dropped: %d",
		time.Since(started).Seconds(), cl.MeanInferenceSec(), droppedChunks, droppedFrames)
}

// rawLabel reconstructs the raw label form ("Scientific_Common") so the
// nonbird class map — which keys on full raw labels — can match it.
func rawLabel(r birdnet.Result) string {
	if r.CommonName == "" {
		return r.ScientificName
	}
	return r.ScientificName + "_" + r.CommonName
}

func formatTop(results []birdnet.Result) string {
	var b strings.Builder
	for i, r := range results {
		if i > 0 {
			b.WriteString(", ")
		}
		name := r.ScientificName
		if r.CommonName != "" {
			name = r.CommonName
		}
		fmt.Fprintf(&b, "%s %.3f", name, r.Confidence)
	}
	return b.String()
}

// startCapture initialises malgo with the ALSA backend and starts capturing
// 48 kHz mono. It returns a stop function that uninitialises device and
// context. Frames are converted to float32 mono and pushed onto out
// (non-blocking, drop when full — *droppedFrames counts those).
var debugFrames = os.Getenv("M2_DEBUG_FRAMES") != ""

func startCapture(deviceName string, out chan<- audio.Frame, droppedFrames *int64) (func(), error) {
	ctx, err := malgo.InitContext([]malgo.Backend{malgo.BackendAlsa}, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("malgo init: %w", err)
	}

	// A zero DeviceID means "use the backend default device". When a
	// specific device is requested, devID holds its ID and must outlive the
	// device (it is referenced by the stop closure and by malgo).
	var devID malgo.DeviceID
	var devIDSet bool
	if deviceName != "" && deviceName != "default" && deviceName != "sysdefault" {
		infos, err := ctx.Devices(malgo.Capture)
		if err != nil {
			ctx.Uninit()
			ctx.Free()
			return nil, fmt.Errorf("enumerate devices: %w", err)
		}
		found := false
		for i := range infos {
			if strings.Contains(strings.ToLower(infos[i].Name()), strings.ToLower(deviceName)) ||
				strings.Contains(strings.ToLower(infos[i].ID.String()), strings.ToLower(deviceName)) {
				devID = infos[i].ID
				devIDSet = true
				log.Printf("capture device: %s", infos[i].Name())
				found = true
				break
			}
		}
		if !found {
			ctx.Uninit()
			ctx.Free()
			return nil, fmt.Errorf("no capture device matching %q", deviceName)
		}
	} else {
		log.Printf("capture device: ALSA %q", deviceName)
	}

	var fmtType malgo.FormatType // set after successful init; read by the callback

	onFrames := func(_, pSamples []byte, frameCount uint32) {
		if debugFrames {
			log.Printf("callback: frameCount=%d pSamples=%d fmt=%v", frameCount, len(pSamples), fmtType)
		}
		if len(pSamples) == 0 {
			return
		}
		var samples []float32
		switch fmtType {
		case malgo.FormatF32:
			n := len(pSamples) / 4
			samples = make([]float32, n)
			for i := range samples {
				samples[i] = f32FromBytes(pSamples[i*4:])
			}
		case malgo.FormatS16:
			n := len(pSamples) / 2
			samples = make([]float32, n)
			for i := range samples {
				samples[i] = float32(int16(binary.LittleEndian.Uint16(pSamples[i*2:]))) / 32768.0
			}
		default:
			return // unsupported capture format; drop
		}
		select {
		case out <- audio.Frame{Samples: samples, Begin: time.Now()}:
		default:
			*droppedFrames++
		}
	}

	tryStart := func(format malgo.FormatType) (*malgo.Device, error) {
		cfg := malgo.DefaultDeviceConfig(malgo.Capture)
		if devIDSet {
			cfg.Capture.DeviceID = devID.Pointer()
		}
		cfg.Capture.Format = format
		cfg.Capture.Channels = 1
		cfg.SampleRate = birdnet.SampleRate
		cfg.PeriodSizeInMilliseconds = 20
		dev, err := malgo.InitDevice(ctx.Context, cfg, malgo.DeviceCallbacks{Data: onFrames})
		if err != nil {
			return nil, err
		}
		if err := dev.Start(); err != nil {
			dev.Uninit()
			return nil, err
		}
		return dev, nil
	}

	device, err := tryStart(malgo.FormatF32)
	if err == nil {
		fmtType = malgo.FormatF32
	} else {
		log.Printf("F32 capture init failed (%v); retrying S16", err)
		device, err = tryStart(malgo.FormatS16)
		if err != nil {
			ctx.Uninit()
			ctx.Free()
			return nil, fmt.Errorf("malgo init device: %w", err)
		}
		fmtType = malgo.FormatS16
	}
	log.Printf("capture started: 48000 Hz mono %s", formatName(fmtType))

	stop := func() {
		_ = devID // keep the device ID alive until the device is uninitialized
		device.Uninit()
		ctx.Uninit()
		ctx.Free()
	}
	return stop, nil
}

func f32FromBytes(b []byte) float32 {
	u := binary.LittleEndian.Uint32(b)
	return *(*float32)(unsafe.Pointer(&u))
}

func formatName(f malgo.FormatType) string {
	switch f {
	case malgo.FormatF32:
		return "f32"
	case malgo.FormatS16:
		return "s16"
	default:
		return fmt.Sprintf("format(%d)", f)
	}
}

func listCaptureDevices() {
	ctx, err := malgo.InitContext([]malgo.Backend{malgo.BackendAlsa}, malgo.ContextConfig{}, nil)
	if err != nil {
		log.Fatalf("malgo init: %v", err)
	}
	defer func() { ctx.Uninit(); ctx.Free() }()

	infos, err := ctx.Devices(malgo.Capture)
	if err != nil {
		log.Fatalf("enumerate: %v", err)
	}
	fmt.Println("capture devices:")
	for i := range infos {
		fmt.Printf("  %s (id: %s, default: %d)\n",
			infos[i].Name(), infos[i].ID.String(), infos[i].IsDefault)
	}
}
