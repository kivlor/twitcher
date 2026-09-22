// twitcher — headless BirdNET bird detection for the Raspberry Pi 3B.
//
// Live mode (default): ALSA capture → 3 s chunker with overlap → BirdNET v2.4
// inference → detections above the confidence threshold are logged and
// persisted to a SQLite database (BirdNET-Go-compatible notes/results schema).
//
// Offline mode (-wav): classify a 48 kHz mono WAV file through the same
// pipeline and persistence path (M1 behaviour, retained for verification and
// hardware-free testing).
//
// Usage:
//
//	twitcher [-model model.tflite] [-labels labels.txt] [-db twitcher.db]
//	    [-device default] [-threshold 0.80] [-overlap 0.33] [-duration 0]
//	    [-lat 0] [-lon 0] [-list-devices] [-wav file.wav]
//
// -duration 0 runs until SIGINT/SIGTERM. -db "" disables persistence.
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
	"github.com/kivlor/twitcher/internal/store"
)

// Date and time layouts for the notes table, matching BirdNET-Go's
// internal/datastore/mapper (DateFormat / TimeFormat).
const (
	dateFormat = "2006-01-02"
	timeFormat = "15:04:05"
)

func main() {
	var (
		modelPath   = flag.String("model", "model.tflite", "BirdNET v2.4 TFLite model path")
		labelsPath  = flag.String("labels", "labels.txt", "labels file path")
		dbPath      = flag.String("db", "twitcher.db", "SQLite database path (\"\" disables persistence)")
		deviceName  = flag.String("device", "default", "ALSA capture device (name or ID substring)")
		threshold   = flag.Float64("threshold", 0.80, "minimum confidence to record a detection")
		overlap     = flag.Float64("overlap", 0.33, "chunk overlap fraction [0,1)")
		duration    = flag.Duration("duration", 0, "stop after this long (0 = run until signal)")
		threads     = flag.Int("threads", 2, "TFLite inference threads")
		latitude    = flag.Float64("lat", 0, "latitude recorded on detections")
		longitude   = flag.Float64("lon", 0, "longitude recorded on detections")
		sensitivity = flag.Float64("sensitivity", 1.0, "BirdNET sensitivity recorded on detections")
		sourceNode  = flag.String("node", hostname(), "source node name recorded on detections")
		verbose     = flag.Bool("v", false, "log every chunk's top-3, even below threshold")
		listDevices = flag.Bool("list-devices", false, "list capture devices and exit")
		wavPath     = flag.String("wav", "", "offline mode: classify this WAV file and exit")
	)
	flag.Parse()

	if *overlap < 0 || *overlap >= 1 {
		log.Fatalf("overlap must be in [0, 1), got %v", *overlap)
	}

	if *listDevices {
		listCaptureDevices()
		return
	}

	// --- classifier -------------------------------------------------------
	cl, err := birdnet.Load(*modelPath, *labelsPath, *threads)
	if err != nil {
		log.Fatalf("load classifier: %v", err)
	}
	defer cl.Close()
	log.Printf("classifier ready: %d outputs, %d labels", cl.NumOutputs(), cl.NumLabels())

	// --- store -----------------------------------------------------------
	var st *store.Store
	if *dbPath != "" {
		st, err = store.Open(*dbPath)
		if err != nil {
			log.Fatalf("open database: %v", err)
		}
		defer st.Close()
		n, _ := st.Count()
		log.Printf("database ready: %s (%d existing notes)", *dbPath, n)
	}

	app := &app{
		cl: cl, st: st,
		threshold: *threshold, latitude: *latitude, longitude: *longitude,
		sensitivity: *sensitivity, sourceNode: *sourceNode, verbose: *verbose,
	}

	if *wavPath != "" {
		runOffline(app, *wavPath, *overlap)
		return
	}

	runLive(app, *deviceName, *overlap, *duration)
}

// app carries the shared pipeline state used by both live and offline modes.
type app struct {
	cl *birdnet.Classifier
	st *store.Store

	threshold   float64
	latitude    float64
	longitude   float64
	sensitivity float64
	sourceNode  string
	verbose     bool

	detections int
}

// process classifies one 3 s chunk and, if the top prediction clears the
// threshold (and is not a non-bird class), logs and persists the detection
// with its top-3 predictions.
func (a *app) process(ch audio.Chunk) {
	results, inferDur, err := a.cl.Classify(ch.PCM, 3)
	if err != nil {
		log.Printf("classify error: %v", err)
		return
	}
	if len(results) == 0 {
		return
	}
	if a.verbose {
		log.Printf("chunk %s infer=%.3fs top3: %s",
			ch.Begin.Format(timeFormat), inferDur.Seconds(), formatTop(results))
	}
	top := results[0]
	if top.Confidence < float32(a.threshold) {
		return
	}
	if nonbird.IsNonSpeciesLabel(rawLabel(top)) {
		log.Printf("[%s] filtered non-bird label: %q %.3f",
			ch.Begin.Format("15:04:05.000"), top.ScientificName, top.Confidence)
		return
	}
	a.detections++
	log.Printf("DETECTION [%s] %s (%s) confidence=%.3f infer=%.3fs",
		ch.Begin.Format("15:04:05.000"), top.CommonName, top.ScientificName, top.Confidence,
		inferDur.Seconds())

	if a.st == nil {
		return
	}
	ts := time.Now()
	note := &store.Note{
		SourceNode:     a.sourceNode,
		Date:           ts.Format(dateFormat),
		Time:           ts.Format(timeFormat),
		BeginTime:      ch.Begin,
		EndTime:        ch.Begin.Add(3 * time.Second),
		SpeciesCode:    speciesCode(top),
		ScientificName: top.ScientificName,
		CommonName:     top.CommonName,
		Confidence:     float64(top.Confidence),
		Latitude:       a.latitude,
		Longitude:      a.longitude,
		Threshold:      a.threshold,
		Sensitivity:    a.sensitivity,
		ProcessingTime: inferDur,
	}
	var top3 []*store.Result
	for _, r := range results {
		top3 = append(top3, &store.Result{Species: r.ScientificName, Confidence: r.Confidence})
	}
	if err := a.st.Save(note, top3); err != nil {
		log.Printf("persist detection failed: %v", err)
	}
}

// runLive is the M2 pipeline: capture callback → frames channel → chunker →
// chunks queue → classifier. Both queues are bounded; under backlog we drop
// the oldest so the pipeline never falls behind real time (N2 defensive
// design).
func runLive(a *app, deviceName string, overlap float64, duration time.Duration) {
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
			a.process(ch)
		}
	}()

	chunkerDone := make(chan struct{})
	go func() {
		defer close(chunkerDone)
		chunker := audio.NewChunker(overlap)
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

	stopCapture, err := startCapture(deviceName, frames, &droppedFrames)
	if err != nil {
		log.Fatalf("capture: %v", err)
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	var timeout <-chan time.Time
	if duration > 0 {
		timeout = time.After(duration)
	}

	started := time.Now()
	var stopReason string
	select {
	case s := <-sig:
		stopReason = fmt.Sprintf("signal %v", s)
	case <-timeout:
		stopReason = fmt.Sprintf("duration %s elapsed", duration)
	}

	log.Printf("stopping (%s)…", stopReason)
	stopCapture()
	close(frames)
	<-chunkerDone
	<-chunksDone

	log.Printf("ran %.1fs | mean inference %.3fs/chunk | detections: %d | chunks dropped: %d | frames dropped: %d",
		time.Since(started).Seconds(), a.cl.MeanInferenceSec(), a.detections, droppedChunks, droppedFrames)
}

// runOffline classifies a WAV file through the same chunker + process path.
func runOffline(a *app, wavPath string, overlap float64) {
	pcm, err := audio.ReadWavMono48k(wavPath)
	if err != nil {
		log.Fatalf("read wav: %v", err)
	}
	log.Printf("wav: %s — %d samples (%.1f s)", wavPath, len(pcm), float64(len(pcm))/birdnet.SampleRate)

	started := time.Now()
	chunker := audio.NewChunker(overlap)
	// Feed in 0.5 s frames; the chunker assembles the 3 s windows.
	const frameSamples = birdnet.SampleRate / 2
	for off := 0; off < len(pcm); off += frameSamples {
		end := off + frameSamples
		if end > len(pcm) {
			end = len(pcm)
		}
		frame := audio.Frame{Samples: pcm[off:end], Begin: started.Add(time.Duration(off) * time.Second / birdnet.SampleRate)}
		chunker.Push(frame, a.process)
	}
	log.Printf("done in %.1fs | mean inference %.3fs/chunk | detections: %d",
		time.Since(started).Seconds(), a.cl.MeanInferenceSec(), a.detections)
}

// rawLabel reconstructs the raw label form ("Scientific_Common") so the
// nonbird class map — which keys on full raw labels — can match it.
func rawLabel(r birdnet.Result) string {
	if r.CommonName == "" {
		return r.ScientificName
	}
	return r.ScientificName + "_" + r.CommonName
}

// speciesCode extracts the eBird species code from a 3-part label
// ("Scientific_Common_Code"). Two-part labels have no code; upstream leaves
// the field empty in that case too.
func speciesCode(r birdnet.Result) string {
	if i := strings.LastIndex(r.CommonName, "_"); i >= 0 {
		return r.CommonName[i+1:]
	}
	return ""
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

func hostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "twitcher"
	}
	return h
}

// startCapture initialises malgo with the ALSA backend and starts capturing
// 48 kHz mono. It returns a stop function that uninitialises device and
// context. Frames are converted to float32 mono and pushed onto out
// (non-blocking, drop when full — *droppedFrames counts those).
var debugFrames = os.Getenv("TWITCHER_DEBUG_FRAMES") != ""

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
