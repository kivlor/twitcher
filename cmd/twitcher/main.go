// twitcher — headless BirdNET bird detection for the Raspberry Pi 3B.
//
// Live mode (default): ALSA capture → 3 s chunker with overlap → BirdNET v2.4
// inference → detections above the confidence threshold are logged, persisted
// to a SQLite database (BirdNET-Go-compatible notes/results schema), and
// optionally emitted as JSON events to MQTT and saved as audio clips.
//
// Offline mode (-wav): classify a 48 kHz mono WAV file through the same
// pipeline and persistence path (M1 behaviour, retained for verification and
// hardware-free testing).
//
// Configuration: built-in defaults < YAML config file (-config) < flags.
// Environment overrides: TWITCHER_DB, TWITCHER_DEVICE, TWITCHER_MQTT_BROKER.
//
// Usage:
//
//	twitcher [-config twitcher.yaml] [-model model.tflite] [-labels labels.txt]
//	    [-db twitcher.db] [-device default] [-threshold 0.80] [-overlap 0.33]
//	    [-duration 0] [-clips] [-wav file.wav] [-list-devices]
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/gen2brain/malgo"

	"github.com/kivlor/twitcher/internal/audio"
	"github.com/kivlor/twitcher/internal/birdnet"
	"github.com/kivlor/twitcher/internal/clips"
	"github.com/kivlor/twitcher/internal/config"
	"github.com/kivlor/twitcher/internal/events"
	"github.com/kivlor/twitcher/internal/labels/nonbird"
	"github.com/kivlor/twitcher/internal/store"
)

// Date and time layouts for the notes table, matching BirdNET-Go's
// internal/datastore/mapper (DateFormat / TimeFormat).
const (
	dateFormat = "2006-01-02"
	timeFormat = "15:04:05"
)

// retentionSweepInterval is how often clip/note retention runs.
const retentionSweepInterval = time.Hour

func main() {
	var (
		configPath  = flag.String("config", "", "YAML config file (optional; flags override)")
		modelPath   = flag.String("model", "", "BirdNET v2.4 TFLite model path")
		labelsPath  = flag.String("labels", "", "labels file path")
		dbPath      = flag.String("db", "", "SQLite database path (\"\" disables persistence)")
		deviceName  = flag.String("device", "", "ALSA capture device (name or ID substring)")
		threshold   = flag.Float64("threshold", 0, "minimum confidence to record a detection")
		overlap     = flag.Float64("overlap", 0, "chunk overlap fraction [0,1)")
		duration    = flag.Duration("duration", 0, "stop after this long (0 = run until signal)")
		threads     = flag.Int("threads", 0, "TFLite inference threads")
		latitude    = flag.Float64("lat", 0, "latitude recorded on detections")
		longitude   = flag.Float64("lon", 0, "longitude recorded on detections")
		sensitivity = flag.Float64("sensitivity", 0, "BirdNET sensitivity recorded on detections")
		sourceNode  = flag.String("node", "", "source node name recorded on detections")
		clipsOn     = flag.Bool("clips", false, "enable detection clip recording")
		clipsDir    = flag.String("clips-dir", "", "clip storage directory")
		retention   = flag.Int("retention-days", 0, "delete clips (and their notes) older than N days; 0 = keep forever")
		mqttBroker  = flag.String("mqtt-broker", "", "MQTT broker URL, e.g. tcp://host:1883 (\"\" disables)")
		verbose     = flag.Bool("v", false, "log every chunk's top-3, even below threshold")
		listDevices = flag.Bool("list-devices", false, "list capture devices and exit")
		wavPath     = flag.String("wav", "", "offline mode: classify this WAV file and exit")
	)
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	// Explicit flags override the config file. Track which were set.
	set := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
	if set["model"] {
		cfg.Model = *modelPath
	}
	if set["labels"] {
		cfg.Labels = *labelsPath
	}
	if set["db"] {
		cfg.DB = *dbPath
	}
	if set["device"] {
		cfg.Device = *deviceName
	}
	if set["threshold"] {
		cfg.Threshold = *threshold
	}
	if set["overlap"] {
		cfg.Overlap = *overlap
	}
	if set["threads"] {
		cfg.Threads = *threads
	}
	if set["lat"] {
		cfg.Latitude = *latitude
	}
	if set["lon"] {
		cfg.Longitude = *longitude
	}
	if set["sensitivity"] {
		cfg.Sensitivity = *sensitivity
	}
	if set["node"] {
		cfg.Node = *sourceNode
	}
	if set["clips"] {
		cfg.Clips.Enabled = *clipsOn
	}
	if set["clips-dir"] {
		cfg.Clips.Dir = *clipsDir
	}
	if set["retention-days"] {
		cfg.Clips.RetentionDays = *retention
	}
	if set["mqtt-broker"] {
		cfg.MQTT.Broker = *mqttBroker
	}
	if cfg.Node == "" {
		cfg.Node = hostname()
	}

	if cfg.Overlap < 0 || cfg.Overlap >= 1 {
		log.Fatalf("overlap must be in [0, 1), got %v", cfg.Overlap)
	}
	if cfg.Clips.Enabled && cfg.Clips.RetentionDays > 0 && cfg.DB == "" {
		log.Fatalf("retention-days requires -db: pruned clips must also prune their notes")
	}

	if *listDevices {
		listCaptureDevices()
		return
	}

	// --- classifier -------------------------------------------------------
	cl, err := birdnet.Load(cfg.Model, cfg.Labels, cfg.Threads)
	if err != nil {
		log.Fatalf("load classifier: %v", err)
	}
	defer cl.Close()
	log.Printf("classifier ready: %d outputs, %d labels", cl.NumOutputs(), cl.NumLabels())

	// --- store -----------------------------------------------------------
	var st *store.Store
	if cfg.DB != "" {
		st, err = store.Open(cfg.DB)
		if err != nil {
			log.Fatalf("open database: %v", err)
		}
		defer st.Close()
		n, _ := st.Count()
		log.Printf("database ready: %s (%d existing notes)", cfg.DB, n)
	}

	// --- clip recorder ----------------------------------------------------
	var recorder *clips.Recorder
	if cfg.Clips.Enabled {
		recorder, err = clips.NewRecorder(cfg.Clips.Dir)
		if err != nil {
			log.Fatalf("clip recorder: %v", err)
		}
		log.Printf("clip recording enabled: %s (retention %d days)", cfg.Clips.Dir, cfg.Clips.RetentionDays)
	}

	// --- event emitter ----------------------------------------------------
	var emitter *events.MQTTEmitter
	if cfg.MQTT.Broker != "" {
		emitter, err = events.NewMQTTEmitter(cfg.MQTT)
		if err != nil {
			log.Fatalf("mqtt emitter: %v", err)
		}
		defer emitter.Close()
		log.Printf("mqtt emitter ready: %s topic=%s", cfg.MQTT.Broker, cfg.MQTT.Topic)
	}

	app := &app{
		cl: cl, st: st, recorder: recorder, emitter: emitter,
		cfg: cfg, verbose: *verbose,
		sweepStop: make(chan struct{}),
	}

	if *wavPath != "" {
		runOffline(app, *wavPath, cfg.Overlap)
		return
	}

	// --- retention sweep (hourly, best-effort) ---------------------------
	sweepDone := make(chan struct{})
	go func() {
		defer close(sweepDone)
		app.sweep() // once at startup
		t := time.NewTicker(retentionSweepInterval)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				app.sweep()
			case <-app.sweepStop:
				return
			}
		}
	}()
	defer func() {
		close(app.sweepStop)
		<-sweepDone
	}()

	runLive(app, cfg.Device, cfg.Overlap, *duration)
}

// app carries the shared pipeline state used by both live and offline modes.
type app struct {
	cl       *birdnet.Classifier
	st       *store.Store
	recorder *clips.Recorder
	emitter  *events.MQTTEmitter
	cfg      config.Config
	verbose  bool

	sweepStop chan struct{}

	detections int
}

// sweep prunes old clips and (with clips) their orphaned notes, plus notes
// older than NotesRetentionDays (§F4/§N3).
func (a *app) sweep() {
	if a.recorder != nil && a.cfg.Clips.RetentionDays > 0 {
		removed, err := a.recorder.Sweep(a.cfg.Clips.RetentionDays)
		if err != nil {
			log.Printf("clip sweep: %v", err)
		} else if removed > 0 {
			log.Printf("clip sweep: removed %d files older than %d days", removed, a.cfg.Clips.RetentionDays)
		}
		if a.st != nil {
			// Notes whose clip file no longer exist are deleted so the DB
			// stays bounded alongside the clip store.
			n, err := a.st.DeleteWithMissingClip(func(clipName string) bool {
				_, err := os.Stat(filepath.Join(a.cfg.Clips.Dir, clipName))
				return err == nil
			})
			if err != nil {
				log.Printf("note sweep: %v", err)
			} else if n > 0 {
				log.Printf("note sweep: removed %d notes with pruned clips", n)
			}
		}
	}
	if a.st != nil && a.cfg.NotesRetentionDays > 0 {
		cutoff := time.Now().AddDate(0, 0, -a.cfg.NotesRetentionDays)
		n, err := a.st.DeleteOlderThan(cutoff)
		if err != nil {
			log.Printf("note retention: %v", err)
		} else if n > 0 {
			log.Printf("note retention: removed %d notes older than %d days", n, a.cfg.NotesRetentionDays)
		}
	}
}

// process classifies one 3 s chunk and, if the top prediction clears the
// threshold (and is not a non-bird class), logs the detection, persists it
// with its top-3 predictions, saves an optional clip, and emits the event.
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
	if top.Confidence < float32(a.cfg.Threshold) {
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

	if a.st == nil && a.recorder == nil && a.emitter == nil {
		return
	}

	ts := time.Now()
	note := &store.Note{
		SourceNode:     a.cfg.Node,
		Date:           ts.Format(dateFormat),
		Time:           ts.Format(timeFormat),
		BeginTime:      ch.Begin,
		EndTime:        ch.Begin.Add(3 * time.Second),
		SpeciesCode:    speciesCode(top),
		ScientificName: top.ScientificName,
		CommonName:     top.CommonName,
		Confidence:     float64(top.Confidence),
		Latitude:       a.cfg.Latitude,
		Longitude:      a.cfg.Longitude,
		Threshold:      a.cfg.Threshold,
		Sensitivity:    a.cfg.Sensitivity,
		ProcessingTime: inferDur,
	}
	var top3 []*store.Result
	for _, r := range results {
		top3 = append(top3, &store.Result{Species: r.ScientificName, Confidence: r.Confidence})
	}

	// Clip first: its name goes onto the note row (matches upstream, where
	// the clip path is stored in notes.clip_name).
	if a.recorder != nil {
		clipName, err := a.recorder.Save(ch.Begin, ch.PCM, top.CommonName, struct {
			ScientificName string  `json:"scientific_name"`
			CommonName     string  `json:"common_name"`
			Confidence     float64 `json:"confidence"`
		}{top.ScientificName, top.CommonName, float64(top.Confidence)})
		if err != nil {
			log.Printf("clip save failed: %v", err)
		} else {
			note.ClipName = clipName
		}
	}

	if a.st != nil {
		if err := a.st.Save(note, top3); err != nil {
			log.Printf("persist detection failed: %v", err)
		}
	}
	if a.emitter != nil {
		a.emitter.Emit(events.DetectionFromNote(note, top3))
	}
}

// runLive is the M2 pipeline: capture callback → frames channel → chunker →
// chunks queue → classifier. All queues are bounded; under backlog we drop
// the oldest so the pipeline never falls behind real time (N2 defensive
// design). Capture is supervised: a device that fails to open (or dies at
// startup) is retried with exponential backoff (F1).
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

	// Capture supervisor: (re)open the device with backoff until it is
	// running or shutdown is requested.
	supervisorDone := make(chan struct{})
	supCtx, supCancel := context.WithCancel(context.Background())
	go func() {
		defer close(supervisorDone)
		backoff := time.Second
		for {
			stopCapture, err := startCapture(deviceName, frames, &droppedFrames)
			if err == nil {
				// Device is live; hold the stop func until shutdown.
				<-supCtx.Done()
				stopCapture()
				return
			}
			log.Printf("capture init failed (%v); retrying in %.0fs", err, backoff.Seconds())
			select {
			case <-time.After(backoff):
			case <-supCtx.Done():
				return
			}
			backoff *= 2
			if backoff > 60*time.Second {
				backoff = 60 * time.Second
			}
		}
	}()

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
	supCancel() // stops the supervisor (device stop or retry loop exit)
	<-supervisorDone
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
