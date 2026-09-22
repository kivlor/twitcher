// Package clips records detection audio windows as 16-bit PCM WAV files and
// prunes them with a retention sweep (EXTRACTION_PLAN §F4; the sweep borrows
// its idea from BirdNET-Go's diskmanager without pulling in its deps).
package clips

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Dir layout: <root>/YYYY/MM/DD/<timestamp>_<common-name>.wav
// plus a sidecar .json for provenance (species, confidence).

// Recorder writes detection clips under a root directory.
type Recorder struct {
	root string
}

// NewRecorder returns a recorder rooted at dir, creating the directory.
func NewRecorder(dir string) (*Recorder, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("clips: mkdir %s: %w", dir, err)
	}
	return &Recorder{root: dir}, nil
}

// Save writes one 3 s window (48 kHz mono float32 samples) and returns the
// clip path relative to the recorder root — the value stored in
// notes.clip_name.
func (r *Recorder) Save(chunkStart time.Time, samples []float32, commonName string, meta any) (string, error) {
	relDir := chunkStart.Format("2006/01/02")
	dir := filepath.Join(r.root, relDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("clips: mkdir: %w", err)
	}
	safe := strings.Map(func(c rune) rune {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
			return c
		case c == ' ', c == '-', c == '_':
			return '-'
		default:
			return -1
		}
	}, commonName)
	base := fmt.Sprintf("%s_%s", chunkStart.Format("20060102_150405.000"), safe)
	wavPath := filepath.Join(relDir, base+".wav")

	if err := writeWav(filepath.Join(r.root, wavPath), samples); err != nil {
		return "", err
	}
	// Provenance sidecar (species/confidence details) — cheap and useful
	// when browsing clips without the DB.
	if mj, err := json.MarshalIndent(meta, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(r.root, relDir, base+".json"), mj, 0o644)
	}
	return filepath.ToSlash(wavPath), nil
}

// writeWav writes float32 mono samples as a 16-bit PCM WAV at 48 kHz.
func writeWav(path string, samples []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("clips: create %s: %w", path, err)
	}
	defer f.Close()

	const sampleRate = 48000
	const bitsPerSample = 16
	dataLen := len(samples) * 2

	w := func(b []byte) error {
		_, err := f.Write(b)
		return err
	}
	le := func(v uint32) []byte {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		return b[:]
	}

	if err := w([]byte("RIFF")); err != nil {
		return err
	}
	if err := w(le(uint32(36 + dataLen))); err != nil {
		return err
	}
	if err := w([]byte("WAVE")); err != nil {
		return err
	}

	if err := w([]byte("fmt ")); err != nil {
		return err
	}
	if err := w(le(16)); err != nil {
		return err
	} // fmt chunk size
	var b2 [2]byte
	binary.LittleEndian.PutUint16(b2[:], 1) // PCM
	if err := w(b2[:]); err != nil {
		return err
	}
	if err := w(b2[:]); err != nil {
		return err
	} // mono
	if err := w(le(sampleRate)); err != nil {
		return err
	}
	if err := w(le(sampleRate * 2)); err != nil {
		return err
	} // byte rate
	binary.LittleEndian.PutUint16(b2[:], 2) // block align
	if err := w(b2[:]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint16(b2[:], bitsPerSample)
	if err := w(b2[:]); err != nil {
		return err
	}

	if err := w([]byte("data")); err != nil {
		return err
	}
	if err := w(le(uint32(dataLen))); err != nil {
		return err
	}

	buf := make([]byte, 0, len(samples)*2)
	var s2 [2]byte
	for _, s := range samples {
		v := int16(0)
		if s > 1 {
			v = 32767
		} else if s < -1 {
			v = -32768
		} else {
			v = int16(s * 32767)
		}
		binary.LittleEndian.PutUint16(s2[:], uint16(v))
		buf = append(buf, s2[:]...)
	}
	return w(buf)
}

// Sweep deletes clip files (and their sidecar .json) with modification
// times older than retentionDays. It returns the number of clips removed.
// The sweep is safe against an empty/missing root.
func (r *Recorder) Sweep(retentionDays int) (int, error) {
	if retentionDays <= 0 {
		return 0, nil
	}
	cutoff := time.Now().AddDate(0, 0, -retentionDays)
	removed := 0
	err := filepath.WalkDir(r.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil //nolint: skip unreadable entries
		}
		if d.IsDir() {
			return nil
		}
		info, statErr := d.Info()
		if statErr != nil || info.ModTime().After(cutoff) {
			return nil
		}
		if ext := filepath.Ext(path); ext == ".wav" || ext == ".json" {
			if os.Remove(path) == nil {
				removed++
			}
		}
		return nil
	})
	// Prune now-empty date directories (best-effort).
	_ = filepath.WalkDir(r.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		entries, _ := os.ReadDir(path)
		if len(entries) == 0 && path != r.root {
			_ = os.Remove(path)
		}
		return nil
	})
	return removed, err
}
