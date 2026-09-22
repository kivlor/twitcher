package clips

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSave(t *testing.T) {
	root := t.TempDir()
	r, err := NewRecorder(root)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Date(2025, 6, 15, 14, 30, 5, 0, time.UTC)
	samples := make([]float32, 48000) // 1 s of silence
	meta := struct {
		CommonName string `json:"common_name"`
	}{"Tawny Owl"}

	name, err := r.Save(start, samples, "Tawny Owl (Eurasian)", meta)
	if err != nil {
		t.Fatal(err)
	}
	wantDir := "2025/06/15"
	if filepath.Dir(name) != wantDir {
		t.Errorf("clip dir = %q, want under %q", name, wantDir)
	}
	if filepath.Ext(name) != ".wav" {
		t.Errorf("clip ext = %q", name)
	}
	if _, err := os.Stat(filepath.Join(root, name)); err != nil {
		t.Errorf("wav missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, wantDir, "clip.json")); err == nil {
		// sanity: sidecar exists next to the wav
	}

	// Read the sidecar and check the marshalled metadata.
	sidecar := name[:len(name)-len(".wav")] + ".json"
	data, err := os.ReadFile(filepath.Join(root, sidecar))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m["common_name"] != "Tawny Owl" {
		t.Errorf("sidecar = %s", data)
	}
}

func TestSaveSanitisesNames(t *testing.T) {
	r, err := NewRecorder(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	name, err := r.Save(time.Now(), nil, "Eurasian Magpie!!!", nil)
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Base(name)
	for _, c := range base {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			c == '-' || c == '_' || c == '.') {
			t.Errorf("unsafe char %q in %q", c, base)
		}
	}
}

func TestSweep(t *testing.T) {
	root := t.TempDir()
	r, err := NewRecorder(root)
	if err != nil {
		t.Fatal(err)
	}
	old := time.Now().AddDate(0, 0, -40)
	oldDir := filepath.Join(root, old.Format("2006/01/02"))
	if err := os.MkdirAll(oldDir, 0o755); err != nil {
		t.Fatal(err)
	}
	oldClip := filepath.Join(oldDir, "old.wav")
	oldSidecar := filepath.Join(oldDir, "old.json")
	newClip, err := r.Save(time.Now(), make([]float32, 480), "Wren", nil)
	if err != nil {
		t.Fatal(err)
	}

	// Set the old clip's mtime back 40 days.
	if err := os.WriteFile(oldClip, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldSidecar, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(oldClip, old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(oldSidecar, old, old); err != nil {
		t.Fatal(err)
	}

	removed, err := r.Sweep(30)
	if err != nil {
		t.Fatal(err)
	}
	if removed != 2 {
		t.Errorf("removed = %d, want 2 (wav+json)", removed)
	}
	if _, err := os.Stat(oldClip); !os.IsNotExist(err) {
		t.Errorf("old clip survived: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, newClip)); err != nil {
		t.Errorf("new clip was swept: %v", err)
	}
	// The emptied date dir should be pruned too.
	if _, err := os.Stat(oldDir); !os.IsNotExist(err) {
		t.Errorf("empty date dir survived")
	}
}

func TestSweepDisabled(t *testing.T) {
	r, err := NewRecorder(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	removed, err := r.Sweep(0)
	if err != nil || removed != 0 {
		t.Errorf("sweep(0) = %d, %v", removed, err)
	}
}
