package events

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/kivlor/twitcher/internal/store"
)

func TestDetectionFromNote(t *testing.T) {
	begin := time.Date(2025, 6, 15, 14, 30, 5, 0, time.UTC)
	note := &store.Note{
		SourceNode:     "pi3b",
		Date:           "2025-06-15",
		Time:           "14:30:05",
		BeginTime:      begin,
		EndTime:        begin.Add(3 * time.Second),
		ScientificName: "Strix aluco",
		CommonName:     "Tawny Owl",
		Confidence:     0.99,
		Latitude:       60.1,
		Longitude:      24.9,
		Threshold:      0.80,
		Sensitivity:    1.0,
		ClipName:       "2025/06/15/x.wav",
	}
	results := []*store.Result{
		{Species: "Strix aluco", Confidence: 0.99},
		{Species: "Bubo bubo", Confidence: 0.01},
	}
	d := DetectionFromNote(note, results)

	// JSON keys must mirror the note row plus results (§F7).
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"begin_time", "end_time", "date", "time",
		"scientific_name", "common_name", "confidence", "latitude", "longitude",
		"threshold", "sensitivity", "clip_name", "results"} {
		if _, ok := m[key]; !ok {
			t.Errorf("missing key %q in %s", key, b)
		}
	}
	if len(d.Results) != 2 || d.Results[0].Confidence != 0.99 {
		t.Errorf("results = %+v", d.Results)
	}
	if d.ScientificName != "Strix aluco" || d.CommonName != "Tawny Owl" {
		t.Errorf("names not mapped: %+v", d)
	}
}
