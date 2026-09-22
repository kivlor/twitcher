package store

import (
	"path/filepath"
	"testing"
	"time"
)

const dateFormat = "2006-01-02"

func newTestStore(t *testing.T) *Store {
	t.Helper()
	st, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	return st
}

func TestSaveAndReadBack(t *testing.T) {
	path := t.TempDir() + "/test.db"
	st, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer st.Close()

	begin := time.Date(2025, 9, 22, 6, 30, 0, 0, time.UTC)
	note := &Note{
		SourceNode:     "pi3",
		Date:           begin.Format("2006-01-02"),
		Time:           begin.Format("15:04:05"),
		BeginTime:      begin,
		EndTime:        begin.Add(3 * time.Second),
		SpeciesCode:    "tawowl1",
		ScientificName: "Strix aluco",
		CommonName:     "Tawny Owl",
		Confidence:     0.998,
		Latitude:       51.5,
		Longitude:      -0.12,
		Threshold:      0.80,
		Sensitivity:    1.0,
		ProcessingTime: 470 * time.Millisecond,
	}
	results := []*Result{
		{Species: "Strix aluco", Confidence: 0.998},
		{Species: "Tyto alba", Confidence: 0.001},
	}
	if err := st.Save(note, results); err != nil {
		t.Fatalf("save: %v", err)
	}
	if note.ID == 0 {
		t.Fatal("note ID not set after save")
	}
	for _, r := range results {
		if r.NoteID != note.ID {
			t.Fatalf("result NoteID %d != note ID %d", r.NoteID, note.ID)
		}
	}

	n, err := st.Count()
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 1 {
		t.Fatalf("count = %d, want 1", n)
	}

	// Read back through a fresh connection, verifying the persisted rows.
	st2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer st2.Close()

	var got Note
	if err := st2.db.Preload("Results").First(&got, note.ID).Error; err != nil {
		t.Fatalf("read back: %v", err)
	}
	if got.ScientificName != "Strix aluco" || got.CommonName != "Tawny Owl" ||
		got.SpeciesCode != "tawowl1" || got.Confidence != 0.998 || got.Date != "2025-09-22" ||
		got.Time != "06:30:00" || got.ProcessingTime != 470*time.Millisecond {
		t.Fatalf("note mismatch: %+v", got)
	}
	if len(got.Results) != 2 || got.Results[0].Species != "Strix aluco" ||
		got.Results[1].Confidence != 0.001 {
		t.Fatalf("results mismatch: %+v", got.Results)
	}

	// Verify table/column names match the BirdNET-Go schema exactly.
	var cols []struct {
		Name string `gorm:"column:name"`
	}
	if err := st2.db.Raw("PRAGMA table_info(notes)").Scan(&cols).Error; err != nil {
		t.Fatalf("pragma: %v", err)
	}
	want := map[string]bool{
		"id": true, "source_node": true, "date": true, "time": true,
		"begin_time": true, "end_time": true, "species_code": true,
		"scientific_name": true, "common_name": true, "confidence": true,
		"latitude": true, "longitude": true, "threshold": true,
		"sensitivity": true, "clip_name": true, "processing_time": true,
		"unlikely": true,
	}
	seen := map[string]bool{}
	for _, c := range cols {
		if !want[c.Name] {
			t.Errorf("unexpected notes column %q", c.Name)
		}
		seen[c.Name] = true
	}
	for c := range want {
		if !seen[c] {
			t.Errorf("missing notes column %q", c)
		}
	}
}

func TestDeleteOlderThan(t *testing.T) {
	st := newTestStore(t)
	old := time.Now().Add(-48 * time.Hour)
	for i, clip := range []string{"", "a.wav"} {
		note := &Note{
			Date:           old.Add(time.Duration(i) * time.Hour).Format(dateFormat),
			Time:           "00:00:00",
			BeginTime:      old.Add(time.Duration(i) * time.Hour),
			EndTime:        old.Add(time.Duration(i) * time.Hour).Add(3 * time.Second),
			ScientificName: "Strix aluco", CommonName: "Tawny Owl",
			Confidence: 0.9, ClipName: clip,
		}
		if err := st.Save(note, []*Result{{Species: "Strix aluco", Confidence: 0.9}}); err != nil {
			t.Fatal(err)
		}
	}
	fresh := &Note{
		Date: time.Now().Format(dateFormat), Time: "00:00:00",
		BeginTime: time.Now(), EndTime: time.Now().Add(3 * time.Second),
		ScientificName: "Strix aluco", CommonName: "Tawny Owl", Confidence: 0.9,
	}
	if err := st.Save(fresh, nil); err != nil {
		t.Fatal(err)
	}

	removed, err := st.DeleteOlderThan(time.Now().Add(-24 * time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if removed != 2 {
		t.Errorf("removed = %d, want 2", removed)
	}
	// Results rows must cascade with their notes.
	var results int64
	st.db.Model(&Result{}).Count(&results)
	if results != 0 {
		t.Errorf("orphaned results: %d", results)
	}
	n, _ := st.Count()
	if n != 1 {
		t.Errorf("remaining notes = %d, want 1", n)
	}
}

func TestDeleteWithMissingClip(t *testing.T) {
	st := newTestStore(t)
	mk := func(clip string, begin time.Time) *Note {
		return &Note{
			Date: begin.Format(dateFormat), Time: "00:00:00",
			BeginTime: begin, EndTime: begin.Add(3 * time.Second),
			ScientificName: "S", CommonName: "C", Confidence: 0.9, ClipName: clip,
		}
	}
	now := time.Now()
	if err := st.Save(mk("kept.wav", now), nil); err != nil {
		t.Fatal(err)
	}
	if err := st.Save(mk("gone.wav", now), nil); err != nil {
		t.Fatal(err)
	}
	if err := st.Save(mk("", now), nil); err != nil {
		t.Fatal(err) // no clip: must never be pruned by clip existence
	}

	exists := map[string]bool{"kept.wav": true}
	removed, err := st.DeleteWithMissingClip(func(name string) bool { return exists[name] })
	if err != nil {
		t.Fatal(err)
	}
	if removed != 1 {
		t.Errorf("removed = %d, want 1", removed)
	}
	n, _ := st.Count()
	if n != 2 {
		t.Errorf("remaining = %d, want 2 (kept.wav + no-clip note)", n)
	}
}
