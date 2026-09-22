// Package events defines the detection event emitted to external sinks
// (MQTT, later Kafka) and the MQTT sink implementation (F7).
//
// Emission is best-effort: the database is the source of truth, events ride
// on a bounded queue drained by a single goroutine, and a downed broker
// means dropped events — never a stalled pipeline.
package events

import (
	"time"

	"github.com/kivlor/twitcher/internal/store"
)

// Result mirrors one row of the top-3 predictions stored in `results`.
type Result struct {
	Species    string  `json:"species"`
	Confidence float32 `json:"confidence"`
}

// Detection is the flat JSON payload published per detection, mirroring the
// note row plus its top-3 results (EXTRACTION_PLAN §F7).
type Detection struct {
	SourceNode     string    `json:"source_node,omitempty"`
	BeginTime      time.Time `json:"begin_time"`
	EndTime        time.Time `json:"end_time"`
	Date           string    `json:"date"`
	Time           string    `json:"time"`
	SpeciesCode    string    `json:"species_code,omitempty"`
	ScientificName string    `json:"scientific_name"`
	CommonName     string    `json:"common_name"`
	Confidence     float64   `json:"confidence"`
	Latitude       float64   `json:"latitude"`
	Longitude      float64   `json:"longitude"`
	Threshold      float64   `json:"threshold"`
	Sensitivity    float64   `json:"sensitivity"`
	ClipName       string    `json:"clip_name,omitempty"`
	Results        []Result  `json:"results"`
}

// DetectionFromNote builds the event payload from a stored note and its
// result rows.
func DetectionFromNote(n *store.Note, results []*store.Result) Detection {
	d := Detection{
		SourceNode:     n.SourceNode,
		BeginTime:      n.BeginTime,
		EndTime:        n.EndTime,
		Date:           n.Date,
		Time:           n.Time,
		SpeciesCode:    n.SpeciesCode,
		ScientificName: n.ScientificName,
		CommonName:     n.CommonName,
		Confidence:     n.Confidence,
		Latitude:       n.Latitude,
		Longitude:      n.Longitude,
		Threshold:      n.Threshold,
		Sensitivity:    n.Sensitivity,
		ClipName:       n.ClipName,
		Results:        make([]Result, 0, len(results)),
	}
	for _, r := range results {
		d.Results = append(d.Results, Result{Species: r.Species, Confidence: r.Confidence})
	}
	return d
}
