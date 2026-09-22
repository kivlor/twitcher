// Package store persists detections to a local SQLite database with a schema
// compatible with BirdNET-Go's `notes` and `results` tables, so external
// tooling written against BirdNET-Go can read our database directly.
//
// The entity definitions mirror BirdNET-Go's
// internal/datastore/entities/note.go and results.go (see
// https://github.com/tphakala/birdnet-go), minus the review/comment/lock
// entities which twitcher does not use.
package store

import (
	"errors"
	"fmt"
	"time"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	gormlogger "gorm.io/gorm/logger"
)

// Note is the GORM model for the 'notes' table; field set and column names
// match BirdNET-Go's NoteEntity for schema compatibility.
type Note struct {
	ID             uint `gorm:"primaryKey"`
	SourceNode     string
	Date           string `gorm:"index:idx_notes_date;index:idx_notes_date_commonname_confidence;index:idx_notes_sciname_date;index:idx_notes_sciname_date_optimized,priority:2"`
	Time           string `gorm:"index:idx_notes_time"`
	BeginTime      time.Time
	EndTime        time.Time
	SpeciesCode    string
	ScientificName string  `gorm:"index:idx_notes_sciname;index:idx_notes_sciname_date;index:idx_notes_sciname_date_optimized,priority:1"`
	CommonName     string  `gorm:"index:idx_notes_comname;index:idx_notes_date_commonname_confidence"`
	Confidence     float64 `gorm:"index:idx_notes_date_commonname_confidence"`
	Latitude       float64
	Longitude      float64
	Threshold      float64
	Sensitivity    float64
	ClipName       string
	ProcessingTime time.Duration
	Unlikely       bool `gorm:"default:false"`

	// Results are saved explicitly by Store.Save in the same transaction;
	// GORM association auto-save is not used.
	Results []Result `gorm:"foreignKey:NoteID;constraint:OnDelete:CASCADE"`
}

// TableName ensures GORM uses the BirdNET-Go table name.
func (Note) TableName() string { return "notes" }

// Result represents an additional species prediction for a detection; maps to
// the 'results' table, mirroring BirdNET-Go's ResultsEntity.
type Result struct {
	ID         uint `gorm:"primaryKey"`
	NoteID     uint `gorm:"index;not null;constraint:OnDelete:CASCADE,OnUpdate:CASCADE;foreignKey:NoteID;references:ID"`
	Species    string
	Confidence float32
}

// TableName ensures GORM uses the BirdNET-Go table name.
func (Result) TableName() string { return "results" }

// Store is a SQLite-backed detection store.
type Store struct {
	db *gorm.DB
}

// Open opens (creating if needed) the SQLite database at path and migrates the
// notes/results schema. WAL mode and a busy timeout are set via the DSN so
// external readers can query the database while twitcher writes
// (EXTRACTION_PLAN §F6).
func Open(path string) (*Store, error) {
	if path == "" {
		return nil, errors.New("store: empty database path")
	}
	dsn := fmt.Sprintf("%s?_journal_mode=WAL&_busy_timeout=30000&_foreign_keys=ON&_synchronous=NORMAL&_cache_size=-16000", path)
	db, err := gorm.Open(sqlite.Open(dsn), &gorm.Config{
		Logger: gormlogger.Discard,
	})
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}
	if err := db.AutoMigrate(&Note{}, &Result{}); err != nil {
		_ = closeDB(db)
		return nil, fmt.Errorf("store: migrate: %w", err)
	}
	return &Store{db: db}, nil
}

func closeDB(db *gorm.DB) error {
	sqlDB, err := db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}

// Close closes the database.
func (s *Store) Close() error { return closeDB(s.db) }

// Save persists a note and its results in a single transaction. IDs are reset
// first so a retried transaction does not reuse stale keys, mirroring
// BirdNET-Go's SaveNote semantics.
func (s *Store) Save(note *Note, results []*Result) error {
	if note == nil {
		return errors.New("store: note cannot be nil")
	}
	note.ID = 0
	for _, r := range results {
		r.ID = 0
		r.NoteID = 0
	}
	return s.db.Transaction(func(tx *gorm.DB) error {
		// Omit Results: saved separately below for explicit error handling.
		if err := tx.Omit("Results").Create(note).Error; err != nil {
			return fmt.Errorf("store: create note: %w", err)
		}
		for _, r := range results {
			r.NoteID = note.ID
			if err := tx.Create(r).Error; err != nil {
				return fmt.Errorf("store: create result: %w", err)
			}
		}
		return nil
	})
}

// Count returns the number of stored notes.
func (s *Store) Count() (int64, error) {
	var n int64
	if err := s.db.Model(&Note{}).Count(&n).Error; err != nil {
		return 0, fmt.Errorf("store: count: %w", err)
	}
	return n, nil
}

// DeleteOlderThan removes notes (and their results, via the FK cascade)
// whose BeginTime is before the given cutoff, returning the number of
// notes removed. Used by the retention sweep to bound DB growth (§N3).
func (s *Store) DeleteOlderThan(cutoff time.Time) (int64, error) {
	res := s.db.Where("begin_time < ?", cutoff).Delete(&Note{})
	if res.Error != nil {
		return 0, fmt.Errorf("store: delete older than %s: %w", cutoff, res.Error)
	}
	return res.RowsAffected, nil
}

// DeleteWithMissingClip removes notes whose ClipName no longer exists on
// disk (their clips were pruned by the retention sweep). clipRoot is the
// directory clip names are relative to; a nil checker keeps all notes.
func (s *Store) DeleteWithMissingClip(exists func(clipName string) bool) (int64, error) {
	var names []string
	if err := s.db.Model(&Note{}).Where("clip_name != ''").Pluck("clip_name", &names).Error; err != nil {
		return 0, fmt.Errorf("store: pluck clip names: %w", err)
	}
	var removed int64
	for _, name := range names {
		if !exists(name) {
			res := s.db.Where("clip_name = ?", name).Delete(&Note{})
			if res.Error != nil {
				return removed, fmt.Errorf("store: delete note with clip %s: %w", name, res.Error)
			}
			removed += res.RowsAffected
		}
	}
	return removed, nil
}

// DB exposes the underlying *gorm.DB for read-side tooling and tests.
func (s *Store) DB() *gorm.DB { return s.db }
