package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "twitcher.yaml")
	yaml := `
db: /var/lib/twitcher/twitcher.db
device: "HD-Audio"
threshold: 0.55
overlap: 0.5
latitude: 60.1
longitude: 24.9
notes_retention_days: 90
clips:
  enabled: true
  dir: /var/lib/twitcher/clips
  retention_days: 14
mqtt:
  broker: tcp://10.0.0.5:1883
  topic: birds/detections
  qos: 0
`
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DB != "/var/lib/twitcher/twitcher.db" {
		t.Errorf("db = %q", cfg.DB)
	}
	if cfg.Threshold != 0.55 || cfg.Overlap != 0.5 {
		t.Errorf("threshold/overlap = %v/%v", cfg.Threshold, cfg.Overlap)
	}
	if !cfg.Clips.Enabled || cfg.Clips.Dir != "/var/lib/twitcher/clips" || cfg.Clips.RetentionDays != 14 {
		t.Errorf("clips = %+v", cfg.Clips)
	}
	if cfg.MQTT.Broker != "tcp://10.0.0.5:1883" || cfg.MQTT.Topic != "birds/detections" || cfg.MQTT.QoS != 0 {
		t.Errorf("mqtt = %+v", cfg.MQTT)
	}
	if cfg.NotesRetentionDays != 90 {
		t.Errorf("notes retention = %d", cfg.NotesRetentionDays)
	}
	// unset fields keep defaults
	if cfg.Model != "model.tflite" || cfg.Threads != 2 || cfg.Sensitivity != 1.0 {
		t.Errorf("defaults not kept: %+v", cfg)
	}
}

func TestLoadDefaults(t *testing.T) {
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DB != "twitcher.db" || cfg.Threshold != 0.80 || cfg.Overlap != 0.33 {
		t.Errorf("unexpected defaults: %+v", cfg)
	}
	if cfg.Clips.RetentionDays != 30 || cfg.MQTT.Topic != "birdnet/detections" {
		t.Errorf("unexpected defaults: %+v", cfg)
	}
}

func TestEnvOverride(t *testing.T) {
	t.Setenv("TWITCHER_DB", "/tmp/env.db")
	t.Setenv("TWITCHER_MQTT_BROKER", "tcp://env:1883")
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DB != "/tmp/env.db" || cfg.MQTT.Broker != "tcp://env:1883" {
		t.Errorf("env not applied: %+v", cfg)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "missing.yaml")); err == nil {
		t.Error("expected error for missing config file")
	}
}
