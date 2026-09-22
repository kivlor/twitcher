// Package config loads twitcher's YAML configuration.
//
// Precedence: built-in defaults < YAML config file < command-line flags.
// A subset of settings can also come from the environment: TWITCHER_DB,
// TWITCHER_DEVICE, TWITCHER_MQTT_BROKER (useful for systemd unit
// Environment= overrides).
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// MQTT configures the optional detection event emitter (F7). Empty Broker
// disables emission entirely.
type MQTT struct {
	Broker      string `yaml:"broker"`       // e.g. "tcp://10.0.0.5:1883"; "" = off
	Topic       string `yaml:"topic"`        // e.g. "birdnet/detections"
	StatusTopic string `yaml:"status_topic"` // last-will / birthplace topic
	ClientID    string `yaml:"client_id"`
	Username    string `yaml:"username"`
	Password    string `yaml:"password"`
	QoS         byte   `yaml:"qos"`
	Retain      bool   `yaml:"retain"`
}

// Clips configures optional detection audio clip recording (F4).
type Clips struct {
	Enabled       bool   `yaml:"enabled"`
	Dir           string `yaml:"dir"`            // clips are stored under Dir/YYYY/MM/DD/
	RetentionDays int    `yaml:"retention_days"` // delete clips older than N days; 0 = keep forever
}

// Config is twitcher's full configuration.
type Config struct {
	Model       string  `yaml:"model"`
	Labels      string  `yaml:"labels"`
	DB          string  `yaml:"db"` // "" disables persistence
	Device      string  `yaml:"device"`
	Threshold   float64 `yaml:"threshold"`
	Overlap     float64 `yaml:"overlap"`
	Threads     int     `yaml:"threads"`
	Latitude    float64 `yaml:"latitude"`
	Longitude   float64 `yaml:"longitude"`
	Sensitivity float64 `yaml:"sensitivity"`
	Node        string  `yaml:"node"` // hostname if empty

	Clips              Clips `yaml:"clips"`
	NotesRetentionDays int   `yaml:"notes_retention_days"` // delete notes older than N days; 0 = off

	MQTT MQTT `yaml:"mqtt"`
}

// Default returns the built-in defaults used when no config file is present.
func Default() Config {
	return Config{
		Model:       "model.tflite",
		Labels:      "labels.txt",
		DB:          "twitcher.db",
		Device:      "default",
		Threshold:   0.80,
		Overlap:     0.33,
		Threads:     2,
		Sensitivity: 1.0,
		Clips: Clips{
			Dir:           "clips",
			RetentionDays: 30,
		},
		NotesRetentionDays: 0,
		MQTT: MQTT{
			Topic:       "birdnet/detections",
			StatusTopic: "birdnet/status",
			ClientID:    "twitcher",
			QoS:         1,
		},
	}
}

// Load reads a YAML config file on top of the defaults. Values not present in
// the file keep their defaults.
func Load(path string) (Config, error) {
	cfg := Default()
	if path == "" {
		cfg.applyEnv()
		return cfg, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("config: read %s: %w", path, err)
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("config: parse %s: %w", path, err)
	}
	cfg.applyEnv()
	return cfg, nil
}

// applyEnv overlays environment overrides on top of file/defaults.
func (c *Config) applyEnv() {
	if v := os.Getenv("TWITCHER_DB"); v != "" {
		c.DB = v
	}
	if v := os.Getenv("TWITCHER_DEVICE"); v != "" {
		c.Device = v
	}
	if v := os.Getenv("TWITCHER_MQTT_BROKER"); v != "" {
		c.MQTT.Broker = v
	}
}
