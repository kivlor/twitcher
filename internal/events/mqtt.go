package events

import (
	"encoding/json"
	"log"
	"sync"
	"time"

	paho "github.com/eclipse/paho.mqtt.golang"

	"github.com/kivlor/twitcher/internal/config"
)

// queueSize bounds the internal event queue; overflow drops the oldest
// pending event (fresh detections win).
const queueSize = 128

// MQTTEmitter publishes detections to an MQTT broker. Publishes while
// disconnected are suppressed; paho's auto-reconnect handles the broker
// coming back, and a last-will announces twitcher as "offline" (§F7).
type MQTTEmitter struct {
	client      paho.Client
	topic       string
	statusTopic string
	qos         byte
	retain      bool

	mu      sync.Mutex
	queue   chan []byte
	dropped uint64
	wg      sync.WaitGroup
}

// NewMQTTEmitter connects to the broker described by cfg. Connect is
// asynchronous: a broker that is down at startup does not block twitcher.
// The returned emitter publishes to cfg.Topic with the given QoS/retain.
func NewMQTTEmitter(cfg config.MQTT) (*MQTTEmitter, error) {
	opts := paho.NewClientOptions().
		AddBroker(cfg.Broker).
		SetClientID(cfg.ClientID).
		SetAutoReconnect(true).
		SetConnectRetry(true). // keep retrying initial connect in background
		SetMaxReconnectInterval(60 * time.Second).
		SetWriteTimeout(5 * time.Second)
	if cfg.Username != "" {
		opts.SetUsername(cfg.Username)
		opts.SetPassword(cfg.Password)
	}
	// Last will: broker-side "offline" notice if twitcher dies uncleanly;
	// we publish "online" ourselves once connected (birth message).
	opts.SetWill(cfg.StatusTopic, "offline", cfg.QoS, true)
	opts.SetOnConnectHandler(func(c paho.Client) {
		if token := c.Publish(cfg.StatusTopic, cfg.QoS, true, "online"); token.WaitTimeout(5*time.Second) && token.Error() != nil {
			log.Printf("mqtt: birth publish failed: %v", token.Error())
		} else {
			log.Printf("mqtt: connected to %s", cfg.Broker)
		}
	})

	e := &MQTTEmitter{
		client:      paho.NewClient(opts),
		topic:       cfg.Topic,
		statusTopic: cfg.StatusTopic,
		qos:         cfg.QoS,
		retain:      cfg.Retain,
		queue:       make(chan []byte, queueSize),
	}
	e.client.Connect() // async (ConnectRetry=true)
	e.wg.Add(1)
	go e.run()
	return e, nil
}

// run drains the queue in a single goroutine, mirroring upstream
// internal/mqtt: publish only when connected, drop otherwise.
func (e *MQTTEmitter) run() {
	defer e.wg.Done()
	for payload := range e.queue {
		if !e.client.IsConnected() {
			e.dropped++
			continue
		}
		token := e.client.Publish(e.topic, e.qos, e.retain, payload)
		if !token.WaitTimeout(5*time.Second) || token.Error() != nil {
			e.dropped++
		}
	}
}

// Emit queues a detection for publication. It never blocks: a full queue
// drops the oldest pending event.
func (e *MQTTEmitter) Emit(d Detection) {
	payload, err := json.Marshal(d)
	if err != nil {
		log.Printf("mqtt: marshal event: %v", err)
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	select {
	case e.queue <- payload:
	default:
		// drop-oldest, same policy as the audio path (N2)
		select {
		case <-e.queue:
		default:
		}
		e.queue <- payload
		e.dropped++
	}
}

// Close waits briefly for queued events to drain, then disconnects so the
// broker-side last-will fires cleanly via a clean disconnect (paho sends
// the will only on abrupt loss; on clean close the "online" retain stays
// until it is overwritten or expires).
func (e *MQTTEmitter) Close() {
	// Give the drain goroutine a moment to flush what is queued.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		e.mu.Lock()
		n := len(e.queue)
		e.mu.Unlock()
		if n == 0 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	close(e.queue)
	e.wg.Wait()
	if e.client.IsConnected() {
		// best-effort explicit offline notice on clean shutdown
		token := e.client.Publish(e.statusTopic, e.qos, true, "offline")
		token.WaitTimeout(3 * time.Second)
	}
	// Disconnect can block while paho's connect-retry loop is mid-attempt
	// (e.g. broker unreachable at shutdown); never let it hang the process.
	done := make(chan struct{})
	go func() {
		e.client.Disconnect(250)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		log.Printf("mqtt: disconnect timed out")
	}
}
