// Package audio holds the capture-side audio utilities: the overlapping
// window chunker that turns a live 48 kHz mono stream into BirdNET-sized
// 3-second windows, plus WAV decoding for fixtures/tests.
package audio

import (
	"encoding/binary"
	"fmt"
	"os"
	"unsafe"
)

// ReadWavMono48k reads a WAV file and returns mono 48 kHz samples as float32
// in [-1, 1). Supports PCM 16-bit and 32-bit integer data, matching the
// fixture files used by the M1/M2 spikes. Mirrors cmd/m1's reader (the
// verified spike) so tests decode identically to the offline runs.
func ReadWavMono48k(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read wav: %w", err)
	}
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("read wav %s: not a RIFF/WAVE file", path)
	}

	var fmtChunk, dataChunk []byte
	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := data[pos+8 : pos+8+size]
		if len(body) < size { // truncated chunk
			return nil, fmt.Errorf("read wav %s: truncated %s chunk", path, id)
		}
		switch id {
		case "fmt ":
			fmtChunk = body
		case "data":
			dataChunk = body
		}
		pos += 8 + size
		if size%2 == 1 { // RIFF chunks are word-aligned
			pos++
		}
	}
	if fmtChunk == nil || dataChunk == nil {
		return nil, fmt.Errorf("read wav %s: missing fmt or data chunk", path)
	}

	audioFormat := binary.LittleEndian.Uint16(fmtChunk[0:2])
	if audioFormat == 0xFFFE && len(fmtChunk) >= 26 { // WAVE_FORMAT_EXTENSIBLE
		audioFormat = binary.LittleEndian.Uint16(fmtChunk[24:26]) // subformat GUID first 2 bytes
	}
	channels := int(binary.LittleEndian.Uint16(fmtChunk[2:4]))
	rate := int(binary.LittleEndian.Uint32(fmtChunk[4:8]))
	bits := int(binary.LittleEndian.Uint16(fmtChunk[14:16]))

	if channels != 1 || rate != 48000 {
		return nil, fmt.Errorf("read wav %s: want mono 48 kHz, got %d ch %d Hz",
			path, channels, rate)
	}

	switch {
	case audioFormat == 1 && bits == 16:
		out := make([]float32, len(dataChunk)/2)
		for i := range out {
			out[i] = float32(int16(binary.LittleEndian.Uint16(dataChunk[i*2:i*2+2]))) / 32768.0
		}
		return out, nil
	case audioFormat == 1 && bits == 32:
		out := make([]float32, len(dataChunk)/4)
		for i := range out {
			out[i] = float32(int32(binary.LittleEndian.Uint32(dataChunk[i*4:i*4+4]))) / 2147483648.0
		}
		return out, nil
	case audioFormat == 3 && bits == 32: // IEEE float
		out := make([]float32, len(dataChunk)/4)
		for i := range out {
			out[i] = float32FromBits(binary.LittleEndian.Uint32(dataChunk[i*4 : i*4+4]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("read wav %s: unsupported format (tag=%d bits=%d)", path, audioFormat, bits)
	}
}

func float32FromBits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
