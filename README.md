# STT (Speech-to-Text) Service

A high-performance, real-time speech recognition service built in Rust that provides both Voice Activity Detection (VAD) and speech transcription capabilities through WebSocket connections.

## Overview

This service combines OpenAI's Whisper model for speech transcription with Silero VAD for voice activity detection, offering a complete solution for real-time audio processing applications. Built with Actix-web and optimized for low-latency streaming scenarios.

## ⚠️ Research Project Status

**This is currently a research and development project.** While the service is functional and demonstrates real-time speech recognition capabilities, it should be considered experimental and not production-ready. The project is actively being developed and tested, with ongoing improvements to audio processing algorithms and transcription quality.

**Key considerations:**
- Audio format support is still being optimized
- Transcription quality varies significantly across different audio formats
- API and configuration may change as the project evolves
- Contributions, feedback, and testing are welcome and encouraged

## Features

### 🎙️ Speech-to-Text (STT)
- **Real-time transcription** using Whisper model
- **Streaming audio processing** with automatic speech detection
- **Session-based processing** with unique session management
- **Intelligent speech buffering** with configurable silence detection
- **Force transcription** capability for remaining audio in buffer
- **Transcription history** tracking per session

### 🔊 Voice Activity Detection (VAD)
- **Real-time speech detection** using Silero VAD model
- **Continuous audio monitoring** to detect speech presence
- **Configurable sensitivity** with threshold settings
- **Low-latency processing** for real-time applications

### 🎵 Audio Processing
- **Multiple audio formats supported**:
  - PCM16 (LINEAR16) - Primary format
  - FLOAT32
  - μ-law (MULAW) - for telephony **Transcription is not the best**
  - A-law (ALAW) - for telephony **Transcription is not the best**
  - PCM8, PCM24, PCM32
- **Automatic sample rate conversion** to 16kHz
- **SIMD-optimized audio conversion** for performance
- **Phantom audio suppression** for μ-law streams
- **Noise gating** and normalization
- **Note: For best result used audio formats : PCM16(LINEAR16) and sample_rate 16kHz**

### 🌐 WebSocket API
- **Real-time streaming** via WebSocket connections
- **Session management** with UUID-based tracking
- **Concurrent processing** with configurable limits
- **Dynamic configuration** during sessions

## API Endpoints

### WebSocket Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/stt` | Real-time speech-to-text streaming |
| `/silence_detection` | Voice activity detection |
| `/` | Health check endpoint |

### WebSocket Commands

| Command | Description |
|---------|-------------|
| `end_session` | Gracefully terminate session |
| `reset_session` | Clear buffers and force transcribe remaining audio |
| `configure` | Update session parameters dynamically |

## Quick Start

### Prerequisites

- Rust 1.70+
- Whisper model file (`ggml-base.bin`) **Any Whisper Models in ggml format**
- Silero VAD model file (`silero_vad.onnx`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd STT
```

2. Place model files in the `models/` directory:
   - `models/ggml-base.bin` (Whisper model)
   - `models/silero_vad.onnx` (Silero VAD model)

3. Build and run:
```bash
cargo run
```

The service will start on `127.0.0.1:8080`

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Server configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8080

# Model paths
WHISPER_MODEL_PATH=models/ggml-base.bin
SILERO_VAD_MODEL_PATH=models/silero_vad.onnx

# Logging
RUST_LOG=info
```

### Default Configuration

```rust
STTConfig {
    sample_rate: 16000,
    silence_duration_ms: 500,
    min_speech_duration_ms: 100,
    language: "en",
    streaming_mode: true,
    max_concurrent_transcriptions: 5,
}
```

### Response Format

#### STT Results
```json
{
  "event": "stt_result",
  "session_id": "uuid",
  "speech_detected": true,
  "transcription": "transcribed text",
  "processing_time_ms": 150
}
```

#### VAD Results
```json
{
  "event": "vad_result",
  "session_id": "uuid",
  "speech_detected": true
}
```

## Architecture

### Core Components

- **STTService** - Main speech recognition orchestration
- **VADService** - Voice activity detection processing
- **AudioConverterService** - Multi-format audio conversion
- **ResamplingService** - Sample rate conversion
- **SessionManager** - WebSocket session handling

### Data Flow

1. **Audio ingestion** via WebSocket binary messages
2. **Format conversion** to standardized PCM16 format
3. **Resampling** to 16kHz if needed
4. **VAD processing** for speech detection
5. **Audio buffering** during speech segments
6. **Transcription** when speech ends or silence threshold reached
7. **Result delivery** via WebSocket JSON responses

## Performance

- **Low latency**: ~30ms processing chunks
- **Concurrent sessions**: Configurable limits (default: 5)
- **SIMD optimization**: Hardware-accelerated audio processing
- **Memory efficient**: Streaming processing with bounded buffers

## Future Maintenance & Improvements

### Audio Format Optimization
- **Transcription Quality**: While the service supports multiple audio formats, transcription quality may vary significantly compared to the optimal PCM16 format at 16kHz sample rate. Non-PCM formats (μ-law, A-law) especially may produce suboptimal results and require algorithm improvements.
- **Format Support Enhancement**: Currently, the service only supports raw byte streams for multiple sample rates and audio formats. Future versions should consider:
  - Support for containerized audio formats (WAV, FLAC, etc.)
  - Improved codec support for compressed formats
  - Better audio preprocessing for non-standard formats

### Planned Improvements
- Enhanced audio normalization algorithms for telephony formats
- Advanced noise reduction preprocessing
- Support for additional audio containers and codecs
- Improved sample rate conversion algorithms
- Better handling of audio artifacts in compressed formats

## Use Cases

- **Real-time voice applications** (voice assistants, transcription services)
- **Audio compress format integration** (supports formats like μ-law/A-law)
- **Streaming audio processing** with low latency requirements
- **Multi-tenant applications** with session isolation
- **Voice-controlled interfaces** requiring reliable speech detection

## Dependencies

### Core Dependencies
- `actix-web` - Web framework
- `tokio` - Async runtime
- `whisper-rs` - Whisper model bindings
- `silero-vad-rs` - Silero VAD bindings
- `rubato` - Audio resampling
- `dasp` - Digital audio signal processing

### Full dependency list available in `Cargo.toml`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please create an issue in the repository.