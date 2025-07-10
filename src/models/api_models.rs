// src/models/api_models.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::services::audio_converter::AudioFormat;

// === REQUEST MODELS ===
#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct VadRequest {
    pub session_id: Uuid,
    pub audio_data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct STTRequest {
    pub session_id: Uuid,
    pub audio_data: Vec<u8>,
    pub language: Option<String>,
    pub enable_timestamps: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct CreateSessionRequest {
    pub config: Option<SessionConfig>,
}

#[allow(unused)]
pub struct VADSessionConfig {
    pub threshold: Option<f32>,
    pub sample_rate: Option<u32>,
    pub language: Option<String>,
}



#[allow(unused)]
#[derive(Debug, Deserialize)]
pub struct SessionConfig {
    pub threshold: Option<f32>,
    pub sample_rate: Option<u32>,
    pub audio_format : Option<AudioFormat>,
    pub language: Option<String>,
    pub silence_duration_ms: Option<u32>, // How long to wait after speech stops
    pub min_speech_duration_ms: Option<u32>, // Minimum speech to transcribe
    pub streaming_mode: Option<bool>,   
}

// === RESPONSE MODELS ===
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
  pub session_id: Uuid,
  pub message: String,
}

#[derive(Debug, Serialize)]
pub struct VadResponse {
  pub session_id: Uuid,
  pub speech_detected: bool,
}

#[derive(Debug, Serialize)]
pub struct BatchVadResponse {
    pub successful_responses: Vec<VadResponse>,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct STTResponse {
    pub session_id: Uuid,
    pub speech_detected: bool,
    pub transcription: Option<TranscriptionResult>,
    pub timestamp: DateTime<Utc>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub duration: f32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TranscriptionSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub confidence: Option<f32>,
}

// === ERROR MODELS ===
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<Uuid>,
}

// === WEBSOCKET MESSAGE MODELS ===
#[derive(Debug, Serialize)]
#[serde(tag = "event")]
pub enum WebSocketResponse {
    #[serde(rename = "session_created")]
    SessionCreated {
        session_id: String,
        message: String,
    },
    #[serde(rename = "vad_result")]
    VadResult {
        session_id: String,
        speech_detected: bool,
    },
    #[serde(rename = "stt_result")]
    STTResult {
        session_id: String,
        speech_detected: bool,
        transcription: Option<String>,
        processing_time_ms: u64,
    },
    #[serde(rename = "error")]
    Error {
        message: String
    },
    #[serde(rename = "session_ended")]
    SessionEnded {
        session_id: String,
        message: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "command")]
pub enum WebSocketCommand {
    #[serde(rename = "end_session")]
    EndSession,
    #[serde(rename = "reset_session")]
    ResetSession,
    #[serde(rename = "configure")]
    Configure { config: SessionConfig },
}


#[derive(Debug, Clone)]
pub struct TranscribeParams {
    pub language: Option<String>,
    pub single_segment: bool,  // For streaming, process as single segment
    pub max_tokens: i32,       // Limit tokens for faster processing
    pub no_context: bool,      // Don't use context from previous audio
}

impl Default for TranscribeParams {
    fn default() -> Self {
        Self {
            language: Some("en".to_string()),
            single_segment: true,
            max_tokens: 224,  // Reasonable default for streaming
            no_context: true,
        }
    }
}

impl TranscribeParams {
    /// Create params optimized for streaming
    pub fn for_streaming() -> Self {
        Self {
            language: Some("en".to_string()),
            single_segment: true,
            max_tokens: 224,
            no_context: true,  // Each chunk processed independently
        }
    }
}

// This is the internal config with defaults applied
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct STTSessionConfig {
    pub language: String,                 // Required, defaults to "en"
    pub silence_duration_ms: u32,         // Required, defaults to 500ms
    pub min_speech_duration_ms: u32,      // Required, defaults to 100ms
    pub streaming_mode: bool,             // Required, defaults to true
}

// Convert from optional client config to required internal config
impl From<SessionConfig> for STTSessionConfig {
    fn from(config: SessionConfig) -> Self {
        STTSessionConfig {
            language: config.language.unwrap_or_else(|| "en".to_string()),
            silence_duration_ms: config.silence_duration_ms.unwrap_or(500),
            min_speech_duration_ms: config.min_speech_duration_ms.unwrap_or(100),
            streaming_mode: config.streaming_mode.unwrap_or(true),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResamplingMethod {
    Custom, Dasp, Rubato, Auto
}
