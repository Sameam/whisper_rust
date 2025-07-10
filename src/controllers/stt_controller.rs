use std::sync::Arc;
use uuid::Uuid;
use anyhow::{Error}; 
use std::result::Result::Ok;

use crate::services::audio_converter::AudioFormat;
use crate::services::silero::VADService; 
use crate::models::api_models::{CreateSessionResponse, STTResponse, TranscriptionResult};
use crate::services::stt::{STTConfig, STTService};

#[derive(Clone)]
pub struct STTController {
  stt_service: Arc<STTService>
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct STTStats {
    pub active_sessions: usize,
    pub active_transcriptions: usize,
}

impl STTController {
  pub async fn new(whisper_model_path : &str, vad_model_path : &str) -> Result<Self, Error> {
    let vad_service = Arc::new(VADService::new(vad_model_path).await?);

    let config = STTConfig::default(); 

    let stt_service = Arc::new(STTService::new(whisper_model_path, vad_service, config).await?);

    Ok(STTController {
      stt_service
    })
  }


  pub async fn create_session(&self, sample_rate: u32) -> Result<CreateSessionResponse, Error> {
    let session_id = self.stt_service.create_session(sample_rate).await?; 

    Ok(CreateSessionResponse {
      session_id: session_id, message: "VAD + Whisper session created sucessfully".to_string()
    })
  }

  pub async fn process_audio(&self, session_id: Uuid, audio_data : &[u8], sample_rate: u32, audio_format: AudioFormat) -> Result<STTResponse, Error> {
    let start_time = std::time::Instant::now(); 

    let transcription_result = self.stt_service.process_audio(&session_id, audio_data, sample_rate, audio_format).await?; 

    let processing_time_ms = start_time.elapsed().as_millis() as u64;

    let response = match transcription_result {
      Some (result) => STTResponse {
        session_id : session_id, speech_detected: true, transcription: Some(result),  timestamp: chrono::Utc::now(), processing_time_ms,
      },
      None => STTResponse { 
        session_id: session_id, speech_detected: false,transcription: None,timestamp: chrono::Utc::now(),processing_time_ms, 
      }
    };

    Ok(response)
  }

  pub async fn force_transcribe(&self, session_id: &Uuid) -> Result<Option<TranscriptionResult>, Error> {
    self.stt_service.force_transcribe(session_id).await
  }
  
  pub fn _get_transcription_history(&self, session_id: &Uuid) -> Result<Vec<TranscriptionResult>, Error> {
    self.stt_service.get_transcription_history(session_id)
  }
  
  pub fn remove_session(&self, session_id: &Uuid) -> bool {
    self.stt_service.remove_session(session_id)
  }
  
  pub async fn _get_stats(&self) -> Result<STTStats, Error> {
    let service_stats = self.stt_service.get_stats().await;
    
    Ok(STTStats {
      active_sessions: service_stats.active_sessions,
      active_transcriptions: service_stats.active_transcriptions,
    })
  }


}