use std::sync::Arc;
use uuid::Uuid;
use anyhow::{Error}; 
use std::result::Result::Ok;

use crate::services::silero::VADService; 
use crate::models::api_models::{CreateSessionResponse, VadResponse};

#[derive(Clone)]
pub struct VADController {
  pub vad_service : Arc<VADService>
}

impl VADController {
  pub async fn new(model_path: &str) -> Result<Self, Error> {
    let vad_service = Arc::new(VADService::new(model_path).await?);
    Ok(VADController { vad_service: vad_service })
  }

  pub async fn create_session(&self, hangover_ms: u32, pad_ms: u32 ) -> Result<CreateSessionResponse, Error> {
    let session_id = self.vad_service.create_session( hangover_ms, pad_ms).await?; 

    Ok(CreateSessionResponse { session_id: session_id, message: "VAD session created sucessfully".to_string() })
  }

  pub async fn process_audio(&self, session_id: &Uuid, audio_data: &[u8]) -> Result<VadResponse, Error> {

    let speech_detected = self.vad_service.process_audio(session_id, audio_data).await?;

    Ok(VadResponse { session_id: *session_id, speech_detected: speech_detected })
  }

  #[allow(unused)]
  pub async fn process_audio_bytes(&self, requests: Vec<(Uuid, &[u8])>) -> Result<Vec<VadResponse>, Error> {
    let results = self.vad_service.process_audio_bytes(requests).await;

    let responses: Result<Vec<VadResponse>, Error> = results.into_iter().map(|result| {
        result.map(|(session_id, speech_detected)| VadResponse {
          session_id,
          speech_detected,
        })
      })
      .collect(); // This collects into Result<Vec<VadResponse>, Error>

    responses // Return the Result directly
  }


  pub fn remove_session(&self, session_id: &Uuid) -> Result<bool, Error> {
    let removed = self.vad_service.remove_session(session_id); 

    Ok(removed)
  }

}

