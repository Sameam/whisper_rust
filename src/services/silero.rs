use dashmap::DashMap;
use std::sync::{Arc};
use tokio::{task};
use uuid::Uuid;
use anyhow::{Error, Ok};

use crate::utils::session::Session;


pub struct VADService {
  sessions: Arc<DashMap<Uuid, Session>>,
  model_path: String,
}

impl VADService {

  pub async fn new(model_path : &str) -> Result<Self, Error> {

    let model_path_string: String = model_path.to_string().clone();
    
    task::spawn_blocking(move || {
      std::fs::metadata(&model_path_string)
    }).await??;

    
    Ok(VADService { 
      sessions: Arc::new(DashMap::new()), 
      model_path: model_path.to_string()
    })
  }

  pub async fn create_session(&self, hangover_ms: u32, pad_ms: u32) -> Result<Uuid, Error> {
    let session_id: Uuid = Uuid::new_v4(); 
    let model_path: String =  self.model_path.clone();

    let session: Session = task::spawn_blocking(move || {
      Session::new(
        &model_path, 0.5,16000, hangover_ms,pad_ms
      )
    }).await??;

    self.sessions.insert(session_id, session);

    log::info!("Created VAD service with session: {}", session_id);

    Ok(session_id)
  }


  pub async fn process_audio(&self, session_id: &Uuid, audio_data: &[u8]) -> Result<bool, Error> {
    let sessions: Arc<DashMap<Uuid, Session>> = self.sessions.clone(); 

    let session_id: Uuid = *session_id;
    let audio_data: Vec<u8> = audio_data.to_vec();

    let speech_detected: bool = task::spawn_blocking(move || -> Result<bool, Error> {
      if let Some(mut session) = sessions.get_mut(&session_id) {
        Ok(session.process_raw_bytes(&audio_data))
      }
      else {
        Err(anyhow::anyhow!("Session not found: {}", session_id))
      }
    }).await??;

    Ok(speech_detected)
  }

  #[allow(unused)]
  pub async fn process_audio_bytes(&self, requests: Vec<(Uuid, &[u8])>) -> Vec<Result<(Uuid, bool), Error>> {
    let futures = requests.into_iter().map(|(session_id, audio_data)| {
      let sessions = self.sessions.clone();
      let audio_data = audio_data.to_vec();
      async move {
        let result = task::spawn_blocking(move || -> Result<bool, Error> {
          if let Some(mut session) = sessions.get_mut(&session_id) {
            Ok(session.process_raw_bytes(&audio_data))
          } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
          }
        }).await?;
        
        Ok((session_id, result?))
      }
    });

    futures::future::join_all(futures).await
  }

  pub fn remove_session(&self, session_id: &Uuid) -> bool {
    let removed = self.sessions.remove(session_id).is_some();
    if removed {
      log::info!("Removed VAD session: {}", session_id)
    }
    removed
  }



}


