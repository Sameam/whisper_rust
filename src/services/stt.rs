use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use tokio::task;
use anyhow::{Error, Result};
use tokio::sync::Semaphore;

use crate::models::whisper_model::{Whisper};
use crate::models::api_models::{TranscriptionResult, TranscribeParams, ResamplingMethod};
use crate::services::silero::VADService;
use crate::services::whisper::AudioBuffer;
use crate::services::resampler::ResamplingService;
use crate::services::audio_converter::{AudioConverterService, AudioFormat};

pub struct STTSession {
  audio_buffer: AudioBuffer,
  speech_buffer: AudioBuffer,
  is_speaking: bool,
  last_speech_end: Option<std::time::Instant>,
  transcription_history : Vec<TranscriptionResult>,
  vad_session_id: Uuid
}

pub struct STTService {
  whisper_model : Arc<Whisper>,
  vad_service: Arc<VADService>,
  resampling_service : Arc<ResamplingService>,
  audio_converter_service: Arc<AudioConverterService>,
  sessions: Arc<DashMap<Uuid, STTSession>>,
  config: STTConfig, 
  transcription_semaphore : Arc<Semaphore>
}

#[derive(Clone)]
#[allow(unused)]
pub struct STTConfig {
  pub sample_rate: u32,
  pub silence_duration_ms: u32,
  pub min_speech_duration_ms: u32,
  pub language: String,
  pub streaming_mode: bool,
  pub max_concurrent_transcriptions: usize,
}

impl STTSession {
  pub fn new(sample_rate: u32, vad_session_id: Uuid) -> Self {
    STTSession { audio_buffer: AudioBuffer::new(sample_rate, 30.0), 
      speech_buffer: AudioBuffer::new(sample_rate, 10.0), 
      is_speaking: false, 
      last_speech_end: None,
      transcription_history: Vec::new(), vad_session_id: vad_session_id 
    }
  }
}

impl Default for STTConfig {
  fn default() -> Self {
    STTConfig { sample_rate: 16000, 
      silence_duration_ms: 500,
      min_speech_duration_ms: 100, 
      language: "en".to_string(), 
      streaming_mode: true, max_concurrent_transcriptions: 5 }
  }
}

impl STTService {
  pub async fn new(model_path: &str,  vad_service : Arc<VADService>,config: STTConfig) -> Result<Self, Error> {
    let whisper_model = task::spawn_blocking({
      let model_path = model_path.to_string();
      move || Whisper::new(&model_path)
    }).await??;

    let transcription_semaphore = Arc::new(Semaphore::new(config.max_concurrent_transcriptions));

    Ok( STTService {
      whisper_model: Arc::new(whisper_model),
      vad_service: vad_service,
      resampling_service : Arc::new(ResamplingService::new(16000)),
      audio_converter_service : Arc::new(AudioConverterService::new()),
      sessions: Arc::new(DashMap::new()),
      config,
      transcription_semaphore: transcription_semaphore
    })
  }

  pub async fn create_session(&self, sample_rate: u32) -> Result<Uuid, Error> {
    let session_id = Uuid::new_v4(); 

    let vad_session_id = self.vad_service.create_session(180, 75).await?; 

    let stt_session = STTSession::new(sample_rate, vad_session_id); 

    self.sessions.insert(session_id, stt_session); 

    Ok(session_id)
  }


  pub async fn process_audio(&self, session_id: &Uuid, audio_chunk: &[u8], sample_rate: u32, audio_format: AudioFormat) -> Result<Option<TranscriptionResult>, Error> {
    log::info!("Enter the whisper process audio");
    log::info!("STT: Processing audio chunk of {} bytes with audio_format {} for session {}", audio_chunk.len(), audio_format._as_str() , session_id);

    let (resampled_bytes, final_audio_format) = if sample_rate != 16000 {
      log::info!("STT: Resampling from {}Hz to 16000Hz", sample_rate);
      self.resampling_service.resample_bytes(audio_chunk, sample_rate, Some(16000),Some(ResamplingMethod::Auto), audio_format, &self.audio_converter_service)?
    } else {
      (audio_chunk.to_vec(), audio_format)
    };

    log::info!("Final audio format {:?} and sample_rate {} for VAD and Whisper for audio process", final_audio_format, sample_rate);

    let samples = match self.audio_converter_service.bytes_to_samples(&resampled_bytes, final_audio_format) {
      Ok(samples) => samples,
      _ => Vec::new()
    };

    // let samples = self.bytes_to_samples(&resampled_bytes); 

    let (_vad_session_id, should_transcribe, speech_samples, _speech_detected) = {
      let mut session = self.sessions.get_mut(session_id).ok_or_else(|| anyhow::anyhow!("Session not found"))?;

      session.audio_buffer.add_samples(&samples);

      let vad_session_id = session.vad_session_id;
            
      // Check VAD using the associated VAD session
      let speech_detected = self.vad_service.process_audio(&vad_session_id, &resampled_bytes).await?;

      let mut should_transcribe = false;
      let mut speech_samples = None;
      
      if speech_detected && !session.is_speaking {
        // Speech started
        session.is_speaking = true;
        session.speech_buffer.clear();
        log::info!("Speech started for session {}", session_id);
      }


      if session.is_speaking {
        // Add to speech buffer
        session.speech_buffer.add_samples(&samples);
        
        if !speech_detected {
          // Speech might have ended
          if session.last_speech_end.is_none() {
            session.last_speech_end = Some(std::time::Instant::now());
          }
          
          // Check if silence duration exceeded threshold
          if let Some(last_end) = session.last_speech_end {
            let silence_duration = last_end.elapsed().as_millis() as u32;
            
            if silence_duration >= self.config.silence_duration_ms {
              // Check minimum speech duration
              let speech_duration_ms = (session.speech_buffer.duration_seconds() * 1000.0) as u32;
              
              if speech_duration_ms >= self.config.min_speech_duration_ms {
                speech_samples = Some(session.speech_buffer.get_samples());
                should_transcribe = true;
              }
              
              // Reset state
              session.is_speaking = false;
              session.last_speech_end = None;
              session.speech_buffer.clear();
              
              log::info!("Speech ended for session {}, duration: {}ms", session_id, speech_duration_ms);
            }
          }
          else {
            // Still speaking, reset silence timer
            session.last_speech_end = None;
          }
        }
      }

      (vad_session_id, should_transcribe, speech_samples, speech_detected)
    };


    if should_transcribe {
      if let Some(samples) = speech_samples {
        let result = self.transcribe_with_semaphore(&samples).await?;
        
        // Store in history
        if let Some(mut session) = self.sessions.get_mut(session_id) {
          session.transcription_history.push(result.clone());
        }
        
        return Ok(Some(result));
      }
    }
    
    Ok(None)


  }

  


  pub async fn transcribe_with_semaphore(&self, samples: &[f32] ) -> Result<TranscriptionResult, Error> {
    log::info!("Enter the transcribe with semphomore");
    let _permit = self.transcription_semaphore.acquire().await?;
        
    // Run transcription in blocking thread pool
    let whisper_model = self.whisper_model.clone();
    let samples = samples.to_vec();
    let language = self.config.language.clone();
    
    task::spawn_blocking(move || {
      let params = if true { // streaming mode
        TranscribeParams::for_streaming()
      } else {
        TranscribeParams {
          language: Some(language),
          ..Default::default()
        }
      };
        
      whisper_model.transcribe(&samples, &params)
    }).await?
  }
    
  /// Get session transcription history
  #[allow(unused)]
  pub fn get_transcription_history(&self, session_id: &Uuid) -> Result<Vec<TranscriptionResult>> {
    self.sessions.get(session_id).map(|session| session.transcription_history.clone()).ok_or_else(|| anyhow::anyhow!("Session not found"))
  }
    
  /// Force transcription of current speech buffer
  pub async fn force_transcribe(&self, session_id: &Uuid) -> Result<Option<TranscriptionResult>> {
    let samples = self.sessions.get(session_id).and_then(|session| {
      if session.speech_buffer.duration_seconds() > 0.0 {
        Some(session.speech_buffer.get_samples())
      } else {
        None
      }
    });
    
    if let Some(samples) = samples {
      let result = self.transcribe_with_semaphore(&samples).await?;
      Ok(Some(result))
    } else {
      Ok(None)
    }
  }
    
  /// Remove session
  pub fn remove_session(&self, session_id: &Uuid) -> bool {
    if let Some((_, session)) = self.sessions.remove(session_id) {
      // Also remove associated VAD session
      let _ = self.vad_service.remove_session(&session.vad_session_id);
      log::info!("Removed STT session: {}", session_id);
      true
    } else {
      false
    }
  }
    
  /// Get current statistics
  #[allow(unused)]
  pub async fn get_stats(&self) -> STTStats {
    STTStats {
      active_sessions: self.sessions.len(),
      active_transcriptions: self.config.max_concurrent_transcriptions - self.transcription_semaphore.available_permits(),
    }
  }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct STTStats {
  pub active_sessions: usize,
  pub active_transcriptions: usize,
}