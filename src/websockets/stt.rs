// src/websockets/stt.rs
use actix::{Actor, StreamHandler, Handler, Message, AsyncContext, WrapFuture, ActorContext, ActorFutureExt};
use actix_web_actors::ws;
use log::{error, info, debug};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::controllers::stt_controller::STTController;
use crate::models::api_models::{WebSocketResponse, WebSocketCommand, SessionConfig};
use crate::services::audio_converter::AudioFormat;

// Internal actor messages
#[derive(Message)]
#[rtype(result = "()")]
pub enum STTActorMessage {
  ProcessAudio(Vec<u8>),
  SendResponse(String),
  Close,
}

pub struct STTWebSocketSession {
  pub stt_controller: Arc<STTController>,
  pub session_id: Option<Uuid>,
  pub last_activity: Instant,
  pub audio_buffer: Vec<u8>,
  pub buffer_size: usize,
  pub config: SessionConfig,
}

impl Actor for STTWebSocketSession {
  type Context = ws::WebsocketContext<Self>;

  fn started(&mut self, ctx: &mut Self::Context) {
    info!("STT WebSocket connection established");
    
    // Create STT session using controller
    let stt_controller = self.stt_controller.clone();
    let sample_rate = if let Some(sample_rate)  = self.config.sample_rate {
      sample_rate
    }
    else {
      16000
    };
    ctx.spawn(
      async move { stt_controller.create_session(sample_rate).await }
      .into_actor(self)
      .map(|result, act, ctx| {
        match result {
          Ok(response) => {
            act.session_id = Some(response.session_id);
            
            // Send session created response
            let ws_response = WebSocketResponse::SessionCreated {
              session_id: response.session_id.to_string(),
              message: response.message,
            };
            
            if let Ok(json) = serde_json::to_string(&ws_response) {
              ctx.notify(STTActorMessage::SendResponse(json));
            }
          }
          Err(e) => {
            error!("Failed to create STT session: {}", e);
            let error_response = WebSocketResponse::Error {
              message: format!("Failed to create session: {}", e),
            };
            
            if let Ok(json) = serde_json::to_string(&error_response) {
              ctx.notify(STTActorMessage::SendResponse(json));
            }
            ctx.notify(STTActorMessage::Close);
          }
        }
      })
    );

    // Set up heartbeat
    self.heartbeat(ctx);
    
    // Set up session timeout
    ctx.run_interval(Duration::from_secs(30), |act, ctx| {
      if Instant::now().duration_since(act.last_activity) > Duration::from_secs(300) {
        info!("STT Session timeout");
        if let Some(session_id) = act.session_id {
          let response = WebSocketResponse::SessionEnded {
            session_id: session_id.to_string(),
            message: "Session timeout".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(STTActorMessage::SendResponse(json));
          }
        }
        ctx.notify(STTActorMessage::Close);
      }
    });
  }

  fn stopped(&mut self, _ctx: &mut Self::Context) {
    if let Some(session_id) = self.session_id {
      info!("Cleaning up STT session: {}", session_id);
      self.stt_controller.remove_session(&session_id);
    }
  }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for STTWebSocketSession {
  fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
    self.last_activity = Instant::now();

    match msg {
      Ok(ws::Message::Binary(data)) => {
        // Handle binary audio data
        if self.session_id.is_some() {
          // Buffer audio data for processing
          self.audio_buffer.extend_from_slice(&data);
          
          // Process when we have enough data
          if self.audio_buffer.len() >= self.buffer_size {
            let chunk = self.audio_buffer.drain(..self.buffer_size).collect();
            ctx.address().do_send(STTActorMessage::ProcessAudio(chunk));
          }
        } else {
          let error_response = WebSocketResponse::Error {
            message: "No active session".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&error_response) {
            ctx.notify(STTActorMessage::SendResponse(json));
          }
        }
      }
      Ok(ws::Message::Text(text)) => {
        // Handle WebSocket commands
        match serde_json::from_str::<WebSocketCommand>(&text) {
          Ok(command) => {
            self.handle_websocket_command(command, ctx);
          }
          Err(e) => {
            error!("Invalid WebSocket command: {}", e);
            let error_response = WebSocketResponse::Error {
              message: format!("Invalid command: {}", e),
            };
            
            if let Ok(json) = serde_json::to_string(&error_response) {
              ctx.notify(STTActorMessage::SendResponse(json));
            }
          }
        }
      }
      Ok(ws::Message::Ping(msg)) => {
        ctx.pong(&msg);
      }
      Ok(ws::Message::Pong(_)) => {
        // Pong received, client is alive
        debug!("Pong received from client");
      }
      Ok(ws::Message::Close(reason)) => {
        info!("WebSocket closing: {:?}", reason);
        if let Some(session_id) = self.session_id {
          let response = WebSocketResponse::SessionEnded {
            session_id: session_id.to_string(),
            message: "Connection closed".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(STTActorMessage::SendResponse(json));
          }
        }
        ctx.close(reason);
        ctx.notify(STTActorMessage::Close);
      }
      Err(e) => {
        error!("WebSocket error: {}", e);
        ctx.stop();
      }
      _ => {}
    }
  }
}

impl Handler<STTActorMessage> for STTWebSocketSession {
  type Result = ();

  fn handle(&mut self, msg: STTActorMessage, ctx: &mut Self::Context) -> Self::Result {
    match msg {
      STTActorMessage::ProcessAudio(audio_data) => {
        self.process_audio_chunk(audio_data, ctx);
      }
      STTActorMessage::SendResponse(response) => {
        ctx.text(response);
      }
      STTActorMessage::Close => {
        ctx.stop();
      }
    }
  }
}

impl STTWebSocketSession {
    pub fn new(stt_controller: Arc<STTController>, sample_rate: u32, audio_format: AudioFormat) -> Self {
      let default_config = SessionConfig {
        threshold: Some(0.5),
        sample_rate: Some(sample_rate),
        audio_format: Some(audio_format),
        language: Some("en".to_string()),
        silence_duration_ms: Some(1000), // How long to wait after speech stops
        min_speech_duration_ms: Some(1000), // Minimum speech to transcribe
        streaming_mode: Some(true),   
      };

      // Calculate buffer size for ~30ms frames at 16kHz
      let buffer_size = (16000 * 256 / 1000) * 2; // 2 bytes per sample (16-bit)

      STTWebSocketSession {
        stt_controller,
        session_id: None,
        last_activity: Instant::now(),
        audio_buffer: Vec::new(),
        buffer_size,
        config: default_config,
      }
    }

    // Heartbeat to keep connection alive
    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
      ctx.run_interval(Duration::from_secs(30), |act, ctx| {
        if Instant::now().duration_since(act.last_activity) > Duration::from_secs(60) {
          // Heartbeat timed out
          info!("Websocket heartbeat failed, disconnecting!");
          ctx.notify(STTActorMessage::Close);
          return;
        }
        
        ctx.ping(b"");
      });
    }

    fn handle_websocket_command(&mut self, command: WebSocketCommand, ctx: &mut ws::WebsocketContext<Self>) {
      match command {
        WebSocketCommand::EndSession => {
          if let Some(session_id) = self.session_id {
            self.stt_controller.remove_session(&session_id);
            
            let response = WebSocketResponse::SessionEnded {
              session_id: session_id.to_string(),
              message: "Session ended by client".to_string(),
            };
            
            if let Ok(json) = serde_json::to_string(&response) {
              ctx.notify(STTActorMessage::SendResponse(json));
            }
          }
          ctx.notify(STTActorMessage::Close);
        }
        WebSocketCommand::ResetSession => {
          if let Some(session_id) = self.session_id {
            // Reset the session (clear buffers)
            self.audio_buffer.clear();
            
            // Force transcribe any remaining audio
            let stt_controller = self.stt_controller.clone();
            let session_id_copy = session_id;
            
            ctx.spawn(
              async move { stt_controller.force_transcribe(&session_id_copy).await }
              .into_actor(self)
              .map(move |result, _, ctx| {
                match result {
                  Ok(Some(transcription)) => {
                    let response = WebSocketResponse::STTResult {
                      session_id: session_id.to_string(),
                      speech_detected: true,
                      transcription: Some(transcription.text),
                      processing_time_ms: 0,
                    };
                    
                    if let Ok(json) = serde_json::to_string(&response) {
                      ctx.notify(STTActorMessage::SendResponse(json));
                    }
                  }
                  Ok(None) => {
                      let response = WebSocketResponse::STTResult {
                        session_id: session_id.to_string(),
                        speech_detected: false,
                        transcription: None,
                        processing_time_ms: 0,
                      };
                      
                      if let Ok(json) = serde_json::to_string(&response) {
                        ctx.notify(STTActorMessage::SendResponse(json));
                      }
                  }
                  Err(e) => {
                    error!("Failed to force transcribe: {}", e);
                    let error_response = WebSocketResponse::Error {
                      message: format!("Failed to force transcribe: {}", e),
                    };
                    
                    if let Ok(json) = serde_json::to_string(&error_response) {
                      ctx.notify(STTActorMessage::SendResponse(json));
                    }
                  }
                }
              })
            );
          }
        }
        WebSocketCommand::Configure { config } => {
          self.config = config;
          
          // Update buffer size if sample rate changed
          if let Some(sample_rate) = self.config.sample_rate {
           
            self.buffer_size =  (sample_rate as usize * 256 / 1000) * 2; 
          }
          else {
            self.buffer_size = (16000 * 256 / 1000) * 2; // 2 bytes per sample (16-bit)
          }
          
          info!("STT configuration updated: {:?}", self.config);
        }
      }
    }

    fn process_audio_chunk(&self, audio_data: Vec<u8>, ctx: &mut ws::WebsocketContext<Self>) {
      if let Some(session_id) = self.session_id {
        let stt_controller = self.stt_controller.clone();
        let sample_rate = self.config.sample_rate.unwrap_or(16000);

        let audio_format = if let Some(audio_format) = self.config.audio_format {
          audio_format
        }
        else {
          AudioFormat::PCM16
        };
      
        ctx.spawn(
          async move {stt_controller.process_audio(session_id, &audio_data, sample_rate, audio_format ).await}
          .into_actor(self)
          .map(|result, _, ctx| {
            match result {
              Ok(stt_response) => {
                // Create WebSocket response
                let ws_response = WebSocketResponse::STTResult {
                  session_id: stt_response.session_id.to_string(),
                  speech_detected: stt_response.speech_detected,
                  transcription: stt_response.transcription.map(|t| t.text),
                  processing_time_ms: stt_response.processing_time_ms,
                };
                
                if let Ok(json) = serde_json::to_string(&ws_response) {
                  ctx.notify(STTActorMessage::SendResponse(json));
                }
              }
              Err(e) => {
                error!("STT processing error: {}", e);
                let error_response = WebSocketResponse::Error {
                  message: format!("STT processing failed: {}", e),
                };
                
                if let Ok(json) = serde_json::to_string(&error_response) {
                 ctx.notify(STTActorMessage::SendResponse(json));
                }
              }
            }
          })
        );
      }
    }
}
