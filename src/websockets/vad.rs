// src/websockets/vad.rs
use actix::{Actor, StreamHandler, Handler, Message, AsyncContext, WrapFuture, ActorContext, ActorFutureExt};
use actix_web_actors::ws;
use log::{error, info};
use uuid::Uuid;
use std::sync::Arc;
use std::time::Instant;

use crate::controllers::vad_controller::VADController;
use crate::models::api_models::{
  SessionConfig, WebSocketCommand, WebSocketResponse
};
use crate::services::audio_converter::AudioFormat;

// Internal actor messages
#[derive(Message)]
#[rtype(result = "()")]
pub enum VadActorMessage {
  ProcessAudio(Vec<u8>),
  SendResponse(String),
  Close,
}

pub struct VadWebSocketSession {
  pub vad_controller: Arc<VADController>,
  pub session_id: Option<Uuid>,
  pub last_activity: Instant,
  pub audio_buffer: Vec<u8>,
  pub buffer_size: usize,
  pub config: SessionConfig,
}

impl Actor for VadWebSocketSession {
  type Context = ws::WebsocketContext<Self>;

  fn started(&mut self, ctx: &mut Self::Context) {
    info!("VAD WebSocket connection established");
    
    // Create VAD session using your existing controller
    let vad_controller = self.vad_controller.clone();
    ctx.spawn(
      async move { vad_controller.create_session(96, 50).await }.into_actor(self).map(|result, act, ctx| {
        match result {
          Ok(response) => {
            act.session_id = Some(response.session_id);
            
            // Send session created response using your WebSocketResponse enum
            let ws_response = WebSocketResponse::SessionCreated {
              session_id: response.session_id.to_string(),
              message: response.message,
            };
            
            if let Ok(json) = serde_json::to_string(&ws_response) {
              ctx.notify(VadActorMessage::SendResponse(json));
            }
          }
          Err(e) => {
            error!("Failed to create VAD session: {}", e);
            let error_response = WebSocketResponse::Error {
              message: format!("Failed to create session: {}", e),
            };
            
            if let Ok(json) = serde_json::to_string(&error_response) {
              ctx.notify(VadActorMessage::SendResponse(json));
            }
            ctx.notify(VadActorMessage::Close);
          }
        }
      })
    );

      // Set up session timeout
    ctx.run_interval(std::time::Duration::from_secs(30), |act, ctx| {
      if Instant::now().duration_since(act.last_activity) > std::time::Duration::from_secs(300) {
        info!("VAD Session timeout");
        if let Some(session_id) = act.session_id {
          let response = WebSocketResponse::SessionEnded {
            session_id: session_id.to_string(),
            message: "Session timeout".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(VadActorMessage::SendResponse(json));
          }
        }
        ctx.notify(VadActorMessage::Close);
      }
    });
  }

  fn stopped(&mut self, _ctx: &mut Self::Context) {
    if let Some(session_id) = self.session_id {
      info!("Cleaning up VAD session: {}", session_id);
      let _ = self.vad_controller.remove_session(&session_id);
    }
  }
}

impl Handler<VadActorMessage> for VadWebSocketSession {
  type Result = ();

  fn handle(&mut self, msg: VadActorMessage, ctx: &mut Self::Context) -> Self::Result {
    match msg {
      VadActorMessage::ProcessAudio(audio_data) => {
        self.process_audio_chunk(audio_data, ctx);
      }
      VadActorMessage::SendResponse(response) => {
        ctx.text(response);
      }
      VadActorMessage::Close => {
        ctx.stop();
      }
    }
  }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for VadWebSocketSession {
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
            ctx.address().do_send(VadActorMessage::ProcessAudio(chunk));
          }
        } else {
          let error_response = WebSocketResponse::Error {
            message: "No active session".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&error_response) {
            ctx.notify(VadActorMessage::SendResponse(json));
          }
        }
      }
      Ok(ws::Message::Text(text)) => {
          // Handle WebSocket commands using your existing models
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
                ctx.notify(VadActorMessage::SendResponse(json));
              }
            }
        }
      }
      Ok(ws::Message::Close(reason)) => {
        info!("WebSocket closing: {:?}", reason);
        if let Some(session_id) = self.session_id {
          let response = WebSocketResponse::SessionEnded {
            session_id: session_id.to_string(),
            message: "Connection closed".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(VadActorMessage::SendResponse(json));
          }
        }
        ctx.close(reason);
        ctx.notify(VadActorMessage::Close);
      }
      Ok(ws::Message::Ping(msg)) => {
        ctx.pong(&msg);
      }
      Err(e) => {
        error!("WebSocket error: {}", e);
        ctx.notify(VadActorMessage::Close);
      }
      _ => {}
    }
  }
}

impl VadWebSocketSession {
  pub fn new(vad_controller: Arc<VADController>) -> Self {
    let default_config = SessionConfig {
      threshold: Some(0.5),
      sample_rate: Some(16000),
      audio_format : Some(AudioFormat::PCM16),
      language: Some("en".to_string()),
      min_speech_duration_ms: Some(0),
      silence_duration_ms: Some(0),
      streaming_mode : Some(false)
    };

    // Calculate buffer size for ~256ms frames at 16kHz
    let buffer_size = (16000 * 256 / 1000) * 2; // 2 bytes per sample (16-bit)

    Self {
      vad_controller,
      session_id: None,
      last_activity: Instant::now(),
      audio_buffer: Vec::new(),
      buffer_size,
      config: default_config,
    }
  }

  fn handle_websocket_command(&mut self, command: WebSocketCommand, ctx: &mut ws::WebsocketContext<Self>) {
    match command {
      WebSocketCommand::EndSession => {
        if let Some(session_id) = self.session_id {
          let _ = self.vad_controller.remove_session(&session_id);
          
          let response = WebSocketResponse::SessionEnded {
            session_id: session_id.to_string(),
            message: "Session ended by client".to_string(),
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(VadActorMessage::SendResponse(json));
          }
        }
        ctx.stop();
      }
      WebSocketCommand::ResetSession => {
        if let Some(session_id) = self.session_id {
          // Reset the session (clear buffers, reset state)
          self.audio_buffer.clear();
          
          let response = WebSocketResponse::VadResult {
            session_id: session_id.to_string(),
            speech_detected: false,
          };
          
          if let Ok(json) = serde_json::to_string(&response) {
            ctx.notify(VadActorMessage::SendResponse(json));
          }
        }
      }
      WebSocketCommand::Configure { config } => {
        self.config = config;
        
        // Update buffer size if sample rate changed
        if let Some(sample_rate) = self.config.sample_rate {
          self.buffer_size = (sample_rate as usize * 256 / 1000) * 2;
        }
        else {
          self.buffer_size = (16000 * 256 / 1000) * 2;
        }
        
        info!("VAD configuration updated: {:?}", self.config);
      }
    }
  }

  fn process_audio_chunk(&self, audio_data: Vec<u8>, ctx: &mut ws::WebsocketContext<Self>) {
    if let Some(session_id) = self.session_id {
      let vad_controller = self.vad_controller.clone();
      
      ctx.spawn(
        async move {
            vad_controller.process_audio(&session_id, &audio_data).await
        }
        .into_actor(self)
        .map(|result, _act, ctx| {
          match result {
            Ok(vad_response) => {
              // Use your existing VadResponse model
              let ws_response = WebSocketResponse::VadResult {
                session_id: vad_response.session_id.to_string(),
                speech_detected: vad_response.speech_detected,
              };
              
              if let Ok(json) = serde_json::to_string(&ws_response) {
                ctx.notify(VadActorMessage::SendResponse(json));
              }
            }
            Err(e) => {
              error!("VAD processing error: {}", e);
              let error_response = WebSocketResponse::Error {
                message: format!("VAD processing failed: {}", e),
            };
              
              if let Ok(json) = serde_json::to_string(&error_response) {
                ctx.notify(VadActorMessage::SendResponse(json));
              }
            }
          }
        })
      );
    }
  }
}