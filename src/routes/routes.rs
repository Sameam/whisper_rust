use actix_web::{web, Error, HttpRequest, HttpResponse, Responder};
use serde_json;
use actix_web_actors::ws;
use std::sync::Arc;
use std::str::FromStr;

use crate::factory::AppState;
use crate::services::audio_converter::AudioFormat;
use crate::websockets::vad::VadWebSocketSession;
use crate::websockets::stt::STTWebSocketSession;


pub struct Routes;

impl Routes {

  #[allow(unused)]
  pub fn new() -> Self {
    Routes {}
  }

  pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/").route(web::get().to(Self::health))); 
    cfg.service(web::resource("/silence_detection").route(web::get().to(Self::silence_detection)));
    cfg.service(web::resource("/stt").route(web::get().to(Self::stt_streaming)));

  }

  async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
      "status": "ok",
      "Info": "Welcome to Rust Whisper STT", 
      "code": 200,
    }))
  }

  async fn silence_detection(req: HttpRequest, stream: web::Payload, app_data: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let vad_controller = Arc::new(app_data.vad_controller.clone());
    let ws = ws::start(VadWebSocketSession::new(vad_controller), &req, stream);
    ws
  }

  async fn stt_streaming(req: HttpRequest, stream: web::Payload, app_data: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let sample_rate = req.headers().get("sample_rate").and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<u32>().ok()).unwrap_or(16000);
    let audio_format_string = req.headers().get("audio_format").and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<String>().ok()).unwrap_or("LINEAR16".to_string());
    let stt_controller = Arc::new(app_data.stt_controller.clone());
    let audio_format = match AudioFormat::from_str(&audio_format_string) {
      Ok(audio_format) => audio_format, 
      _ => AudioFormat::PCM16 
    };
    let ws = ws::start(STTWebSocketSession::new(stt_controller, sample_rate, audio_format), &req, stream);
    ws
  }
}