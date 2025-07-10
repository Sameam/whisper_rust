use crate::config::Config;
use crate::controllers::stt_controller::STTController;
use crate::{routes::routes::Routes, controllers::vad_controller::VADController};
use anyhow::{Error}; 
use actix_web::{web, App};

#[derive(Clone)]
pub struct AppState {
  pub vad_controller: VADController,
  pub stt_controller: STTController
}

impl AppState {

  #[allow(unused)]
  pub async fn new(app_config: &Config) -> Result<Self, Error> {
    let model_path = "models/silero_vad.onnx".to_string();
    let whisper_model_path = "models/ggml-base.bin".to_string();
    // let whisper_model_path = "models/ggml-distil-small.en.bin".to_string();
    
    let vad_controller = VADController::new(&model_path).await?;
    let stt_controller = STTController::new(&whisper_model_path, &model_path).await?;

    Ok(AppState { vad_controller, stt_controller})
  }

}

#[allow(unused)]
pub struct CreateApp {
  app_state: AppState,
  app_settings: Config,
}

impl CreateApp {
    #[allow(unused)]
    pub async fn new(app_settings: Config) -> Result<Self, Error> {
      let app_state = AppState::new(&app_settings).await?;
      Ok(CreateApp { app_state, app_settings  })
    }

    pub fn build_app(&self,) -> App<impl actix_web::dev::ServiceFactory<actix_web::dev::ServiceRequest,Config = (),Response = actix_web::dev::ServiceResponse<impl actix_web::body::MessageBody>,Error = actix_web::Error,InitError = (),>,> {
      App::new()
      .app_data(web::Data::new(self.app_state.clone()))
      .configure(Routes::configure)
    }
}