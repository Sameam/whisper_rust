use crate::utils::utils;
use ndarray::{s, Array2, ArrayD, IxDyn};
use std::path::Path;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use dashmap::DashMap;


static LOADED_MODELS: Lazy<DashMap<String, Arc<Mutex<ort::session::Session>>>> = Lazy::new(|| {
  DashMap::new()
});

pub struct Silero {
  session: Arc<Mutex<ort::session::Session>>,
  sample_rate: ArrayD<i64>,
  state: ArrayD<f32>,
}

impl Silero {
  pub fn new(sample_rate_enum: utils::SampleRate, model_path: impl AsRef<Path>,) -> Result<Self, ort::Error> {
    let model_path = model_path.as_ref().to_string_lossy().to_string();
    // let models_map = LOADED_MODELS.clone();

    let session = {
      if let Some(existing_session) = LOADED_MODELS.get(&model_path) {
        log::info!("Reusing existing ONNX model from cache");
        existing_session.clone()
      } else {
        log::info!("Loading ONNX model for the first time: {}", model_path);
        let new_session = ort::session::Session::builder()?.commit_from_file(&model_path)?;
        let session_arc = Arc::new(Mutex::new(new_session));
        LOADED_MODELS.insert(model_path.clone(), session_arc.clone());
        session_arc
      }
    };
    
    log::info!("Using ONNX model (cached or newly loaded)");

    // Load the ONNX model
    // Initialize RNN state [2,1,128]
    let state: ArrayD<f32> = ArrayD::zeros(IxDyn(&[2,1,128]));
    // Prepare sample rate tensor [1]
    let sr_val: i64 = sample_rate_enum.into();
    let sample_rate: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&[1]), vec![sr_val]).unwrap();
    Ok( Silero { session, sample_rate, state})
  }

  /// Reset RNN state between streams
  pub fn reset(&mut self) {
    self.state.fill(0.0);
  }

  /// Compute speech probability on one 480-sample window
  pub fn calc_level(&mut self, audio_frame: &[i16]) -> Result<f32, ort::Error> {
    // Normalize to f32 in [-1,1] and truncate to 480 samples
    let mut data: Vec<f32> = audio_frame
      .iter()
      .map(|&x| x as f32 / i16::MAX as f32)
      .collect();

    data.resize(480, 0.0);            // if shorter, pad with zeros

    // Build the 1×480 tensor safely
    let mut frame: Array2<f32> = Array2::from_shape_vec((1, data.len()), data).unwrap();
    frame = frame.slice(s![.., ..480]).to_owned();

    // Build inputs using the inputs! macro
    let inps = ort::inputs![frame, std::mem::take(&mut self.state),self.sample_rate.clone(),]?;

    let session_guard = self.session.lock().map_err(|e| ort::Error::new(format!("Failed to lock session: {}", e)))?;

    // Run inference using the locked session guard.
    let res = session_guard.run(ort::session::SessionInputs::ValueSlice::<3>(&inps))?;

    // Update RNN state from 'stateN' output
    // extract a view and then clone into owned state
    let new_state_view = res.get("stateN").unwrap().try_extract_tensor()?;

    self.state = new_state_view.to_owned();

    // Pull probability from 'output' (shape [1])
    let raw: (&[i64], &[f32]) = res.get("output").unwrap().try_extract_raw_tensor::<f32>()?;
    let prob = raw.1[0];
    log::info!("Speech probability: {:.4}", prob);
    Ok(prob)
  }

}

impl Drop for Silero {
  fn drop(&mut self) {
    // Any per-instance cleanup
    self.reset();
  }
}