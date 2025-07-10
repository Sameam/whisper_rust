use std::sync::{Arc};
use std::path::Path; 
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy}; 
use once_cell::sync::Lazy; 
use dashmap::DashMap; 
use anyhow::Error; 

use crate::models::api_models::{TranscribeParams, TranscriptionResult, TranscriptionSegment};

static LOADED_MODELS: Lazy<Arc<DashMap<String, Arc<WhisperContext>>>> = Lazy::new(|| {
  Arc::new(DashMap::new())
});

#[derive(Clone)]
#[allow(unused)]
pub struct Whisper {
  context: Arc<WhisperContext>,
  model_type: String,
}

impl Whisper {
  pub fn new(model_path: impl AsRef<Path>) -> Result<Self, Error> {
    let model_path: String = model_path.as_ref().to_string_lossy().to_string();

    let contexts =  if let Some(existing) = LOADED_MODELS.get(&model_path) {
      log::info!("Reusing existing Whisper model from cache");
      existing.clone()
    }
    else {
      log::info!("Loading Whisper model for the first time: {}", model_path);
      let ctx_params: WhisperContextParameters = WhisperContextParameters::default();

      let ctx: WhisperContext = WhisperContext::new_with_params(&model_path, ctx_params).map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;
          
      let context_arc = Arc::new(ctx);
      LOADED_MODELS.insert(model_path.clone(), context_arc.clone());
      context_arc
    };

    let model_type : String = Self::detect_model_type(&model_path);
    log::info!("Using whisper model_type: {}",model_type);
    Ok(Whisper { context: contexts, model_type: model_type })
  }

  pub fn transcribe(&self, sample: &[f32], params: &TranscribeParams ) -> Result<TranscriptionResult, Error> {
    let mut full_params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    let mut state = self.context.create_state().map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;

    let language = params.language.as_deref().unwrap_or("en");
    full_params.set_language(Some(language));

    full_params.set_n_threads(6);  // Use multiple threads (adjust based on CPU)
    
    // Configure parameters
    full_params.set_print_special(false);
    full_params.set_print_progress(false);
    full_params.set_print_realtime(false);
    full_params.set_print_timestamps(false);
    
    // Streaming optimizations
    full_params.set_single_segment(params.single_segment);
    full_params.set_max_tokens(params.max_tokens);
    full_params.set_no_context(params.no_context);
    
    // Quality settings
    full_params.set_suppress_blank(true);
    full_params.set_suppress_non_speech_tokens(true);
    full_params.set_temperature(0.0);


    full_params.set_translate(false);

    if sample.is_empty() {
      return Err(anyhow::anyhow!("Empty audio sample provided"));
    }
    
    if sample.len() < 480 {  // Minimum samples needed for processing
      log::warn!("Audio sample too short: {} samples", sample.len());
      return Ok(TranscriptionResult {
        text: String::new(),
        segments: vec![],
        language: params.language.clone(),
        duration: 0.0,
        confidence: Some(0.0),
      });
    }

    state.full(full_params, sample).map_err(|e| anyhow::anyhow!("Transcription failed: {}", e))?;

    let mut segments = Vec::new(); 

    let segment_count = state.full_n_segments().map_err(|e| anyhow::anyhow!("Failed to get segment count: {}", e))?;
        
    for i in 0..segment_count {
      let text = state.full_get_segment_text(i).map_err(|e| anyhow::anyhow!("Failed to get segment text: {}", e))?;
      
      let start = state.full_get_segment_t0(i).map_err(|e| anyhow::anyhow!("Failed to get segment start: {}", e))? as f32 / 100.0;
      
      let end = state.full_get_segment_t1(i).map_err(|e| anyhow::anyhow!("Failed to get segment end: {}", e))? as f32 / 100.0;
      

      segments.push(TranscriptionSegment {
        text,
        start: start,
        end: end,
        confidence: Some(0.0)
      });
    }

    let full_text = segments.iter().map(|s| s.text.trim()).collect::<Vec<_>>().join(" ");

    // Calculate duration based on sample count and assumed 16kHz sample rate
    // Note: Whisper internally expects 16kHz, so this calculation is correct
    let duration = if let Some(duration ) = segments.last().map(|s| s.end).or(Some(sample.len() as f32 / 16000.0)) {
      duration
    }
    else {
      0.0
    };


    log::info!("Whisper: Final transcription: '{}' with duration {}", full_text, duration);
    Ok(TranscriptionResult { text: full_text, segments: segments, language: params.language.clone(), duration: duration, confidence: Some(0.0) })
  }


  fn detect_model_type(model_path: &str) -> String {
    let path_lower = model_path.to_lowercase();
    
    if path_lower.contains("tiny") {
      "tiny".to_string()
    } else if path_lower.contains("base") {
      "base".to_string()
    } else if path_lower.contains("small") {
      "small".to_string()
    } else if path_lower.contains("medium") {
      "medium".to_string()
    } else if path_lower.contains("large") {
      "large".to_string()
    } else {
      "unknown".to_string()
    }
  }
}

