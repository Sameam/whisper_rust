use anyhow::{Error}; 
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, FftFixedIn}; 
use dasp::{signal, signal::Signal, signal::interpolate::Converter, interpolate::linear::Linear};
use std::sync::{Arc, Mutex}; 
use dashmap::DashMap; 
use wide::{f32x8, i16x16};

use crate::{models::api_models::ResamplingMethod, services::audio_converter::{AudioConverterService, AudioFormat}};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResamplerKey {
  source_rate: u32,
  target_rate: u32,
}

pub struct ResamplingService {
  sinc_resamplers: Arc<DashMap<ResamplerKey, Arc<Mutex<SincFixedIn<f64>>>>>,
  fft_resamplers: Arc<DashMap<ResamplerKey, Arc<Mutex<FftFixedIn<f64>>>>>,
  target_sample_rate: u32,
  default_method : ResamplingMethod
}

/// Helper function to correctly widen an i16x16 into two i32x8 vectors.
#[inline(always)]
fn i16x16_to_f32x8s(vec: i16x16) -> Result<(f32x8, f32x8), Error> {
  let array = vec.to_array(); // -> [i16; 16]

  let mut lo_f32 = [0.0f32; 8];
  let mut hi_f32 = [0.0f32; 8];

  for i in 0..8 {
    lo_f32[i] = array[i] as f32;
    hi_f32[i] = array[i + 8] as f32;
  }

  Ok((f32x8::new(lo_f32), f32x8::new(hi_f32)))
}

/// Helper to take two f32x8 vectors, round, clamp, convert to i16, and store in the output.
#[inline(always)]
fn store_f32x8s_as_i16s(even_f: f32x8, odd_f: f32x8, output: &mut Vec<i16>) -> Result<(), Error> {
  let even_arr = even_f.round().to_array(); // -> [f32; 8]
  let odd_arr = odd_f.round().to_array();   // -> [f32; 8]

  for i in 0..8 {
    // Clamp the f32 value within the i16 range before casting
    let even_clamped = even_arr[i].max(i16::MIN as f32).min(i16::MAX as f32);
    let odd_clamped = odd_arr[i].max(i16::MIN as f32).min(i16::MAX as f32);

    output.push(even_clamped as i16);
    output.push(odd_clamped as i16);
  }

  Ok(())
}


impl ResamplingService {
  pub fn new(target_sample_rate: u32) -> Self {
    ResamplingService { sinc_resamplers: Arc::new(DashMap::new()), 
      fft_resamplers : Arc::new(DashMap::new()),
      target_sample_rate: target_sample_rate,
      default_method : ResamplingMethod::Auto }
  }

  #[allow(unused)]
  pub fn new_methods(target_sample_rate: u32, method: ResamplingMethod ) -> Self {
    ResamplingService { sinc_resamplers: Arc::new(DashMap::new()), 
      fft_resamplers : Arc::new(DashMap::new()),
      target_sample_rate: target_sample_rate,
      default_method : method
    }
  }

  #[allow(unused)]
  pub fn set_default_method(&mut self, method: ResamplingMethod) {
    self.default_method = method
  }

  /// Resample raw audio bytes (assumed to be 16-bit PCM) from source_rate to target_rate
  pub fn resample_bytes(&self, bytes: &[u8], source_rate: u32, target_rate: Option<u32>, method : Option<ResamplingMethod>, audio_format: AudioFormat, audio_converter: &AudioConverterService) -> Result<(Vec<u8>, AudioFormat), Error> {
    let target_rate = target_rate.unwrap_or(16000);
    let method : ResamplingMethod = method.unwrap_or(self.default_method);

    let expected_bytes_per_sample = audio_format.bytes_per_sample();
    if bytes.len() % expected_bytes_per_sample != 0 {
      return Err(anyhow::anyhow!("Audio bytes length must be multiple of {} for {:?} format (got {})", expected_bytes_per_sample, audio_format, bytes.len()));
    }

    if source_rate == target_rate {
      return Ok((bytes.to_vec(), audio_format));
    }

    let i16_bytes = audio_converter.to_pcm16_bytes(bytes, audio_format)?;

    // Convert bytes to i16 samples
    let mut i16_samples = Vec::with_capacity(i16_bytes.len() / 2);
    for chunk in i16_bytes.chunks_exact(2) {
      let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
      i16_samples.push(sample);
    }

    let result = self.resample_i16(&i16_samples, source_rate, Some(target_rate), Some(method))?;

    let output_bytes: Vec<u8> = result.iter().flat_map(|&sample| sample.to_le_bytes()).collect();     
    Ok((output_bytes, AudioFormat::PCM16))
  }

  pub fn resample_i16(&self, audio_chunk: &[i16], source_rate: u32, target_rate: Option<u32>, method : Option<ResamplingMethod>) -> Result<Vec<i16>, Error> {
    let target_rate = target_rate.unwrap_or(16000); 
    let method : ResamplingMethod = method.unwrap_or(self.default_method);

    if source_rate == target_rate {
      return Ok(audio_chunk.to_vec());
    }

    let method  = match method {
      ResamplingMethod::Auto => self.determine_best_method(source_rate, target_rate),
      _other => Ok(ResamplingMethod::Dasp)
    }?;

    let use_simd = self.should_use_simd(audio_chunk.len())?;

    match method {
      ResamplingMethod::Custom => {
        if source_rate == 8000 && target_rate == 16000 && use_simd {
          log::info!("Enter the custom resampling");
          let result = self.resample_i16_8k_to_16k_simd(audio_chunk, source_rate, Some(target_rate))?;
          return Ok(result);
        }
        else {
          log::info!("Enter the preemphasis resampling");
          let result = self.resample_i16_8k_to_16k_custom(audio_chunk, source_rate, Some(target_rate))?;
          return Ok(result);
        }
      },

      ResamplingMethod::Dasp => {
        log::info!("Enter the dasp resampling");
        let result = self.resample_i16_with_preemphasis(audio_chunk, source_rate, target_rate)?;
        return Ok(result);
      },

      ResamplingMethod::Rubato => {
        log::info!("Enter the rubato resampling");
        let result = self.resample_i16_with_rubato(audio_chunk, source_rate, target_rate)?;
        return Ok(result);
      },

      ResamplingMethod::Auto => unreachable!(),
    }

  }

   #[allow(unused)]
  pub fn resample_f32(&self, audio_chunk: &[f32], source_sample_rate: u32, target_sample_rate: Option<u32>, method: Option<ResamplingMethod>) -> Result<Vec<f32>, Error> {
    let target_rate = target_sample_rate.unwrap_or(000); 
    let method = method.unwrap_or(self.default_method);

    if source_sample_rate == target_rate {
      log::info!("No resampling needed {}Hz -> {}Hz", source_sample_rate, target_rate);
      return Ok(audio_chunk.to_vec());
    }

    let method  = match method {
      ResamplingMethod::Auto => self.determine_best_method(source_sample_rate, target_rate),
      _other => Ok(ResamplingMethod::Dasp)
    }?;

    let use_simd = self.should_use_simd(audio_chunk.len())?;

    match method {
      ResamplingMethod::Custom => {
        if source_sample_rate == 8000 && target_rate == 16000 && use_simd {
          let result = self.resample_f32_8k_to_16k_simd(audio_chunk, source_sample_rate, Some(target_rate))?;
          return Ok(result);
        }
        else {
          let result = self.resample_f32_8k_to_16k_custom(audio_chunk, source_sample_rate, Some(target_rate))?; 
          return Ok(result);
        }
      },

      ResamplingMethod::Dasp => {
        let result = self.resample_f32_with_dasp(audio_chunk, source_sample_rate, target_rate)?; 
        return Ok(result);
      },

      ResamplingMethod::Rubato => {
        let result = self.resample_f32_with_rubato(audio_chunk, source_sample_rate, target_rate)?;
        return Ok(result);
      },

      ResamplingMethod::Auto => unreachable!(),
    }
  }



  // CUSTOM FUNCTION IMPLEMENTATION
  fn resample_i16_8k_to_16k_custom(&self, input: &[i16], source_rate: u32, target_rate: Option<u32>) -> Result<Vec<i16>, Error> {
    // Your existing custom implementation for i16
    let _source_rate = source_rate; 
    let _target_rate = target_rate;
    let mut samples_16k = Vec::with_capacity(input.len() * 2);
    
    for i in 0..input.len() {
      // Current sample
      samples_16k.push(input[i]);
      
      // Interpolated sample
      if i < input.len() - 1 {
        // Linear interpolation between current and next sample
        let interpolated = (input[i] as i32 + input[i + 1] as i32) / 2;
        samples_16k.push(interpolated as i16);
      } else {
        // Last sample: just duplicate
        samples_16k.push(input[i]);
      }
    }
    
    // Apply simple lowpass filter to smooth the result
    let mut filtered = vec![0i16; samples_16k.len()];
    
    // 3-tap filter [0.25, 0.5, 0.25]
    for i in 1..samples_16k.len() - 1 {
      let sum = (samples_16k[i - 1] as i32) / 4 + (samples_16k[i] as i32) / 2 + (samples_16k[i + 1] as i32) / 4;
      filtered[i] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    filtered[0] = samples_16k[0];
    filtered[samples_16k.len() - 1] = samples_16k[samples_16k.len() - 1];
    
    Ok(filtered)
  }

  fn resample_i16_8k_to_16k_simd(&self, input: &[i16], source_rate: u32, target_rate: Option<u32>) -> Result<Vec<i16>, Error> {
    // Fallback for small inputs where the SIMD setup isn't worth it.
    if input.len() < 18 { // Main loop requires a window of 18 (16+2)
      // You can call your non-SIMD optimized function here
      return Ok(self.resample_i16_8k_to_16k_custom(input, source_rate, target_rate)?);
    }

    let mut output = Vec::with_capacity(input.len() * 2);

    // --- Scalar Pre-loop ---
    output.push(input[0]);
    output.push(((input[0] as f32 + input[1] as f32) * 0.5).round() as i16);
    let mut i = 1;

    // --- SIMD Core Loop ---
    // Define SIMD floating-point constants
    let c_0_125 = f32x8::splat(0.125);
    let c_0_75 = f32x8::splat(0.75);
    let c_0_5 = f32x8::splat(0.5);

    // Process chunks of 16 i16s at a time.
    while i + 16 + 1 < input.len() {
      // 1. LOAD a wide chunk of i16s
      let prev_i16 = i16x16::new(input[i-1..i-1+16].try_into().unwrap());
      let curr_i16 = i16x16::new(input[i..i+16].try_into().unwrap());
      let next_i16 = i16x16::new(input[i+1..i+1+16].try_into().unwrap());
      
      // Cast the i32 vectors to f32 vectors
      let (prev_lo, prev_hi) = i16x16_to_f32x8s(prev_i16)?;
      let (curr_lo, curr_hi) = i16x16_to_f32x8s(curr_i16)?;
      let (next_lo, next_hi) = i16x16_to_f32x8s(next_i16)?;

      let prev = [prev_lo, prev_hi];
      let curr = [curr_lo, curr_hi];
      let next = [next_lo, next_hi];

      for j in 0..2 { // Process the low and high parts
        // 3. COMPUTE using f32 SIMD
        let even_f = prev[j] * c_0_125 + curr[j] * c_0_75 + next[j] * c_0_125;
        let odd_f = curr[j] * c_0_5 + next[j] * c_0_5;
        let _ = store_f32x8s_as_i16s(even_f, odd_f, &mut output);
      }

      i += 16;
    }

    // --- Scalar Post-loop for the remainder ---
    while i < input.len() -1 {
      let prev = input[i - 1] as f32;
      let curr = input[i] as f32;
      let next = input[i + 1] as f32;
      let even_sample = prev * 0.125 + curr * 0.75 + next * 0.125;
      let odd_sample = curr * 0.5 + next * 0.5;
      output.push(even_sample.round() as i16);
      output.push(odd_sample.round() as i16);
      i += 1;
    }
    
    let last = *input.last().unwrap();
    output.push(last);
    output.push(last);

    Ok(output)
  }

  fn resample_f32_8k_to_16k_simd(&self, input: &[f32], source_rate: u32, target_rate: Option<u32>) -> Result<Vec<f32>, Error> {
    if input.len() < 18 { // Main loop requires a window of 18 (16+2)
      // You can call your non-SIMD optimized function here
      return Ok(self.resample_f32_8k_to_16k_custom(input, source_rate, target_rate)?);
    }

    let mut output = Vec::with_capacity(input.len() * 2);

    output.push(input[0]);
    if input.len() > 1 {
        output.push((input[0] + input[1]) * 0.5);
    } else {
        output.push(input[0]);
        return Ok(output);
    }

    let mut i = 1;

    // SIMD constants
    let c_0_125 = f32x8::splat(0.125);
    let c_0_75 = f32x8::splat(0.75);
    let c_0_5 = f32x8::splat(0.5);

    // SIMD main loop - process 8 f32s at a time
    while i + 8 + 1 < input.len() {
      // Bounds check
      if i < 1 || i + 8 >= input.len() {
        break;
      }

      // Load 8 consecutive f32 samples for prev, curr, and next
      let prev_slice = &input[i-1..i-1+8];
      let curr_slice = &input[i..i+8];
      let next_slice = &input[i+1..i+1+8];

      // Convert slices to arrays for SIMD
      let prev_array: [f32; 8] = prev_slice.try_into().map_err(|_| anyhow::anyhow!("Failed to convert prev slice to array"))?;
      let curr_array: [f32; 8] = curr_slice.try_into().map_err(|_| anyhow::anyhow!("Failed to convert curr slice to array"))?;
      let next_array: [f32; 8] = next_slice.try_into().map_err(|_| anyhow::anyhow!("Failed to convert next slice to array"))?;

      // Load into SIMD vectors
      let prev_f32 = f32x8::new(prev_array);
      let curr_f32 = f32x8::new(curr_array);
      let next_f32 = f32x8::new(next_array);

      // Compute filtered samples using SIMD
      // Even samples: apply 3-tap filter [0.125, 0.75, 0.125]
      let even_f = prev_f32 * c_0_125 + curr_f32 * c_0_75 + next_f32 * c_0_125;
      
      // Odd samples: linear interpolation [0.5, 0.5]
      let odd_f = curr_f32 * c_0_5 + next_f32 * c_0_5;

      // Store results - interleave even and odd samples
      let even_array = even_f.to_array();
      let odd_array = odd_f.to_array();

      for j in 0..8 {
        output.push(even_array[j]);  // Even sample (filtered)
        output.push(odd_array[j]);   // Odd sample (interpolated)
      }

      i += 8;
    }

    // Scalar post-loop for remaining samples
    while i < input.len().saturating_sub(1) {
      let prev = input[i - 1];
      let curr = input[i];
      let next = input[i + 1];
      
      // Apply same filter as SIMD version
      let even_sample = prev * 0.125 + curr * 0.75 + next * 0.125;
      let odd_sample = curr * 0.5 + next * 0.5;
      
      output.push(even_sample);
      output.push(odd_sample);
      i += 1;
    }
    
    // Handle last sample
    if let Some(&last) = input.last() {
      output.push(last);
      output.push(last);
    }

    log::debug!("SIMD f32 resampling: {} -> {} samples", input.len(), output.len());
    Ok(output)
  }

  fn should_use_simd(&self, audio_len: usize) -> Result<bool, Error> {

    if audio_len < 18 {
      return Ok(false);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))] {
      return Ok(true);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))] {
        // Other architectures - use scalar
      return Ok(false);
    }

  }


  #[allow(unused)]
  fn resample_f32_8k_to_16k_custom(&self, input: &[f32], source_rate: u32, target_rate: Option<u32>) -> Result<Vec<f32>, Error> {
    // Convert to i16, use the i16 implementation, then convert back to f32
    let i16_input: Vec<i16> = input.iter().map(|&s| (s * i16::MAX as f32) as i16).collect();
    
    let i16_output = self.resample_i16_8k_to_16k_custom(&i16_input, source_rate, target_rate)?;
    
    let f32_output: Vec<f32> = i16_output.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
    
    Ok(f32_output)
  }


  // IMPLEMENTATION FOR DASP

  pub fn resample_i16_dasp(&self,input: &[i16], source_rate: u32, target_rate: u32) -> Result<Vec<i16>, Error> {
    // Create a signal from the input samples
    let source_signal = signal::from_iter(input.iter().map(|&sample| [sample as f32]));
    let interpolator = Linear::new([0.0], [0.0]);
    // Create a signal that interpolates at 0.5 sample increments (2x upsampling)
    let converter = Converter::from_hz_to_hz(
      source_signal,
      interpolator,
      source_rate as f64,
      target_rate as f64,
    );
    
    // Collect the first output_len samples
    Ok(converter.take(input.len() * 2).map(|frame| frame[0] as i16).collect())
  }

  pub fn resample_i16_with_preemphasis(&self, input: &[i16], source_rate: u32, target_rate: u32) -> Result<Vec<i16>, Error> {
    // Apply pre-emphasis filter (y[n] = x[n] - 0.95*x[n-1])
    let mut emphasized = Vec::with_capacity(input.len());
    let pre_factor = 0.95;
    
    for i in 0..input.len() {
      let current = input[i] as f32;
      let previous = if i > 0 { input[i-1] as f32 } else { 0.0 };
      let emphasized_sample = current - pre_factor * previous;
      emphasized.push(emphasized_sample as i16);
    }
    
    // Now resample the pre-emphasized signal
    Ok(self.resample_i16_dasp(&emphasized, source_rate, target_rate)?)
  }

   #[allow(unused)]
  fn resample_f32_with_dasp(&self, input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>, Error> {
    // Create a signal from the input samples
    let source_signal = signal::from_iter(input.iter().map(|&sample| [sample as f32]));
    let interpolator = Linear::new([0.0], [0.0]);
    // Create a signal that interpolates at 0.5 sample increments (2x upsampling)
    let converter = Converter::from_hz_to_hz(
      source_signal,
      interpolator,
      source_rate as f64,
      target_rate as f64,
    );
    
    // Collect the samples
    Ok(converter.take(input.len() * 2).map(|frame| frame[0]).collect())
  }


  // === RUBATO IMPLEMENTATION ===
   #[allow(unused)]
  fn get_resampler(&self, source_rate: u32, target_rate: u32) -> Result<Arc<Mutex<SincFixedIn<f64>>>, Error> {
    let key = ResamplerKey { source_rate, target_rate };
    
    // Check if we already have a resampler for this combination
    if let Some(resampler) = self.sinc_resamplers.get(&key) {
      return Ok(resampler.clone());
    }
    
    // Create new resampler with high quality settings
    let params = if source_rate < target_rate {
      // General upsampling parameters
      SincInterpolationParameters {
        sinc_len: 8,  // Short filter for upsampling
        f_cutoff: 0.45 * (source_rate as f32 / target_rate as f32).min(1.0),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 16,
        window: WindowFunction::Hann,
      }
    } else {
      // Downsampling - use higher quality
      SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
      }
    };
    
    let ratio = target_rate as f64 / source_rate as f64;
    
    log::debug!("Creating new resampler for {}Hz -> {}Hz (ratio: {})", source_rate, target_rate, ratio);
    
    let resampler = SincFixedIn::<f64>::new(
      ratio,
      2.0,
      params,
      1024,
      1
    )?;
    
    let resampler_arc = Arc::new(Mutex::new(resampler));
    self.sinc_resamplers.insert(key, resampler_arc.clone());
    
    Ok(resampler_arc)
  }


   #[allow(unused)]
  fn resample_i16_with_rubato(&self, input: &[i16], source_rate: u32, target_rate: u32) -> Result<Vec<i16>, Error> {
    // Convert i16 samples to f64 for resampling
    let audio_f64: Vec<f64> = input.iter().map(|&s| s as f64 / i16::MAX as f64).collect();
    
    // Get or create resampler for this rate combination
    let resampler_mutex = self.get_resampler(source_rate, target_rate)?;
    
    // Lock the mutex to get mutable access
    let mut resampler = resampler_mutex.lock().map_err(|e| anyhow::anyhow!("Failed to lock resampler: {}", e))?;
    
    // Prepare input/output buffers
    let frames = audio_f64.len();
    let mut input_frames = vec![audio_f64];
    
    // Calculate output size
    let output_frames = resampler.output_frames_next();
    let mut output_buffer = vec![vec![0.0; output_frames]; 1];
    
    // Perform resampling
    resampler.process_into_buffer(&input_frames, &mut output_buffer, None)?;
    
    // Convert back to i16
    let resampled: Vec<i16> = output_buffer[0].iter()
        .map(|&s| (s * i16::MAX as f64).round() as i16)
        .collect();
    
    Ok(resampled)
  }


  #[allow(unused)]
  fn resample_f32_with_rubato(&self, input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>, Error> {
    // Convert f32 samples to f64 for resampling
    let audio_f64: Vec<f64> = input.iter().map(|&s| s as f64).collect();
    
    // Get or create resampler for this rate combination
    let resampler_mutex = self.get_resampler(source_rate, target_rate)?;
    
    // Lock the mutex to get mutable access
    let mut resampler = resampler_mutex.lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock resampler: {}", e))?;
    
    // Prepare input/output buffers
    let frames = audio_f64.len();
    let mut input_frames = vec![audio_f64];
    
    // Calculate output size
    let output_frames = resampler.output_frames_next();
    let mut output_buffer = vec![vec![0.0; output_frames]; 1];
    
    // Perform resampling
    resampler.process_into_buffer(&input_frames, &mut output_buffer, None)?;
    
    // Convert back to f32
    let resampled: Vec<f32> = output_buffer[0].iter().map(|&s| s as f32).collect();
    
    Ok(resampled)
  }


  fn determine_best_method(&self, source_rate: u32, target_rate: u32) -> Result<ResamplingMethod, Error> {
    if source_rate == 8000 && target_rate == 16000 {
      return Ok(ResamplingMethod::Custom);
    }

    // For common speech sample rates, use DASP
    if (source_rate <= 24000 && target_rate <= 24000) && 
        (source_rate == 8000 || source_rate == 16000 || source_rate == 24000) &&
        (target_rate == 8000 || target_rate == 16000 || target_rate == 24000) {
        return Ok(ResamplingMethod::Dasp);
    }
    
    // For higher quality audio or music, use Rubato
    if source_rate >= 44100 || target_rate >= 44100 {
        return Ok(ResamplingMethod::Rubato);
    } 

    Ok(ResamplingMethod::Dasp)
  
  }

  /// Clear the resampler cache
  #[allow(unused)]
  pub fn clear_cache(&self) {
    log::info!("Clearing resampler cache ({} Sinc, {} FFT entries)", self.sinc_resamplers.len(),self.fft_resamplers.len());
    self.sinc_resamplers.clear();
    self.fft_resamplers.clear();
  }
  
  #[allow(unused)]
  pub fn cache_size(&self) -> (usize, usize) {
    (self.sinc_resamplers.len(), self.fft_resamplers.len())
  }
  
  /// Set the default target sample rate
   #[allow(unused)]
  pub fn set_default_target_rate(&mut self, rate: u32) {
    log::info!("Setting default target rate to {}Hz", rate);
    self.target_sample_rate = rate;
  }
}

