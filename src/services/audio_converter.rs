use anyhow::{Error}; 
use wide::{f32x8}; 
use serde::{Serialize, Deserialize}; 
use std::sync::{Arc}; 
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum AudioFormat {
  #[serde(rename="LINEAR16")]
  PCM16,
  FLOAT32,
  PCM8,
  MULAW,
  ALAW,
  PCM24,
  PCM32,
}

impl AudioFormat { 

  pub fn _as_str(&self) -> &'static str {
    match self {
      &AudioFormat::PCM16   => "LINEAR16",
      &AudioFormat::FLOAT32 => "FLOAT32",
      &AudioFormat::PCM8    => "PCM8",
      &AudioFormat::PCM24   => "PCM24",
      &AudioFormat::PCM32   => "PCM32",
      &AudioFormat::ALAW    => "ALAW",
      &AudioFormat::MULAW   => "MULAW"
    }
  }
}



impl FromStr for AudioFormat {

  type Err = String; 

  fn from_str(audio_format: &str) -> Result<Self, Self::Err> {
    match audio_format.trim().to_uppercase().as_str() {
      "LINEAR16" => Ok(AudioFormat::PCM16),
      "FLOAT32"  => Ok(AudioFormat::FLOAT32),
      "PCM8"     => Ok(AudioFormat::PCM8),
      "PCM24"    => Ok(AudioFormat::PCM24),
      "PCM32"    => Ok(AudioFormat::PCM32),
      "ALAW"     => Ok(AudioFormat::ALAW),
      "MULAW"    => Ok(AudioFormat::MULAW),
      _          => Err(format!("Unknown audio formated: {}", audio_format)),
    }
  }
}

#[derive(Debug, Clone)]
pub struct AudioConverterConfig {
  /// Enable SIMD optimizations
  pub enable_simd: bool,
  /// Minimum samples to use SIMD
  pub simd_threshold: usize,
  /// Apply normalization after conversion
  pub enable_normalization: bool,
  /// DC offset removal threshold
  pub dc_offset_threshold: f32,
  /// Audio gain limits
  pub max_gain: f32,
  pub target_amplitude: f32,
  pub mulaw_noise_gate: i16,
  /// A-law specific settings
  pub alaw_noise_gate: i16,
}

#[derive(Clone, Debug)]
pub struct AudioConverterService {
  config: AudioConverterConfig,
  // pre computed look up table 
  mulaw_decode_table: Arc<[f32; 256]>,  // Direct f32 conversion
  alaw_decode_table: Arc<[f32; 256]>,   // Direct f32 conversion
}

impl Default for AudioFormat {
  fn default() -> Self {
    AudioFormat::PCM16
  }
}

impl AudioFormat {
  pub fn bytes_per_sample(&self) -> usize {
    match self {
      AudioFormat::MULAW | AudioFormat::PCM8 | AudioFormat::ALAW => 1,
      AudioFormat::PCM16 => 2, 
      AudioFormat::PCM24 => 3,
      AudioFormat::PCM32 | AudioFormat::FLOAT32 => 4
    }
  }

  // function to check if audio format supported 
  pub fn is_supported(&self) -> bool {
    matches!(self, AudioFormat::PCM16 | AudioFormat::MULAW | AudioFormat::ALAW | AudioFormat::FLOAT32)
  }
}

impl Default for AudioConverterConfig {
  fn default() -> Self {
    AudioConverterConfig { 
      enable_simd: Self::detect_simd_support(), 
      simd_threshold: 16, 
      enable_normalization: true, 
      dc_offset_threshold: 0.05, 
      max_gain: 2.0, 
      target_amplitude: 0.3,
      mulaw_noise_gate: 64,        // Suppress low-level quantization noise
      alaw_noise_gate: 32,  
    } // Suppress low-level quantization noise 
  }
}

impl AudioConverterConfig {
  fn detect_simd_support() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))] { 
      true 
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))] { 
      false 
    }
  }
}

unsafe impl Send for AudioConverterService {}
unsafe impl Sync for AudioConverterService {} 

impl AudioConverterService {
  pub fn new() -> Self {
    Self::new_with_config(AudioConverterConfig::default())
  }

  pub fn new_with_config(config: AudioConverterConfig) -> Self {
    // Pre-compute lookup tables with direct f32 conversion
    let mut mulaw_table = [0.0f32; 256];
    let mut alaw_table = [0.0f32; 256];
    
    for i in 0..256 {
      let mulaw_i16 = Self::decode_mulaw_static(i as u8);
      let alaw_i16 = Self::decode_alaw_static(i as u8);
      
      // Very aggressive noise gating for μ-law to prevent phantom audio
      mulaw_table[i] = if mulaw_i16.abs() < config.mulaw_noise_gate || mulaw_i16 == 0 {
        0.0
      } else {
        mulaw_i16 as f32 / 32768.0
      };
      
      alaw_table[i] = if alaw_i16.abs() < config.alaw_noise_gate {
        0.0
      } else {
        alaw_i16 as f32 / 32768.0
      };
    }
    
    AudioConverterService {
      config,
      mulaw_decode_table: Arc::new(mulaw_table),
      alaw_decode_table: Arc::new(alaw_table),
    }
  }



  // Main conversion function 
  pub fn bytes_to_samples(&self, bytes: &[u8], format: AudioFormat) -> Result<Vec<f32>, Error> {
    if !format.is_supported() {
      return Err(anyhow::anyhow!("Unsupported audio format: {:?}", format));
    }
    
    if bytes.is_empty() {
      return Ok(Vec::new());
    }


    let mut samples = match format {
      AudioFormat::PCM16 => self.convert_pcm16_to_f32(bytes)?,
      AudioFormat::FLOAT32 => self.convert_float32_to_f32(bytes)?, 
      AudioFormat::MULAW => self.convert_mulaw_to_f32(bytes)?,
      AudioFormat::ALAW => self.convert_alaw_to_f32(bytes)?,
      _ => return Err(anyhow::anyhow!("Format {:?} not yet implemented", format))
    };

    if self.config.enable_normalization {
      self.apply_normalization(&mut samples);
    }

    Ok(samples)
  }

  /// Convert any supported format to PCM16 bytes for resampling
  pub fn to_pcm16_bytes(&self, bytes: &[u8], format: AudioFormat) -> Result<Vec<u8>, Error> {
    if !format.is_supported() {
      return Err(anyhow::anyhow!("Unsupported audio format: {:?}", format));
    }
    
    if bytes.is_empty() {
      return Ok(Vec::new());
    }
    
    match format {
      AudioFormat::PCM16 => {
        Ok(bytes.to_vec())
      },
      
      AudioFormat::FLOAT32 => {
        if bytes.len() % 4 != 0 {
          return Err(anyhow::anyhow!("Float32 data length must be multiple of 4"));
        }
        
        let pcm16_bytes: Vec<u8> = bytes.chunks_exact(4)
          .flat_map(|chunk| {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            // Clamp to valid range and convert
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm_sample = (clamped * 32767.0) as i16;
            pcm_sample.to_le_bytes()
          })
          .collect();
        Ok(pcm16_bytes)
      },
      
      AudioFormat::MULAW => {
        let pcm16_bytes: Vec<u8> = bytes.iter().flat_map(|&byte| {
          let decoded = Self::decode_mulaw_static(byte);
          // Apply noise gating during PCM16 conversion too
          let gated = if decoded.abs() < self.config.mulaw_noise_gate {
            0i16
          } else {
            decoded
          };
          gated.to_le_bytes()
        })
        .collect();
    
        // Apply phantom audio suppression to PCM16 data
        Ok(self.suppress_phantom_audio_pcm16(pcm16_bytes))
      },
      
      AudioFormat::ALAW => {
        let pcm16_bytes: Vec<u8> = bytes.iter()
          .flat_map(|&byte| {
            let decoded = Self::decode_alaw_static(byte);
            // Apply noise gating here too
            let gated = if decoded.abs() < self.config.alaw_noise_gate {
              0i16
            } else {
              decoded
            };
            gated.to_le_bytes()
          })
          .collect();
        Ok(pcm16_bytes)
      },
      
      _ => Err(anyhow::anyhow!("Format {:?} not yet implemented for resampling", format))
    }
  }


  #[allow(unused)]
  pub fn get_config(&self) -> &AudioConverterConfig {
    &self.config
  }

  #[allow(unused)]
  pub fn with_config(&self, config: AudioConverterConfig) -> Self {
    // Need to regenerate lookup tables if noise gate settings changed
    Self::new_with_config(config)
  }


  // === PCM16 Conversion ===
  fn convert_pcm16_to_f32(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {

    if bytes.len() % 2 != 0 {
      return Err(anyhow::anyhow!("PCM16 data length must be even, got {} bytes", bytes.len()));
    }


    if self.config.enable_simd && bytes.len() >= self.config.simd_threshold * 2 {
      self.convert_pcm16_to_f32_simd(bytes)
    } else {
      self.convert_pcm16_to_f32_scalar(bytes)
    }
  }
    
  fn convert_pcm16_to_f32_scalar(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {
    Ok(bytes.chunks_exact(2)
      .map(|chunk| {
          let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
          sample as f32 / 32768.0
      })
      .collect())
  }
    
  fn convert_pcm16_to_f32_simd(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    let scale = f32x8::splat(1.0 / 32768.0);
    
    // Process 8 i16s at a time using SIMD
    let chunks = bytes.chunks_exact(16); // 8 samples * 2 bytes
    let remainder = chunks.remainder();
    
    for chunk in chunks {
      // Load 8 i16 values
      let i16_array: [i16; 8] = [
        i16::from_le_bytes([chunk[0], chunk[1]]),
        i16::from_le_bytes([chunk[2], chunk[3]]),
        i16::from_le_bytes([chunk[4], chunk[5]]),
        i16::from_le_bytes([chunk[6], chunk[7]]),
        i16::from_le_bytes([chunk[8], chunk[9]]),
        i16::from_le_bytes([chunk[10], chunk[11]]),
        i16::from_le_bytes([chunk[12], chunk[13]]),
        i16::from_le_bytes([chunk[14], chunk[15]]),
      ];
      
      let f32_array = [
        i16_array[0] as f32,
        i16_array[1] as f32,
        i16_array[2] as f32,
        i16_array[3] as f32,
        i16_array[4] as f32,
        i16_array[5] as f32,
        i16_array[6] as f32,
        i16_array[7] as f32,
      ];

      // Apply scaling using SIMD
      let f32_vec = f32x8::new(f32_array) * scale;
      
      // Store results
      let scaled_array = f32_vec.to_array();
      samples.extend_from_slice(&scaled_array);
    }

    for chunk in remainder.chunks_exact(2) {
      let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
      samples.push(sample as f32 / 32768.0);
    }
    
    Ok(samples)


  }


  // === Float32 Conversion ===
  fn convert_float32_to_f32(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {

    if bytes.len() % 4 != 0 {
      return Err(anyhow::anyhow!("Float32 data length must be multiple of 4, got {} bytes", bytes.len()));
    }

    let samples: Vec<f32> = bytes.chunks_exact(4)
      .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
      .collect();
    
    // Validate float values for safety in multithreaded context
    for (i, &sample) in samples.iter().enumerate() {
      if !sample.is_finite() {
        log::warn!("Non-finite float sample at index {}: {}", i, sample);
      }
      if sample.abs() > 2.0 {
        log::warn!("Float sample out of expected range at index {}: {}", i, sample);
      }
    }
    
    Ok(samples)

  }

  // === μ-law Conversion ===
  fn convert_mulaw_to_f32(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {
    // Use pre-computed lookup table with noise gating already applied
    let samples = bytes.iter()
      .map(|&byte| self.mulaw_decode_table[byte as usize])
      .collect();

    Ok(self.suppress_phantom_audio(samples))
  }

  // === A-law Conversion ===
  fn convert_alaw_to_f32(&self, bytes: &[u8]) -> Result<Vec<f32>, Error> {
    // Use pre-computed lookup table with noise gating already applied
    Ok(bytes.iter()
      .map(|&byte| self.alaw_decode_table[byte as usize])
      .collect())
  }


  /// Targeted phantom audio suppression - only remove clear artifacts at the end
  fn suppress_phantom_audio(&self, mut samples: Vec<f32>) -> Vec<f32> {
    if samples.len() < 80 { // Less than 10ms at 8kHz
      return samples;
    }
    
    // Only target the very end of the audio stream where phantom audio typically appears
    // Check progressively smaller tail sections
    let check_lengths = [80, 40, 20]; // 10ms, 5ms, 2.5ms at 8kHz
    
    for &check_len in &check_lengths {
      if samples.len() > check_len * 2 { // Ensure we have enough audio before the tail
        let tail_start = samples.len() - check_len;
        let tail = &samples[tail_start..];
        let main_section = &samples[..tail_start];
        
        // Calculate energy metrics
        let tail_rms = (tail.iter().map(|&s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
        let tail_max = tail.iter().fold(0.0f32, |max, &s| max.max(s.abs()));
        let main_rms = (main_section.iter().map(|&s| s * s).sum::<f32>() / main_section.len() as f32).sqrt();
        
        // Only remove if:
        // 1. Tail has very low energy compared to main content AND
        // 2. Tail energy is below absolute threshold AND  
        // 3. Main content has reasonable energy (to avoid removing all-quiet audio)
        let is_phantom = main_rms > 0.003 &&  // Main content has real energy
                        tail_rms < main_rms * 0.02 &&  // Tail is much quieter (50x)
                        tail_max < 0.008;  // Tail has very low peaks
        
        if is_phantom {
          // Zero out only this specific tail section
          for sample in &mut samples[tail_start..] {
            *sample = 0.0;
          }
          log::debug!("Removed phantom audio tail: {}ms (RMS: {:.6} vs main: {:.6})", check_len * 125 / 100, tail_rms, main_rms); // 125/100 converts samples to ms at 8kHz
          break; // Only remove one section
        }
      }
    }
    
    samples
  }


  // Suppress phantom audio in PCM16 byte data
  fn suppress_phantom_audio_pcm16(&self, mut pcm16_bytes: Vec<u8>) -> Vec<u8> {
    if pcm16_bytes.len() < 160 { // Less than 80 samples (10ms at 8kHz)
      return pcm16_bytes;
    }
    
    let sample_count = pcm16_bytes.len() / 2;
    if sample_count < 80 {
      return pcm16_bytes;
    }
    
    // Only check small tail sections for phantom audio
    let check_lengths = [40, 20, 10]; // 5ms, 2.5ms, 1.25ms at 8kHz
    
    for &check_samples in &check_lengths {
      if sample_count > check_samples * 2 { // Ensure enough main content
        let tail_start_sample = sample_count - check_samples;
        let tail_start_byte = tail_start_sample * 2;
        let main_end_byte = tail_start_byte;
        
        // Analyze tail section
        let mut tail_samples = Vec::new();
        for i in (tail_start_byte..pcm16_bytes.len()).step_by(2) {
          if i + 1 < pcm16_bytes.len() {
            let sample = i16::from_le_bytes([pcm16_bytes[i], pcm16_bytes[i + 1]]);
            tail_samples.push(sample);
          }
        }
        
        // Analyze main section (just before the tail)
        let main_check_samples = check_samples.min(tail_start_sample);
        let main_start_byte = (tail_start_sample - main_check_samples) * 2;
        let mut main_samples = Vec::new();
        for i in (main_start_byte..main_end_byte).step_by(2) {
          if i + 1 < pcm16_bytes.len() {
            let sample = i16::from_le_bytes([pcm16_bytes[i], pcm16_bytes[i + 1]]);
            main_samples.push(sample);
          }
        }
        
        if !tail_samples.is_empty() && !main_samples.is_empty() {
          let tail_rms = (tail_samples.iter().map(|&s| (s as f32) * (s as f32)).sum::<f32>() / tail_samples.len() as f32).sqrt();
          let tail_max = tail_samples.iter().fold(0i16, |max, &s| max.max(s.abs()));
          let main_rms = (main_samples.iter().map(|&s| (s as f32) * (s as f32)).sum::<f32>() / main_samples.len() as f32).sqrt();
          
          // Only remove if:
          // 1. Main content has reasonable energy (indicating real speech)
          // 2. Tail is much quieter than main content (indicating phantom audio)
          // 3. Tail has very low absolute energy
          let is_phantom = main_rms > 200.0 &&  // Main content has real energy
                          tail_rms < main_rms * 0.05 &&  // Tail is 20x quieter 0.05
                          (tail_max as f32) < 400.0;  // Tail has very low peaks
          
          if is_phantom {
            // Zero out only this specific tail section
            for i in (tail_start_byte..pcm16_bytes.len()).step_by(2) {
              if i + 1 < pcm16_bytes.len() {
                pcm16_bytes[i] = 0;
                pcm16_bytes[i + 1] = 0;
              }
            }
            log::debug!("Removed phantom PCM16 tail: {} samples (RMS: {:.1} vs main: {:.1})", check_samples, tail_rms, main_rms);
            break; // Only remove one section
          }
        }
      }
    }
    
    pcm16_bytes
  }


  // === Improved μ-law decoder with aggressive phantom audio suppression ===
  fn decode_mulaw_static(mulaw: u8) -> i16 {
    // Only filter the most obvious silence codes
    match mulaw {
      // Standard μ-law silence codes
      0x7F | 0xFF => return 0,
      _ => {}
    }
    
    // Standard ITU-T G.711 μ-law decoding
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 8159;
    
    let mulaw = (!mulaw) as i32;
    let sign = if (mulaw & 0x80) != 0 { -1 } else { 1 };
    let exponent = (mulaw >> 4) & 0x07;
    let mantissa = mulaw & 0x0F;
    
    // ITU-T G.711 standard calculation
    let magnitude = if exponent == 0 {
      (mantissa << 1) + 33
    } else {
      let shifted = (mantissa << 1) + 33;
      (shifted << exponent) - BIAS
    };
    
    let result = sign * magnitude.min(CLIP);
    result.clamp(-CLIP, CLIP) as i16
  }



  fn decode_alaw_static(alaw: u8) -> i16 {
    let alaw = alaw ^ 0x55;
    let sign = if (alaw & 0x80) != 0 { -1 } else { 1 };
    let exponent = (alaw >> 4) & 0x07;
    let mantissa = alaw & 0x0F;
    
    let magnitude = if exponent == 0 {
      ((mantissa as i16) << 1) + 1
    } else {
      let base = ((mantissa as i16) << 1) + 33;
      let shift = (exponent - 1) as usize;
      if shift <= 12 {  // Prevent overflow
        (base << shift).min(32767)
      } else {
        32767
      }
    };
    
    let result = sign * magnitude;
    result.clamp(-32767, 32767)
  }

  // === Normalization ===
  fn apply_normalization(&self, samples: &mut [f32]) {
    if samples.is_empty() {
      return;
    }
    
    // Calculate RMS instead of peak for more natural normalization
    let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    
    // Remove DC offset if significant
    let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
    if dc_offset.abs() > self.config.dc_offset_threshold {
      for sample in samples.iter_mut() {
        *sample -= dc_offset;
      }
    }
    
    // Apply gentle normalization based on RMS
    if rms > 0.001 && rms < 0.1 {
      let target_rms = self.config.target_amplitude * 0.3; // Conservative target
      let gain = (target_rms / rms).min(self.config.max_gain);
      
      if gain > 1.1 {
        for sample in samples.iter_mut() {
          *sample *= gain;
        }
      }
    }
  }

}












