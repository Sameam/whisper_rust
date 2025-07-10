use std::collections::VecDeque;

use crate::{models::silero::Silero, utils::vad_iter::VadIter, utils::utils::VadParams, utils::utils::SampleRate};

pub struct Session {
  vad: VadIter,
  buffer : VecDeque<i16>,
  in_speech : bool
}

impl Session {
  pub fn new(model_path: &str, threshold: f32, sample_rate: u32, hangover_ms: u32, pad_ms: u32,) -> anyhow::Result<Self> {
    log::info!( "Creating new VAD session with threshold: {}, sample_rate: {}",threshold, sample_rate);

    let sample_rate_enum = match sample_rate {
      16000 => SampleRate::SixteenkHz,
      8000  => SampleRate::EightkHz,
      _ => panic!("Unsupported sample rate {}", sample_rate),
    };

    let silero: Silero = Silero::new(sample_rate_enum, model_path)?;
    
    log::info!("Model loaded successfully");

    let mut params : VadParams = VadParams::default();
    params.sample_rate = sample_rate as usize;
    params.threshold = threshold;
    params.frame_size = 32;  // 32ms windows -> 512 samples @16kHz
    params.min_speech_duration_ms = 50;
    params.min_silence_duration_ms = hangover_ms as usize;
    params.speech_pad_ms = pad_ms as usize;

    
    log::info!("VAD iterator created");

    let vad: VadIter = VadIter::new(silero, params.clone());
    let in_speech: bool = false;

    return Ok(Session { vad, buffer: VecDeque::new(), in_speech });
  }

  pub fn process_raw_bytes(&mut self, bytes: &[u8]) -> bool {
    log::info!("Processing raw bytes of size: {} bytes", bytes.len());

    let mut samples = Vec::with_capacity(bytes.len() / 2);

    for chunk in bytes.chunks_exact(2) {
      if chunk.len() == 2 {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample);
      }
    }

    log::info!("Converted {} bytes to {} i16 samples", bytes.len(), samples.len());

    self.streaming_chunk(&samples)
  }

  // function to save sample chunk into a wav file. 
  #[allow(unused)]
  fn save_samples_to_wav(samples: &[i16], filename: &str, sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(filename, spec)?;
    
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    
    writer.finalize()?;
    Ok(())
  }


  pub fn streaming_chunk(&mut self, chunk: &[i16]) -> bool {
    log::info!("Processing chunk of size: {}", chunk.len());
    
    if chunk.iter().all(|&s| s == 0) {
      // pure zero → definitely silence
      if self.in_speech { /* handle end of speech */ }
      return false;
    }

    const CHUNK_SIZE: usize = 480;
    const HOP_SIZE: usize = CHUNK_SIZE / 2;  // 50% overlap

    // 2) Append to internal buffer
    // self.buffer.extend_from_slice(chunk);
    self.buffer.extend(chunk.iter().cloned());

    // 3) Process as many full 512-sample frames as we can
    let mut speech_detected: bool = false;
    let mut idx: i32 = 0;
    

    while self.buffer.len() >= CHUNK_SIZE {
      // pull out exactly 512 samples
      let mut frame: [i16; CHUNK_SIZE]  = [0i16; CHUNK_SIZE];
      for i in 0..CHUNK_SIZE { 
        frame[i] = *self.buffer.get(i).unwrap();
      }
      
      match self.vad.process(&frame) {
        Ok(true)  => {
          log::info!("✅ Speech detected in frame {}", idx);
          if !self.in_speech {
            self.in_speech = true;
          }
          speech_detected = true;
        }
        Ok(false) => {
          log::info!("🔇 no speech in frame {}", idx);
          if self.in_speech {
            self.in_speech = false;
            self.reset();      // clear RNN state & buffer
          }
          speech_detected = false;
        }
        Err(e)    => {
          log::error!("⚠️ VAD error on frame {}: {:?}", idx, e);
          speech_detected = false;
        }
      };

      // 3c) Drop hop-size samples to get 50% overlap
      for _ in 0..HOP_SIZE {
        self.buffer.pop_front();
      }
      idx += 1;

    }

    log::info!("streaming_chunk returning {}", speech_detected);
    return speech_detected;

  }

  pub fn reset(&mut self) {
    self.vad.reset_states();
  }


}
