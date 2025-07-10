use std::collections::VecDeque;


#[derive(Debug)]
pub struct AudioBuffer {
  samples: VecDeque<f32>,
  sample_rate: u32,
  max_duration_seconds: f32
}

impl AudioBuffer {
  pub fn new(sample_rate: u32, max_duration_second: f32) -> Self {
    AudioBuffer { samples: VecDeque::new(), sample_rate: sample_rate , max_duration_seconds: max_duration_second}
  }

  pub fn add_samples(&mut self, samples: &[f32]) {
    self.samples.extend(samples);

    let max_samples = (self.sample_rate as f32 * self.max_duration_seconds) as usize;

    while self.samples.len() > max_samples {
      self.samples.pop_front();
    }

    log::debug!("AudioBuffer: Added {} samples, total: {}, duration: {:.2}s", samples.len(), self.samples.len(),self.duration_seconds());
  }

  pub fn get_samples(&self) -> Vec<f32> {
    self.samples.iter().copied().collect()
  }

  pub fn clear(&mut self) {
    self.samples.clear()
  }

  pub fn duration_seconds(&self) -> f32 {
    self.samples.len() as f32 / self.sample_rate as f32 
  }


}