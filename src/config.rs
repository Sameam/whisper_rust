use log; 
use dotenv;

#[derive(Clone)]
#[allow(unused)]
pub struct Config {}

impl Config {
  pub fn load() -> Self {
    match dotenv::dotenv() {
      Ok(_) => log::info!("Loaded .env file"),
      Err(_) => log::error!("No .env file found"),
    }

    Config {}
  }
}