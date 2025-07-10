use anyhow::Result;
use actix_web::HttpServer;
use std::env; 
use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use env_logger::{Builder, Env};
use log::LevelFilter;
use chrono::Local;

use crate::factory::CreateApp;
use crate::config::Config;

mod models;
mod factory;
mod config;
mod routes;
mod services;
mod controllers;
mod utils;
mod websockets;

#[allow(unused)]
fn setup_logging() -> std::io::Result<()> {
  // Create logs directory
  std::fs::create_dir_all("logs")?;
  
  // Create log file with timestamp
  let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
  let log_file = format!("logs/stt_{}.log", timestamp);
  let file = File::create(log_file)?;
  
  // Configure logger with custom format
  let mut builder = Builder::from_env(Env::default().default_filter_or("debug"));
  builder.target(env_logger::Target::Pipe(Box::new(file))).filter_level(LevelFilter::Debug).format(|buf, record| {
    writeln!(
      buf,
      "{} [{}] - {}: {}",
      Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
      record.level(),
      record.target(),
      record.args()
    )
  }).init();
  
  log::info!("Logging initialized");
  Ok(())
}

#[actix_web::main]
async fn main() -> Result<()> {
  
  if env::var_os("RUST_LOG").is_none() {
    env::set_var("RUST_LOG", "actix_web=info,info"); // Default to info for actix_web and your app
  }
  env_logger::init();

  // setup_logging()?;

  dotenv::dotenv().ok();

  let config: Config = Config::load();

  let factory = CreateApp::new(config.clone()).await?;
  let factory = Arc::new(factory); // Wrap in Arc for sharing

  let server_builder = HttpServer::new(move || {
      // Clone the Arc for each worker
      let factory = factory.clone();
      
      // Use the already created factory
      factory.build_app().wrap(actix_web::middleware::Logger::default())
  });


  let server = server_builder.bind(("127.0.0.1", 8080))?;

  server.run().await?;

  Ok(())
}