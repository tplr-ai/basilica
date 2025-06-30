use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::debug;

#[async_trait]
pub trait KeyValueStorage {
    async fn get_f64(&self, key: &str) -> Result<Option<f64>>;
    async fn set_f64(&self, key: &str, value: f64) -> Result<()>;
    async fn get_i64(&self, key: &str) -> Result<Option<i64>>;
    async fn set_i64(&self, key: &str, value: i64) -> Result<()>;
    async fn get_string(&self, key: &str) -> Result<Option<String>>;
    async fn set_string(&self, key: &str, value: &str) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<bool>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn increment(&self, key: &str) -> Result<i64>;
    async fn health_check(&self) -> Result<()>;
}

#[derive(Clone)]
pub struct MemoryStorage {
    data: Arc<RwLock<HashMap<String, String>>>,
    file_path: Option<PathBuf>,
}

impl MemoryStorage {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            file_path: None,
        })
    }

    pub async fn with_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file_path = path.into();
        let storage = Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            file_path: Some(file_path.clone()),
        };

        if file_path.exists() {
            storage.load_from_file().await?;
        }

        Ok(storage)
    }

    async fn load_from_file(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Ok(mut file) = File::open(path) {
                let mut contents = String::new();
                file.read_to_string(&mut contents)?;

                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&contents) {
                    *self.data.write().unwrap() = map;
                    debug!(
                        "Loaded {} entries from {:?}",
                        self.data.read().unwrap().len(),
                        path
                    );
                }
            }
        }
        Ok(())
    }

    async fn save_to_file(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let data = self.data.read().unwrap().clone();
            let json = serde_json::to_string_pretty(&data)?;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;

            file.write_all(json.as_bytes())?;
            file.sync_all()?;
            debug!("Saved {} entries to {:?}", data.len(), path);
        }
        Ok(())
    }

    fn parse_value<T>(&self, value: &str) -> Option<T>
    where
        T: std::str::FromStr,
    {
        value.parse().ok()
    }
}

#[async_trait]
impl KeyValueStorage for MemoryStorage {
    async fn get_f64(&self, key: &str) -> Result<Option<f64>> {
        let data = self.data.read().unwrap();
        Ok(data.get(key).and_then(|v| self.parse_value(v)))
    }

    async fn set_f64(&self, key: &str, value: f64) -> Result<()> {
        self.set_string(key, &value.to_string()).await
    }

    async fn get_i64(&self, key: &str) -> Result<Option<i64>> {
        let data = self.data.read().unwrap();
        Ok(data.get(key).and_then(|v| self.parse_value(v)))
    }

    async fn set_i64(&self, key: &str, value: i64) -> Result<()> {
        self.set_string(key, &value.to_string()).await
    }

    async fn get_string(&self, key: &str) -> Result<Option<String>> {
        let data = self.data.read().unwrap();
        Ok(data.get(key).cloned())
    }

    async fn set_string(&self, key: &str, value: &str) -> Result<()> {
        {
            let mut data = self.data.write().unwrap();
            data.insert(key.to_string(), value.to_string());
        }
        self.save_to_file().await?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let existed = {
            let mut data = self.data.write().unwrap();
            data.remove(key).is_some()
        };
        if existed {
            self.save_to_file().await?;
        }
        Ok(existed)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let data = self.data.read().unwrap();
        Ok(data.contains_key(key))
    }

    async fn increment(&self, key: &str) -> Result<i64> {
        let new_value = {
            let mut data = self.data.write().unwrap();
            let current = data
                .get(key)
                .and_then(|v| self.parse_value::<i64>(v))
                .unwrap_or(0);
            let new_value = current + 1;
            data.insert(key.to_string(), new_value.to_string());
            new_value
        };
        self.save_to_file().await?;
        Ok(new_value)
    }

    async fn health_check(&self) -> Result<()> {
        let _data = self.data.read().unwrap();
        debug!("Memory storage health check passed");
        Ok(())
    }
}
