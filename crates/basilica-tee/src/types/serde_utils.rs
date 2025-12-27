//! Serialization utilities for TEE types.

use serde::{Deserialize, Deserializer, Serializer};

/// Helper module for hex serialization of byte vectors.
///
/// Use with `#[serde(with = "hex_bytes")]` on a `Vec<u8>` field.
pub mod hex_bytes {
    use super::*;

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

/// Helper module for optional hex serialization.
pub mod optional_hex_bytes {
    use super::*;

    pub fn serialize<S>(bytes: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match bytes {
            Some(b) => serializer.serialize_some(&hex::encode(b)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => hex::decode(&s).map(Some).map_err(serde::de::Error::custom),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct TestStruct {
        #[serde(with = "super::hex_bytes")]
        data: Vec<u8>,
    }

    #[test]
    fn test_hex_bytes_serialize() {
        let test = TestStruct {
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let json = serde_json::to_string(&test).unwrap();
        assert!(json.contains("deadbeef"));
    }

    #[test]
    fn test_hex_bytes_deserialize() {
        let json = r#"{"data":"deadbeef"}"#;
        let parsed: TestStruct = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }
}
