//! VDF data structures and type definitions

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdfParameters {
    pub modulus: Vec<u8>,        // N = p * q (RSA modulus)
    pub generator: Vec<u8>,      // g ∈ Z*_N
    pub difficulty: u64,         // t = number of sequential squarings
    pub challenge_seed: Vec<u8>, // x = derived from previous attestation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdfProof {
    #[serde(
        serialize_with = "serialize_bytes_as_hex",
        deserialize_with = "deserialize_hex_as_bytes"
    )]
    pub output: Vec<u8>, // y = g^(2^t) mod N
    #[serde(
        serialize_with = "serialize_bytes_as_hex",
        deserialize_with = "deserialize_hex_as_bytes"
    )]
    pub proof: Vec<u8>, // π (proof of correct computation)
    pub computation_time_ms: u64,
    pub algorithm: VdfAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum VdfAlgorithm {
    Wesolowski,
    Pietrzak,
    #[default]
    SimpleSequential, // Simplified version for demonstration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdfChallenge {
    pub parameters: VdfParameters,
    pub expected_computation_time_ms: u64,
    pub max_allowed_time_ms: u64,
    pub min_required_time_ms: u64,
}

impl VdfParameters {
    pub fn new(
        modulus: Vec<u8>,
        generator: Vec<u8>,
        difficulty: u64,
        challenge_seed: Vec<u8>,
    ) -> Self {
        Self {
            modulus,
            generator,
            difficulty,
            challenge_seed,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.modulus.is_empty()
            && !self.generator.is_empty()
            && self.difficulty > 0
            && !self.challenge_seed.is_empty()
    }
}

impl VdfProof {
    pub fn new(
        output: Vec<u8>,
        proof: Vec<u8>,
        computation_time_ms: u64,
        algorithm: VdfAlgorithm,
    ) -> Self {
        Self {
            output,
            proof,
            computation_time_ms,
            algorithm,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.output.is_empty() && !self.proof.is_empty()
    }
}

impl VdfChallenge {
    pub fn new(
        parameters: VdfParameters,
        expected_computation_time_ms: u64,
        max_allowed_time_ms: u64,
        min_required_time_ms: u64,
    ) -> Self {
        Self {
            parameters,
            expected_computation_time_ms,
            max_allowed_time_ms,
            min_required_time_ms,
        }
    }

    pub fn is_within_time_bounds(&self, computation_time_ms: u64) -> bool {
        computation_time_ms >= self.min_required_time_ms
            && computation_time_ms <= self.max_allowed_time_ms
    }

    pub fn is_valid(&self) -> bool {
        self.parameters.is_valid()
            && self.min_required_time_ms <= self.expected_computation_time_ms
            && self.expected_computation_time_ms <= self.max_allowed_time_ms
    }
}

fn serialize_bytes_as_hex<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&hex::encode(bytes))
}

fn deserialize_hex_as_bytes<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let hex_str = String::deserialize(deserializer)?;
    hex::decode(&hex_str).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdf_parameters_validation() {
        let valid_params = VdfParameters::new(vec![1, 2, 3], vec![4, 5, 6], 100, vec![7, 8, 9]);
        assert!(valid_params.is_valid());

        let invalid_params = VdfParameters::new(vec![], vec![4, 5, 6], 100, vec![7, 8, 9]);
        assert!(!invalid_params.is_valid());
    }

    #[test]
    fn test_vdf_proof_validation() {
        let valid_proof =
            VdfProof::new(vec![1, 2, 3], vec![4, 5, 6], 1000, VdfAlgorithm::Wesolowski);
        assert!(valid_proof.is_valid());

        let invalid_proof = VdfProof::new(vec![], vec![4, 5, 6], 1000, VdfAlgorithm::Wesolowski);
        assert!(!invalid_proof.is_valid());
    }

    #[test]
    fn test_vdf_challenge_time_bounds() {
        let params = VdfParameters::new(vec![1, 2, 3], vec![4, 5, 6], 100, vec![7, 8, 9]);
        let challenge = VdfChallenge::new(params, 1000, 2000, 500);

        assert!(challenge.is_within_time_bounds(750));
        assert!(challenge.is_within_time_bounds(1500));
        assert!(!challenge.is_within_time_bounds(300));
        assert!(!challenge.is_within_time_bounds(2500));
    }

    #[test]
    fn test_vdf_challenge_validation() {
        let valid_params = VdfParameters::new(vec![1, 2, 3], vec![4, 5, 6], 100, vec![7, 8, 9]);
        let valid_challenge = VdfChallenge::new(valid_params, 1000, 2000, 500);
        assert!(valid_challenge.is_valid());

        let invalid_params = VdfParameters::new(vec![], vec![4, 5, 6], 100, vec![7, 8, 9]);
        let invalid_challenge = VdfChallenge::new(invalid_params, 1000, 2000, 500);
        assert!(!invalid_challenge.is_valid());
    }

    #[test]
    fn test_vdf_serialization() {
        let proof = VdfProof::new(
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            1000,
            VdfAlgorithm::Wesolowski,
        );

        let json = serde_json::to_string(&proof).unwrap();

        // Verify hex serialization
        assert!(json.contains("\"output\":\"01020304\""));
        assert!(json.contains("\"proof\":\"05060708\""));

        let deserialized: VdfProof = serde_json::from_str(&json).unwrap();

        assert_eq!(proof.output, deserialized.output);
        assert_eq!(proof.proof, deserialized.proof);
        assert_eq!(proof.computation_time_ms, deserialized.computation_time_ms);
    }
}
