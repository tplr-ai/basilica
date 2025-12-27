//! TEE-related error types

use thiserror::Error;

/// TEE-related errors
#[derive(Error, Debug)]
pub enum TeeError {
    #[error("TDX quote generation failed: {0}")]
    TdxQuoteGeneration(String),

    #[error("TDX quote parsing failed: {0}")]
    TdxQuoteParsing(String),

    #[error("TDX quote verification failed: {0}")]
    TdxQuoteVerification(String),

    #[error("MRTD mismatch: expected {expected}, got {actual}")]
    MrtdMismatch { expected: String, actual: String },

    #[error("RTMR mismatch at index {index}")]
    RtmrMismatch { index: usize },

    #[error("Nonce verification failed")]
    NonceVerificationFailed,

    #[error("GPU attestation failed: {0}")]
    GpuAttestation(String),

    #[error("GPU CC verification failed: {0}")]
    GpuCcVerification(String),

    #[error("GPU not in Confidential Compute mode")]
    GpuNotInCcMode,

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("NVML error: {0}")]
    Nvml(String),

    #[error("Certificate error: {0}")]
    Certificate(String),

    #[error("Command execution failed: {0}")]
    CommandExecution(String),

    #[error("Binary not found: {0}")]
    BinaryNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Hex decode error: {0}")]
    HexDecode(#[from] hex::FromHexError),
}

/// Result type for TEE operations
pub type TeeResult<T> = Result<T, TeeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TeeError::TdxQuoteGeneration("test error".to_string());
        assert_eq!(err.to_string(), "TDX quote generation failed: test error");
    }

    #[test]
    fn test_mrtd_mismatch_display() {
        let err = TeeError::MrtdMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "MRTD mismatch: expected abc123, got def456"
        );
    }

    #[test]
    fn test_rtmr_mismatch_display() {
        let err = TeeError::RtmrMismatch { index: 2 };
        assert_eq!(err.to_string(), "RTMR mismatch at index 2");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let tee_err: TeeError = io_err.into();
        assert!(matches!(tee_err, TeeError::Io(_)));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let tee_err: TeeError = json_err.into();
        assert!(matches!(tee_err, TeeError::Json(_)));
    }

    #[test]
    fn test_hex_error_conversion() {
        let hex_err = hex::decode("invalid hex!").unwrap_err();
        let tee_err: TeeError = hex_err.into();
        assert!(matches!(tee_err, TeeError::HexDecode(_)));
    }
}
