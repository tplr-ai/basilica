//! TDX Quote parsing structures
//!
//! Implements parsing for Intel TDX Quote Version 4 structure.
//! Based on Intel TDX Module Architecture Specification.

use crate::error::{TeeError, TeeResult};
use std::convert::TryInto;

/// TDX Quote Version 4 structure
///
/// Based on Intel TDX Module Architecture Specification
#[derive(Debug, Clone)]
pub struct TdxQuoteV4 {
    /// Quote header (48 bytes)
    pub header: QuoteHeader,
    /// TD Report body (584 bytes)
    pub td_report: TdReport,
    /// Signature data length
    pub signature_data_len: u32,
    /// Signature data (variable length)
    pub signature_data: Vec<u8>,
}

/// Quote Header (48 bytes)
#[derive(Debug, Clone)]
pub struct QuoteHeader {
    /// Version of the quote structure (must be 4)
    pub version: u16,
    /// Attestation key type (2 = ECDSA-256-with-P-256)
    pub attestation_key_type: u16,
    /// TEE type (0x81 = TDX)
    pub tee_type: u32,
    /// Reserved bytes
    pub reserved: [u8; 2],
    /// QE Vendor ID (16 bytes)
    pub qe_vendor_id: [u8; 16],
    /// User data from Quote request (20 bytes)
    pub user_data: [u8; 20],
}

/// TD Report Body (584 bytes)
#[derive(Debug, Clone)]
pub struct TdReport {
    /// TEE TCB SVN (16 bytes)
    pub tee_tcb_svn: [u8; 16],
    /// MRSEAM (48 bytes)
    pub mr_seam: [u8; 48],
    /// MRSIGNER_SEAM (48 bytes)
    pub mr_signer_seam: [u8; 48],
    /// SEAM Attributes (8 bytes)
    pub seam_attributes: [u8; 8],
    /// TD Attributes (8 bytes)
    pub td_attributes: [u8; 8],
    /// XFAM (8 bytes)
    pub xfam: [u8; 8],
    /// MRTD - Build-time measurement of TD (48 bytes)
    pub mr_td: [u8; 48],
    /// MRCONFIGID (48 bytes)
    pub mr_config_id: [u8; 48],
    /// MROWNER (48 bytes)
    pub mr_owner: [u8; 48],
    /// MROWNERCONFIG (48 bytes)
    pub mr_owner_config: [u8; 48],
    /// RTMR[0] - Firmware/initrd measurements (48 bytes)
    pub rtmr0: [u8; 48],
    /// RTMR[1] - OS kernel measurements (48 bytes)
    pub rtmr1: [u8; 48],
    /// RTMR[2] - Application measurements (48 bytes)
    pub rtmr2: [u8; 48],
    /// RTMR[3] - Reserved (48 bytes)
    pub rtmr3: [u8; 48],
    /// Report data (64 bytes) - contains nonce + cert hash
    pub report_data: [u8; 64],
}

/// TDX TEE type constant
pub const TDX_TEE_TYPE: u32 = 0x81;

/// Quote version 4
pub const QUOTE_VERSION_4: u16 = 4;

/// Header size in bytes
pub const HEADER_SIZE: usize = 48;

/// TD Report size in bytes
pub const TD_REPORT_SIZE: usize = 584;

/// Minimum quote size (header + report + signature length field)
pub const MIN_QUOTE_SIZE: usize = HEADER_SIZE + TD_REPORT_SIZE + 4;

impl TdxQuoteV4 {
    /// Parse TDX quote from bytes
    pub fn parse(data: &[u8]) -> TeeResult<Self> {
        if data.len() < MIN_QUOTE_SIZE {
            return Err(TeeError::TdxQuoteParsing(format!(
                "Quote too short: {} bytes (minimum {})",
                data.len(),
                MIN_QUOTE_SIZE
            )));
        }

        let header = QuoteHeader::parse(&data[0..HEADER_SIZE])?;

        // Verify quote version
        if header.version != QUOTE_VERSION_4 {
            return Err(TeeError::TdxQuoteParsing(format!(
                "Unsupported quote version: {} (expected {})",
                header.version, QUOTE_VERSION_4
            )));
        }

        // Verify TEE type (TDX = 0x81)
        if header.tee_type != TDX_TEE_TYPE {
            return Err(TeeError::TdxQuoteParsing(format!(
                "Not a TDX quote, tee_type: 0x{:02x} (expected 0x{:02x})",
                header.tee_type, TDX_TEE_TYPE
            )));
        }

        let td_report = TdReport::parse(&data[HEADER_SIZE..HEADER_SIZE + TD_REPORT_SIZE])?;

        // Parse signature data length
        let sig_len_offset = HEADER_SIZE + TD_REPORT_SIZE;
        let signature_data_len = u32::from_le_bytes(
            data[sig_len_offset..sig_len_offset + 4]
                .try_into()
                .map_err(|_| {
                    TeeError::TdxQuoteParsing("Failed to parse signature length".into())
                })?,
        );

        // Extract signature data
        let sig_data_start = sig_len_offset + 4;
        let sig_data_end = sig_data_start + signature_data_len as usize;

        if data.len() < sig_data_end {
            return Err(TeeError::TdxQuoteParsing(
                "Quote truncated: signature data extends beyond buffer".into(),
            ));
        }

        let signature_data = data[sig_data_start..sig_data_end].to_vec();

        Ok(Self {
            header,
            td_report,
            signature_data_len,
            signature_data,
        })
    }

    /// Get the MRTD (build-time measurement)
    pub fn mrtd(&self) -> &[u8; 48] {
        &self.td_report.mr_td
    }

    /// Get all RTMRs as an array
    pub fn rtmrs(&self) -> [[u8; 48]; 4] {
        [
            self.td_report.rtmr0,
            self.td_report.rtmr1,
            self.td_report.rtmr2,
            self.td_report.rtmr3,
        ]
    }

    /// Get RTMR at specified index
    pub fn rtmr(&self, index: usize) -> Option<&[u8; 48]> {
        match index {
            0 => Some(&self.td_report.rtmr0),
            1 => Some(&self.td_report.rtmr1),
            2 => Some(&self.td_report.rtmr2),
            3 => Some(&self.td_report.rtmr3),
            _ => None,
        }
    }

    /// Get report data
    pub fn report_data(&self) -> &[u8; 64] {
        &self.td_report.report_data
    }

    /// Extract nonce from report data (first 32 bytes)
    pub fn nonce(&self) -> &[u8] {
        &self.td_report.report_data[0..32]
    }

    /// Extract cert hash from report data (bytes 32-64)
    pub fn cert_hash(&self) -> &[u8] {
        &self.td_report.report_data[32..64]
    }

    /// Verify that the quote contains the expected nonce
    pub fn verify_nonce(&self, expected_nonce: &[u8]) -> bool {
        if expected_nonce.len() > 32 {
            return false;
        }
        self.nonce()[..expected_nonce.len()] == *expected_nonce
    }

    /// Verify MRTD matches expected value
    pub fn verify_mrtd(&self, expected: &[u8; 48]) -> bool {
        self.td_report.mr_td == *expected
    }

    /// Get MRTD as hex string
    pub fn mrtd_hex(&self) -> String {
        hex::encode(self.mrtd())
    }

    /// Get quote as raw bytes for re-serialization
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MIN_QUOTE_SIZE + self.signature_data.len());
        bytes.extend_from_slice(&self.header.to_bytes());
        bytes.extend_from_slice(&self.td_report.to_bytes());
        bytes.extend_from_slice(&self.signature_data_len.to_le_bytes());
        bytes.extend_from_slice(&self.signature_data);
        bytes
    }
}

impl QuoteHeader {
    fn parse(data: &[u8]) -> TeeResult<Self> {
        if data.len() < HEADER_SIZE {
            return Err(TeeError::TdxQuoteParsing(format!(
                "Header data too short: {} bytes",
                data.len()
            )));
        }

        Ok(Self {
            version: u16::from_le_bytes(data[0..2].try_into().unwrap()),
            attestation_key_type: u16::from_le_bytes(data[2..4].try_into().unwrap()),
            tee_type: u32::from_le_bytes(data[4..8].try_into().unwrap()),
            reserved: data[8..10].try_into().unwrap(),
            qe_vendor_id: data[10..26].try_into().unwrap(),
            user_data: data[26..46].try_into().unwrap(),
        })
    }

    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..2].copy_from_slice(&self.version.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.attestation_key_type.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.tee_type.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.reserved);
        bytes[10..26].copy_from_slice(&self.qe_vendor_id);
        bytes[26..46].copy_from_slice(&self.user_data);
        bytes
    }
}

impl TdReport {
    fn parse(data: &[u8]) -> TeeResult<Self> {
        if data.len() < TD_REPORT_SIZE {
            return Err(TeeError::TdxQuoteParsing(format!(
                "TD Report data too short: {} bytes",
                data.len()
            )));
        }

        Ok(Self {
            tee_tcb_svn: data[0..16].try_into().unwrap(),
            mr_seam: data[16..64].try_into().unwrap(),
            mr_signer_seam: data[64..112].try_into().unwrap(),
            seam_attributes: data[112..120].try_into().unwrap(),
            td_attributes: data[120..128].try_into().unwrap(),
            xfam: data[128..136].try_into().unwrap(),
            mr_td: data[136..184].try_into().unwrap(),
            mr_config_id: data[184..232].try_into().unwrap(),
            mr_owner: data[232..280].try_into().unwrap(),
            mr_owner_config: data[280..328].try_into().unwrap(),
            rtmr0: data[328..376].try_into().unwrap(),
            rtmr1: data[376..424].try_into().unwrap(),
            rtmr2: data[424..472].try_into().unwrap(),
            rtmr3: data[472..520].try_into().unwrap(),
            report_data: data[520..584].try_into().unwrap(),
        })
    }

    fn to_bytes(&self) -> [u8; TD_REPORT_SIZE] {
        let mut bytes = [0u8; TD_REPORT_SIZE];
        bytes[0..16].copy_from_slice(&self.tee_tcb_svn);
        bytes[16..64].copy_from_slice(&self.mr_seam);
        bytes[64..112].copy_from_slice(&self.mr_signer_seam);
        bytes[112..120].copy_from_slice(&self.seam_attributes);
        bytes[120..128].copy_from_slice(&self.td_attributes);
        bytes[128..136].copy_from_slice(&self.xfam);
        bytes[136..184].copy_from_slice(&self.mr_td);
        bytes[184..232].copy_from_slice(&self.mr_config_id);
        bytes[232..280].copy_from_slice(&self.mr_owner);
        bytes[280..328].copy_from_slice(&self.mr_owner_config);
        bytes[328..376].copy_from_slice(&self.rtmr0);
        bytes[376..424].copy_from_slice(&self.rtmr1);
        bytes[424..472].copy_from_slice(&self.rtmr2);
        bytes[472..520].copy_from_slice(&self.rtmr3);
        bytes[520..584].copy_from_slice(&self.report_data);
        bytes
    }
}

/// Create a minimal valid TDX quote for testing (public for use in other test modules)
#[cfg(test)]
pub fn create_test_quote(mrtd: [u8; 48], rtmr0: [u8; 48], report_data: [u8; 64]) -> Vec<u8> {
    let mut quote = vec![0u8; MIN_QUOTE_SIZE + 100]; // Add some signature data

    // Header (48 bytes)
    // Version = 4
    quote[0..2].copy_from_slice(&4u16.to_le_bytes());
    // Attestation key type = 2 (ECDSA)
    quote[2..4].copy_from_slice(&2u16.to_le_bytes());
    // TEE type = 0x81 (TDX)
    quote[4..8].copy_from_slice(&0x81u32.to_le_bytes());
    // Reserved + QE Vendor ID + User Data = remaining 40 bytes (zeros)

    // TD Report starts at offset 48
    let report_offset = HEADER_SIZE;
    // TEE TCB SVN (16 bytes) - offset 0
    // MR_SEAM (48 bytes) - offset 16
    // MR_SIGNER_SEAM (48 bytes) - offset 64
    // SEAM Attributes (8 bytes) - offset 112
    // TD Attributes (8 bytes) - offset 120
    // XFAM (8 bytes) - offset 128
    // MRTD at offset 136
    quote[report_offset + 136..report_offset + 184].copy_from_slice(&mrtd);
    // MR_CONFIG_ID (48 bytes) - offset 184
    // MR_OWNER (48 bytes) - offset 232
    // MR_OWNER_CONFIG (48 bytes) - offset 280
    // RTMR0 at offset 328
    quote[report_offset + 328..report_offset + 376].copy_from_slice(&rtmr0);
    // RTMR1 (48 bytes) - offset 376
    // RTMR2 (48 bytes) - offset 424
    // RTMR3 (48 bytes) - offset 472
    // Report data at offset 520
    quote[report_offset + 520..report_offset + 584].copy_from_slice(&report_data);

    // Signature data length (after report)
    let sig_len_offset = HEADER_SIZE + TD_REPORT_SIZE;
    quote[sig_len_offset..sig_len_offset + 4].copy_from_slice(&100u32.to_le_bytes());

    quote
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_quote() {
        let mrtd = [0xAAu8; 48];
        let rtmr0 = [0xBBu8; 48];
        let report_data = [0xCCu8; 64];

        let quote_bytes = create_test_quote(mrtd, rtmr0, report_data);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        assert_eq!(quote.header.version, 4);
        assert_eq!(quote.header.tee_type, 0x81);
        assert_eq!(*quote.mrtd(), mrtd);
        assert_eq!(*quote.rtmr(0).unwrap(), rtmr0);
        assert_eq!(*quote.report_data(), report_data);
    }

    #[test]
    fn test_parse_quote_too_short() {
        let short_data = vec![0u8; 100];
        let result = TdxQuoteV4::parse(&short_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too short"));
    }

    #[test]
    fn test_parse_wrong_version() {
        let mut quote_bytes = create_test_quote([0u8; 48], [0u8; 48], [0u8; 64]);
        // Set version to 3
        quote_bytes[0..2].copy_from_slice(&3u16.to_le_bytes());

        let result = TdxQuoteV4::parse(&quote_bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported quote version"));
    }

    #[test]
    fn test_parse_wrong_tee_type() {
        let mut quote_bytes = create_test_quote([0u8; 48], [0u8; 48], [0u8; 64]);
        // Set TEE type to SGX (0x00)
        quote_bytes[4..8].copy_from_slice(&0x00u32.to_le_bytes());

        let result = TdxQuoteV4::parse(&quote_bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a TDX quote"));
    }

    #[test]
    fn test_verify_nonce() {
        let mut report_data = [0u8; 64];
        report_data[0..8].copy_from_slice(b"testnon0");

        let quote_bytes = create_test_quote([0u8; 48], [0u8; 48], report_data);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        assert!(quote.verify_nonce(b"testnon0"));
        assert!(!quote.verify_nonce(b"wrongnon"));
    }

    #[test]
    fn test_verify_mrtd() {
        let mrtd = [0x42u8; 48];
        let quote_bytes = create_test_quote(mrtd, [0u8; 48], [0u8; 64]);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        assert!(quote.verify_mrtd(&mrtd));
        assert!(!quote.verify_mrtd(&[0x00u8; 48]));
    }

    #[test]
    fn test_mrtd_hex() {
        let mrtd = [0xABu8; 48];
        let quote_bytes = create_test_quote(mrtd, [0u8; 48], [0u8; 64]);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        let hex = quote.mrtd_hex();
        assert_eq!(hex, "ab".repeat(48));
    }

    #[test]
    fn test_rtmrs() {
        let rtmr0 = [0x01u8; 48];
        let quote_bytes = create_test_quote([0u8; 48], rtmr0, [0u8; 64]);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        let rtmrs = quote.rtmrs();
        assert_eq!(rtmrs[0], rtmr0);
        assert_eq!(rtmrs[1], [0u8; 48]);
    }

    #[test]
    fn test_cert_hash() {
        let mut report_data = [0u8; 64];
        report_data[32..64].copy_from_slice(&[0xFFu8; 32]);

        let quote_bytes = create_test_quote([0u8; 48], [0u8; 48], report_data);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        assert_eq!(quote.cert_hash(), &[0xFFu8; 32]);
    }

    #[test]
    fn test_to_bytes_roundtrip() {
        let mrtd = [0xAAu8; 48];
        let quote_bytes = create_test_quote(mrtd, [0u8; 48], [0u8; 64]);
        let quote = TdxQuoteV4::parse(&quote_bytes).unwrap();

        let serialized = quote.to_bytes();
        let reparsed = TdxQuoteV4::parse(&serialized).unwrap();

        assert_eq!(*reparsed.mrtd(), mrtd);
        assert_eq!(reparsed.header.version, quote.header.version);
    }

    #[test]
    fn test_signature_data_truncated() {
        let mut quote_bytes = create_test_quote([0u8; 48], [0u8; 48], [0u8; 64]);
        // Set signature length to be larger than remaining data
        let sig_len_offset = HEADER_SIZE + TD_REPORT_SIZE;
        quote_bytes[sig_len_offset..sig_len_offset + 4].copy_from_slice(&10000u32.to_le_bytes());

        let result = TdxQuoteV4::parse(&quote_bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("truncated"));
    }
}
