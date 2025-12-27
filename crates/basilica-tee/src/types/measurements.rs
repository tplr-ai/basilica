//! TDX Measurement Types
//!
//! Types for representing and comparing TDX measurements (MRTD, RTMRs).

/// Expected measurements for TDX verification.
///
/// Used to verify that a TDX quote contains the expected measurements.
/// Any measurement set to `None` will match any value (permissive mode).
#[derive(Debug, Clone, Default)]
pub struct ExpectedMeasurements {
    /// MRTD - Build-time measurement of TD (48 bytes)
    pub mrtd: Option<[u8; 48]>,
    /// RTMR[0] - Firmware/initrd measurements
    pub rtmr0: Option<[u8; 48]>,
    /// RTMR[1] - OS kernel measurements
    pub rtmr1: Option<[u8; 48]>,
    /// RTMR[2] - Application measurements
    pub rtmr2: Option<[u8; 48]>,
    /// RTMR[3] - Reserved
    pub rtmr3: Option<[u8; 48]>,
}

impl ExpectedMeasurements {
    /// Create new expected measurements from config.
    pub fn from_config(config: &crate::config::TdxConfig) -> Self {
        Self {
            mrtd: config.expected_mrtd_bytes(),
            rtmr0: config.expected_rtmr_bytes(0),
            rtmr1: config.expected_rtmr_bytes(1),
            rtmr2: config.expected_rtmr_bytes(2),
            rtmr3: config.expected_rtmr_bytes(3),
        }
    }

    /// Check if MRTD matches expected value.
    ///
    /// Returns `true` if no expected MRTD is set or if it matches.
    pub fn matches_mrtd(&self, mrtd: &[u8; 48]) -> bool {
        self.mrtd.as_ref().is_none_or(|expected| expected == mrtd)
    }

    /// Check if RTMR at index matches expected value.
    ///
    /// Returns `true` if no expected RTMR is set for the index,
    /// if the index is out of range, or if it matches.
    pub fn matches_rtmr(&self, index: usize, rtmr: &[u8; 48]) -> bool {
        let expected = match index {
            0 => &self.rtmr0,
            1 => &self.rtmr1,
            2 => &self.rtmr2,
            3 => &self.rtmr3,
            _ => return true,
        };
        expected.as_ref().is_none_or(|e| e == rtmr)
    }

    /// Check if all measurements match.
    pub fn matches_all(&self, mrtd: &[u8; 48], rtmrs: &[[u8; 48]; 4]) -> bool {
        self.matches_mrtd(mrtd)
            && self.matches_rtmr(0, &rtmrs[0])
            && self.matches_rtmr(1, &rtmrs[1])
            && self.matches_rtmr(2, &rtmrs[2])
            && self.matches_rtmr(3, &rtmrs[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_mrtd() {
        let expected = ExpectedMeasurements {
            mrtd: Some([0x01u8; 48]),
            ..Default::default()
        };

        assert!(expected.matches_mrtd(&[0x01u8; 48]));
        assert!(!expected.matches_mrtd(&[0x02u8; 48]));
    }

    #[test]
    fn test_empty_matches_any() {
        let expected = ExpectedMeasurements::default();

        assert!(expected.matches_mrtd(&[0x01u8; 48]));
        assert!(expected.matches_mrtd(&[0x00u8; 48]));
        assert!(expected.matches_rtmr(0, &[0xFFu8; 48]));
    }

    #[test]
    fn test_rtmr_matching() {
        let expected = ExpectedMeasurements {
            rtmr0: Some([0xAAu8; 48]),
            rtmr1: Some([0xBBu8; 48]),
            ..Default::default()
        };

        assert!(expected.matches_rtmr(0, &[0xAAu8; 48]));
        assert!(!expected.matches_rtmr(0, &[0x00u8; 48]));
        assert!(expected.matches_rtmr(1, &[0xBBu8; 48]));
        // Index 2 not set, should match any
        assert!(expected.matches_rtmr(2, &[0x00u8; 48]));
        // Invalid index should always match
        assert!(expected.matches_rtmr(5, &[0x00u8; 48]));
    }

    #[test]
    fn test_matches_all() {
        let expected = ExpectedMeasurements {
            mrtd: Some([0xAAu8; 48]),
            rtmr0: Some([0xBBu8; 48]),
            ..Default::default()
        };

        let mrtd = [0xAAu8; 48];
        let rtmrs = [[0xBBu8; 48], [0x00u8; 48], [0x00u8; 48], [0x00u8; 48]];

        assert!(expected.matches_all(&mrtd, &rtmrs));

        let wrong_mrtd = [0x00u8; 48];
        assert!(!expected.matches_all(&wrong_mrtd, &rtmrs));
    }
}
