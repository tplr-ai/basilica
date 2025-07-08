//! Word provider implementation for HUID generation
//!
//! This module provides the StaticWordProvider that uses the compiled-in
//! word lists from the words module.

use crate::executor_identity::{
    constants::{MIN_ADJECTIVE_COUNT, MIN_NOUN_COUNT},
    words::{ADJECTIVES, NOUNS},
    WordProvider,
};
use anyhow::Result;

/// Static word provider using compiled-in word lists
#[derive(Debug, Clone)]
pub struct StaticWordProvider;

impl StaticWordProvider {
    /// Creates a new static word provider
    pub fn new() -> Self {
        Self
    }
}

impl WordProvider for StaticWordProvider {
    fn get_adjective(&self, index: usize) -> Option<&str> {
        ADJECTIVES.get(index).copied()
    }

    fn get_noun(&self, index: usize) -> Option<&str> {
        NOUNS.get(index).copied()
    }

    fn adjective_count(&self) -> usize {
        ADJECTIVES.len()
    }

    fn noun_count(&self) -> usize {
        NOUNS.len()
    }

    fn validate_word_lists(&self) -> Result<()> {
        let adj_count = self.adjective_count();
        let noun_count = self.noun_count();

        if adj_count < MIN_ADJECTIVE_COUNT {
            anyhow::bail!(
                "Insufficient adjectives: found {}, minimum required {}",
                adj_count,
                MIN_ADJECTIVE_COUNT
            );
        }

        if noun_count < MIN_NOUN_COUNT {
            anyhow::bail!(
                "Insufficient nouns: found {}, minimum required {}",
                noun_count,
                MIN_NOUN_COUNT
            );
        }

        // Verify all words are valid (non-empty)
        for i in 0..adj_count {
            if let Some(word) = self.get_adjective(i) {
                if word.is_empty() {
                    anyhow::bail!("Empty adjective at index {}", i);
                }
            } else {
                anyhow::bail!("Failed to get adjective at index {}", i);
            }
        }

        for i in 0..noun_count {
            if let Some(word) = self.get_noun(i) {
                if word.is_empty() {
                    anyhow::bail!("Empty noun at index {}", i);
                }
            } else {
                anyhow::bail!("Failed to get noun at index {}", i);
            }
        }

        Ok(())
    }
}

impl Default for StaticWordProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_word_provider() {
        let provider = StaticWordProvider::new();

        // Test counts match the actual arrays
        assert_eq!(provider.adjective_count(), 117);
        assert_eq!(provider.noun_count(), 167);

        // Test getting valid indices
        assert_eq!(provider.get_adjective(0), Some("agile"));
        assert_eq!(provider.get_noun(0), Some("anchor"));

        // Test getting last valid indices
        assert_eq!(provider.get_adjective(116), Some("zero"));
        assert_eq!(provider.get_noun(166), Some("zone"));

        // Test getting invalid indices
        assert_eq!(provider.get_adjective(117), None);
        assert_eq!(provider.get_noun(167), None);
        assert_eq!(provider.get_adjective(usize::MAX), None);
        assert_eq!(provider.get_noun(usize::MAX), None);
    }

    #[test]
    fn test_word_provider_validation() {
        let provider = StaticWordProvider::new();

        // Should pass validation with current word lists
        assert!(provider.validate_word_lists().is_ok());
    }

    #[test]
    fn test_word_provider_all_words_accessible() {
        let provider = StaticWordProvider::new();

        // Verify all adjectives are accessible
        for i in 0..provider.adjective_count() {
            let word = provider.get_adjective(i);
            assert!(word.is_some(), "Adjective at index {i} should exist");
            assert!(
                !word.unwrap().is_empty(),
                "Adjective at index {i} should not be empty"
            );
        }

        // Verify all nouns are accessible
        for i in 0..provider.noun_count() {
            let word = provider.get_noun(i);
            assert!(word.is_some(), "Noun at index {i} should exist");
            assert!(
                !word.unwrap().is_empty(),
                "Noun at index {i} should not be empty"
            );
        }
    }

    #[test]
    fn test_default_implementation() {
        let provider1 = StaticWordProvider::new();
        let provider2 = StaticWordProvider;

        // Both should behave identically
        assert_eq!(provider1.adjective_count(), provider2.adjective_count());
        assert_eq!(provider1.noun_count(), provider2.noun_count());
        assert_eq!(provider1.get_adjective(0), provider2.get_adjective(0));
        assert_eq!(provider1.get_noun(0), provider2.get_noun(0));
    }
}
