//! Static word lists for HUID generation
//!
//! This module contains carefully curated lists of adjectives and nouns
//! used to generate human-readable identifiers. Each word is:
//! - 3-12 characters long
//! - Lowercase only
//! - Easy to pronounce and remember
//! - Non-offensive and unambiguous

/// List of adjectives for HUID generation
pub const ADJECTIVES: [&str; 117] = [
    // A
    "agile", "alert", "amber", "ancient", "arctic", "atomic", "azure", // B
    "bold", "brave", "bright", "bronze", "busy", // C
    "calm", "clear", "clever", "cosmic", "crisp", "cyber", // D
    "dark", "deep", "digital", "dynamic", // E
    "eager", "early", "easy", "electric", "emerald", "epic", // F
    "fast", "fierce", "final", "first", "fluid", "fresh", "frozen", // G
    "gentle", "golden", "grand", "great", "green", // H
    "happy", "hardy", "hidden", "holy", "humble", // I
    "icy", "inner", "iron", // J
    "jolly", "jumbo", // K
    "keen", "kind", // L
    "large", "last", "liquid", "little", "lively", "lunar", // M
    "magic", "major", "mighty", "modern", "mystic", // N
    "naval", "neat", "noble", "northern", // O
    "orange", "outer", // P
    "pacific", "perfect", "plain", "polar", "prime", "proud", "pure", // Q
    "quantum", "quick", "quiet", // R
    "radiant", "rapid", "ready", "royal", // S
    "sacred", "safe", "secret", "sharp", "shiny", "silent", "silver", "simple", "sleek", "smart",
    "smooth", "solar", "solid", "sonic", "stable", "stellar", "strong", "super", "swift",
    // T
    "tiny", "true", "turbo", // U
    "ultra", "unique", "urban", // V
    "vast", "velvet", "vital", // W
    "warm", "wild", "wise",  // Y
    "young", // Z
    "zen", "zero",
];

/// List of nouns for HUID generation
pub const NOUNS: [&str; 167] = [
    // A
    "anchor", "angel", "apple", "archer", "armor", "arrow", "atlas", // B
    "badge", "beacon", "bear", "bird", "blade", "bolt", "bridge", // C
    "canyon", "castle", "chain", "charm", "city", "cloud", "comet", "compass", "coral", "crown",
    "crystal", // D
    "dawn", "delta", "diamond", "dome", "dragon", "dream", "drift", // E
    "eagle", "echo", "edge", "ember", "engine", // F
    "falcon", "field", "fire", "flame", "flash", "fleet", "forest", "forge", "fountain", "fox",
    // G
    "galaxy", "garden", "gate", "gear", "gem", "ghost", "giant", "glacier", "grove", "guard",
    // H
    "hammer", "harbor", "hawk", "heart", "helm", "hero", "horizon", // I
    "island", "ivory", // J
    "jade", "jaguar", "jewel", // K
    "kernel", "key", "knight", // L
    "lake", "lance", "leaf", "light", "lion", "lotus", // M
    "marble", "mask", "meteor", "mirror", "moon", "mountain", // N
    "nebula", "nexus", "node", "nova", // O
    "oasis", "ocean", "oracle", "orbit", "orchid", // P
    "palace", "panda", "park", "path", "pearl", "phoenix", "pilot", "planet", "portal", "prism",
    "pulse", // Q
    "quartz", "quest", // R
    "rainbow", "ranger", "raven", "realm", "reef", "ridge", "ring", "river", "rocket", "ruby",
    // S
    "sage", "sail", "sapphire", "scout", "seal", "shadow", "shark", "shield", "ship", "shore",
    "signal", "sky", "spark", "spear", "sphere", "spider", "spirit", "spring", "star", "stone",
    "storm", "stream", "summit", "sword", // T
    "temple", "thunder", "tiger", "titan", "torch", "tower", "trail", "tree", "tribe",
    // V
    "valley", "vault", "vector", "vertex", "village", "vortex", // W
    "walker", "warrior", "watch", "water", "wave", "whale", "wind", "wing", "wolf", "world",
    // Z
    "zenith", "zone",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_adjectives_count() {
        assert_eq!(
            ADJECTIVES.len(),
            117,
            "ADJECTIVES must contain exactly 117 words"
        );
    }

    #[test]
    fn test_nouns_count() {
        assert_eq!(NOUNS.len(), 167, "NOUNS must contain exactly 167 words");
    }

    #[test]
    fn test_adjectives_unique() {
        let unique: HashSet<_> = ADJECTIVES.iter().collect();
        assert_eq!(
            unique.len(),
            ADJECTIVES.len(),
            "ADJECTIVES contains duplicate words"
        );
    }

    #[test]
    fn test_nouns_unique() {
        let unique: HashSet<_> = NOUNS.iter().collect();
        assert_eq!(unique.len(), NOUNS.len(), "NOUNS contains duplicate words");
    }

    #[test]
    fn test_word_length_constraints() {
        for adj in &ADJECTIVES {
            assert!(
                adj.len() >= 3 && adj.len() <= 12,
                "Adjective '{}' length {} is not between 3-12 characters",
                adj,
                adj.len()
            );
        }

        for noun in &NOUNS {
            assert!(
                noun.len() >= 3 && noun.len() <= 12,
                "Noun '{}' length {} is not between 3-12 characters",
                noun,
                noun.len()
            );
        }
    }

    #[test]
    fn test_words_lowercase() {
        for adj in &ADJECTIVES {
            assert_eq!(
                adj.to_lowercase(),
                *adj,
                "Adjective '{adj}' is not lowercase"
            );
        }

        for noun in &NOUNS {
            assert_eq!(noun.to_lowercase(), *noun, "Noun '{noun}' is not lowercase");
        }
    }

    #[test]
    fn test_words_no_special_chars() {
        for adj in &ADJECTIVES {
            assert!(
                adj.chars().all(|c| c.is_ascii_lowercase()),
                "Adjective '{adj}' contains non-lowercase ASCII characters"
            );
        }

        for noun in &NOUNS {
            assert!(
                noun.chars().all(|c| c.is_ascii_lowercase()),
                "Noun '{noun}' contains non-lowercase ASCII characters"
            );
        }
    }

    #[test]
    fn test_no_overlap_between_lists() {
        let adj_set: HashSet<_> = ADJECTIVES.iter().collect();
        let noun_set: HashSet<_> = NOUNS.iter().collect();

        let overlap: Vec<_> = adj_set.intersection(&noun_set).collect();
        assert!(
            overlap.is_empty(),
            "Words appear in both lists: {overlap:?}"
        );
    }

    #[test]
    fn test_total_combinations() {
        let total = ADJECTIVES.len() * NOUNS.len() * 65536;
        assert_eq!(
            total, 1_280_507_904,
            "Total combinations should be 1,280,507,904"
        );
    }

    #[test]
    fn test_alphabetical_organization() {
        // This test is informational - it helps verify words are somewhat organized
        // but doesn't enforce strict alphabetical order
        let mut adj_first_letters = HashSet::new();
        for adj in &ADJECTIVES {
            adj_first_letters.insert(adj.chars().next().unwrap());
        }

        let mut noun_first_letters = HashSet::new();
        for noun in &NOUNS {
            noun_first_letters.insert(noun.chars().next().unwrap());
        }

        // We should have good coverage of the alphabet
        assert!(
            adj_first_letters.len() >= 15,
            "Adjectives should cover at least 15 different starting letters, found {}",
            adj_first_letters.len()
        );

        assert!(
            noun_first_letters.len() >= 15,
            "Nouns should cover at least 15 different starting letters, found {}",
            noun_first_letters.len()
        );
    }
}
