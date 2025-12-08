//! SSH public key validation utilities.

/// Supported SSH public key type prefixes.
///
/// These cover all standard OpenSSH key types:
/// - RSA keys (`ssh-rsa`)
/// - Ed25519 keys (`ssh-ed25519`)
/// - DSA keys (`ssh-dss`)
/// - ECDSA keys (`ecdsa-sha2-nistp256`, `ecdsa-sha2-nistp384`, `ecdsa-sha2-nistp521`)
/// - FIDO/U2F security keys (`sk-ecdsa-sha2-*`, `sk-ssh-ed25519@openssh.com`)
const VALID_KEY_PREFIXES: &[&str] = &[
    "ssh-rsa ",
    "ssh-ed25519 ",
    "ssh-dss ",
    "ecdsa-sha2-",
    "sk-ecdsa-sha2-",
    "sk-ssh-ed25519@openssh.com ",
];

/// Validates SSH public key format.
///
/// Returns `true` if the key starts with a recognized SSH key algorithm prefix.
///
/// # Examples
///
/// ```
/// use basilica_common::ssh::is_valid_ssh_public_key;
///
/// assert!(is_valid_ssh_public_key("ssh-rsa AAAAB3... user@host"));
/// assert!(is_valid_ssh_public_key("ssh-ed25519 AAAAC3... user@host"));
/// assert!(is_valid_ssh_public_key("sk-ssh-ed25519@openssh.com AAAA... user@host"));
/// assert!(!is_valid_ssh_public_key("invalid-key"));
/// ```
pub fn is_valid_ssh_public_key(public_key: &str) -> bool {
    VALID_KEY_PREFIXES
        .iter()
        .any(|prefix| public_key.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_rsa_key() {
        assert!(is_valid_ssh_public_key(
            "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC user@host"
        ));
    }

    #[test]
    fn test_valid_ed25519_key() {
        assert!(is_valid_ssh_public_key(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI user@host"
        ));
    }

    #[test]
    fn test_valid_dsa_key() {
        assert!(is_valid_ssh_public_key(
            "ssh-dss AAAAB3NzaC1kc3MAAACBAP user@host"
        ));
    }

    #[test]
    fn test_valid_ecdsa_keys() {
        assert!(is_valid_ssh_public_key(
            "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHA user@host"
        ));
        assert!(is_valid_ssh_public_key(
            "ecdsa-sha2-nistp384 AAAAE2VjZHNhLXNoYTItbmlzdHA user@host"
        ));
        assert!(is_valid_ssh_public_key(
            "ecdsa-sha2-nistp521 AAAAE2VjZHNhLXNoYTItbmlzdHA user@host"
        ));
    }

    #[test]
    fn test_valid_security_key_ecdsa() {
        assert!(is_valid_ssh_public_key(
            "sk-ecdsa-sha2-nistp256@openssh.com AAAA user@host"
        ));
    }

    #[test]
    fn test_valid_security_key_ed25519() {
        assert!(is_valid_ssh_public_key(
            "sk-ssh-ed25519@openssh.com AAAA user@host"
        ));
    }

    #[test]
    fn test_invalid_keys() {
        assert!(!is_valid_ssh_public_key("invalid-key"));
        assert!(!is_valid_ssh_public_key(""));
        assert!(!is_valid_ssh_public_key("ssh-rsa")); // missing space and data
        assert!(!is_valid_ssh_public_key("-----BEGIN RSA PRIVATE KEY-----"));
    }
}
