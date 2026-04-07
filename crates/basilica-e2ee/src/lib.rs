use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use hkdf::Hkdf;
use kem::{Decapsulate, Encapsulate};
use ml_kem::array::typenum::Unsigned;
use ml_kem::{Encoded, EncodedSizeUser, KemCore, MlKem768};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sharks::{Share, Sharks};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};
use zeroize::{Zeroize, ZeroizeOnDrop};

const AEAD_KEY_LEN: usize = 32;
const AEAD_NONCE_LEN: usize = 12;
const HKDF_SALT_LEN: usize = 16;

pub const REQUEST_INFO: &[u8] = b"basilica-req-v1";
pub const RESPONSE_INFO: &[u8] = b"basilica-resp-v1";
pub const STREAM_INFO: &[u8] = b"basilica-stream-v1";

type MlKemDecapsulationKey = <MlKem768 as KemCore>::DecapsulationKey;
type MlKemEncapsulationKey = <MlKem768 as KemCore>::EncapsulationKey;
type MlKemCiphertext = ml_kem::Ciphertext<MlKem768>;

#[derive(Debug, thiserror::Error)]
pub enum E2eeError {
    #[error("invalid base64 payload: {0}")]
    InvalidBase64(#[from] base64::DecodeError),
    #[error("invalid ML-KEM key material")]
    InvalidKeyMaterial,
    #[error("invalid ML-KEM ciphertext")]
    InvalidCiphertext,
    #[error("invalid blob format")]
    InvalidBlob,
    #[error("encryption failed")]
    EncryptFailed,
    #[error("decryption failed")]
    DecryptFailed,
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("attestation is stale or malformed")]
    InvalidAttestation,
    #[error("invalid share bundle configuration")]
    InvalidShareConfig,
    #[error("invalid Shamir share")]
    InvalidShare,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrustTier {
    Enterprise,
    Community,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttestationKind {
    Tdx,
    Tpm,
    Synthetic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AttestationBinding {
    pub nonce: String,
    pub e2ee_public_key_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tls_public_key_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AttestationEvidence {
    pub workload_id: String,
    pub instance_id: String,
    pub trust_tier: TrustTier,
    pub kind: AttestationKind,
    pub issued_at_unix_secs: i64,
    pub expires_at_unix_secs: i64,
    pub quote_base64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_evidence_base64: Option<String>,
    pub binding: AttestationBinding,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub claims: BTreeMap<String, String>,
}

impl AttestationEvidence {
    pub fn verify_binding(
        &self,
        public_key_base64: &str,
        max_age_secs: i64,
    ) -> Result<(), E2eeError> {
        let now = now_unix_secs();
        if self.expires_at_unix_secs < now {
            return Err(E2eeError::InvalidAttestation);
        }

        if now.saturating_sub(self.issued_at_unix_secs) > max_age_secs {
            return Err(E2eeError::InvalidAttestation);
        }

        let actual_hash = sha256_hex(&BASE64.decode(public_key_base64.as_bytes())?);
        if actual_hash != self.binding.e2ee_public_key_sha256 {
            return Err(E2eeError::InvalidAttestation);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct E2eeInstanceDescriptor {
    pub workload_id: String,
    pub instance_id: String,
    pub trust_tier: TrustTier,
    pub e2ee_public_key: String,
    pub attestation: AttestationEvidence,
    #[serde(default)]
    pub nonces: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

impl E2eeInstanceDescriptor {
    pub fn verify(&self, max_age_secs: i64) -> Result<(), E2eeError> {
        self.attestation
            .verify_binding(&self.e2ee_public_key, max_age_secs)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct E2eeRegistrationRequest {
    pub workload_id: String,
    pub instance_id: String,
    pub trust_tier: TrustTier,
    pub endpoint_url: String,
    pub e2ee_public_key: String,
    pub attestation: AttestationEvidence,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct E2eeRegistrationResponse {
    pub accepted: bool,
    pub registered_at_unix_secs: i64,
    pub workload_id: String,
    pub instance_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TpmKeyReleasePolicy {
    pub shares_required: u8,
    pub total_shares: u8,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_pcrs: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_measurements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommunityKeyShareBundle {
    pub key_id: String,
    pub policy: TpmKeyReleasePolicy,
    pub shares: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ConfidentialRequestEnvelope {
    pub request_path: String,
    #[serde(default = "default_request_method")]
    pub request_method: String,
    #[serde(default = "default_content_type")]
    pub content_type: String,
    pub request_body_base64: String,
    pub response_public_key: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

impl ConfidentialRequestEnvelope {
    pub fn new_json(
        request_path: impl Into<String>,
        payload: &impl Serialize,
        response_public_key: String,
        stream: bool,
    ) -> Result<Self, E2eeError> {
        let body = serde_json::to_vec(payload)?;
        Ok(Self {
            request_path: request_path.into(),
            request_method: default_request_method(),
            content_type: default_content_type(),
            request_body_base64: BASE64.encode(body),
            response_public_key,
            stream,
            headers: BTreeMap::new(),
        })
    }

    pub fn request_body_bytes(&self) -> Result<Vec<u8>, E2eeError> {
        BASE64
            .decode(self.request_body_base64.as_bytes())
            .map_err(E2eeError::from)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StreamInitEvent {
    pub e2e_init: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StreamChunkEvent {
    pub e2e: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StreamErrorEvent {
    pub e2e_error: String,
}

#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ServerKeypair {
    decapsulation_key: Vec<u8>,
    encapsulation_key: Vec<u8>,
}

impl ServerKeypair {
    pub fn generate() -> Self {
        let (dk, ek) = MlKem768::generate(&mut OsRng);
        Self {
            decapsulation_key: dk.as_bytes().as_slice().to_vec(),
            encapsulation_key: ek.as_bytes().as_slice().to_vec(),
        }
    }

    pub fn public_key_base64(&self) -> String {
        BASE64.encode(&self.encapsulation_key)
    }

    pub fn public_key_sha256_hex(&self) -> String {
        sha256_hex(&self.encapsulation_key)
    }

    pub fn decapsulation_key_base64(&self) -> String {
        BASE64.encode(&self.decapsulation_key)
    }

    fn decapsulation_key(&self) -> Result<MlKemDecapsulationKey, E2eeError> {
        decode_key::<MlKemDecapsulationKey>(&self.decapsulation_key)
    }
}

#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ClientResponseSecret {
    decapsulation_key: Vec<u8>,
}

impl ClientResponseSecret {
    fn new(decapsulation_key: Vec<u8>) -> Self {
        Self { decapsulation_key }
    }

    fn decapsulation_key(&self) -> Result<MlKemDecapsulationKey, E2eeError> {
        decode_key::<MlKemDecapsulationKey>(&self.decapsulation_key)
    }
}

pub fn generate_server_keypair() -> ServerKeypair {
    ServerKeypair::generate()
}

pub fn random_nonce_base64() -> String {
    let mut bytes = [0u8; 24];
    OsRng.fill_bytes(&mut bytes);
    BASE64.encode(bytes)
}

pub fn split_secret_shares(
    secret: &[u8],
    policy: TpmKeyReleasePolicy,
) -> Result<CommunityKeyShareBundle, E2eeError> {
    if secret.is_empty()
        || policy.shares_required == 0
        || policy.shares_required > policy.total_shares
        || policy.total_shares == 0
    {
        return Err(E2eeError::InvalidShareConfig);
    }

    let sharks = Sharks(policy.shares_required);
    let dealer = sharks.dealer(secret);
    let shares = dealer
        .take(policy.total_shares as usize)
        .map(|share| BASE64.encode(Vec::from(&share)))
        .collect();

    Ok(CommunityKeyShareBundle {
        key_id: sha256_hex(secret),
        policy,
        shares,
    })
}

pub fn recover_secret_from_shares(
    bundle: &CommunityKeyShareBundle,
    provided_shares: &[String],
) -> Result<Vec<u8>, E2eeError> {
    if provided_shares.len() < bundle.policy.shares_required as usize {
        return Err(E2eeError::InvalidShareConfig);
    }

    let shares: Vec<Share> = provided_shares
        .iter()
        .map(|encoded| {
            let decoded = BASE64.decode(encoded.as_bytes())?;
            Share::try_from(decoded.as_slice()).map_err(|_| E2eeError::InvalidShare)
        })
        .collect::<Result<_, E2eeError>>()?;

    Sharks(bundle.policy.shares_required)
        .recover(&shares)
        .map_err(|_| E2eeError::InvalidShare)
}

pub fn seal_request(
    server_public_key_base64: &str,
    envelope: &ConfidentialRequestEnvelope,
) -> Result<(Vec<u8>, ClientResponseSecret), E2eeError> {
    let server_public_key =
        decode_key::<MlKemEncapsulationKey>(&BASE64.decode(server_public_key_base64.as_bytes())?)?;
    let (response_dk, response_ek) = MlKem768::generate(&mut OsRng);
    let response_secret = ClientResponseSecret::new(response_dk.as_bytes().as_slice().to_vec());

    let mut envelope_with_response_key = envelope.clone();
    envelope_with_response_key.response_public_key =
        BASE64.encode(response_ek.as_bytes().as_slice());
    let request_plaintext = serde_json::to_vec(&envelope_with_response_key)?;

    let (ciphertext, shared_secret) = server_public_key
        .encapsulate(&mut OsRng)
        .map_err(|_| E2eeError::EncryptFailed)?;

    let request_key = derive_aead_key(
        ciphertext.as_slice(),
        shared_secret.as_slice(),
        REQUEST_INFO,
    )?;
    let sealed = seal_payload(&request_key, &request_plaintext)?;

    let mut blob = Vec::with_capacity(ciphertext.as_slice().len() + sealed.len());
    blob.extend_from_slice(ciphertext.as_slice());
    blob.extend_from_slice(&sealed);
    Ok((blob, response_secret))
}

pub fn open_request(
    server_keypair: &ServerKeypair,
    blob: &[u8],
) -> Result<ConfidentialRequestEnvelope, E2eeError> {
    let ciphertext_len = <MlKem768 as KemCore>::CiphertextSize::USIZE;
    if blob.len() <= ciphertext_len + AEAD_NONCE_LEN {
        return Err(E2eeError::InvalidBlob);
    }

    let (ciphertext_bytes, sealed_payload) = blob.split_at(ciphertext_len);
    let ciphertext = decode_ciphertext(ciphertext_bytes)?;
    let shared_secret = server_keypair
        .decapsulation_key()?
        .decapsulate(&ciphertext)
        .map_err(|_| E2eeError::DecryptFailed)?;
    let request_key = derive_aead_key(ciphertext_bytes, shared_secret.as_slice(), REQUEST_INFO)?;
    let plaintext = open_payload(&request_key, sealed_payload)?;
    Ok(serde_json::from_slice(&plaintext)?)
}

pub fn seal_response(
    response_public_key_base64: &str,
    plaintext: &[u8],
) -> Result<Vec<u8>, E2eeError> {
    let response_public_key = decode_key::<MlKemEncapsulationKey>(
        &BASE64.decode(response_public_key_base64.as_bytes())?,
    )?;
    let (ciphertext, shared_secret) = response_public_key
        .encapsulate(&mut OsRng)
        .map_err(|_| E2eeError::EncryptFailed)?;
    let response_key = derive_aead_key(
        ciphertext.as_slice(),
        shared_secret.as_slice(),
        RESPONSE_INFO,
    )?;
    let sealed = seal_payload(&response_key, plaintext)?;

    let mut blob = Vec::with_capacity(ciphertext.as_slice().len() + sealed.len());
    blob.extend_from_slice(ciphertext.as_slice());
    blob.extend_from_slice(&sealed);
    Ok(blob)
}

pub fn open_response(
    client_secret: &ClientResponseSecret,
    blob: &[u8],
) -> Result<Vec<u8>, E2eeError> {
    let ciphertext_len = <MlKem768 as KemCore>::CiphertextSize::USIZE;
    if blob.len() <= ciphertext_len + AEAD_NONCE_LEN {
        return Err(E2eeError::InvalidBlob);
    }

    let (ciphertext_bytes, sealed_payload) = blob.split_at(ciphertext_len);
    let ciphertext = decode_ciphertext(ciphertext_bytes)?;
    let shared_secret = client_secret
        .decapsulation_key()?
        .decapsulate(&ciphertext)
        .map_err(|_| E2eeError::DecryptFailed)?;
    let response_key = derive_aead_key(ciphertext_bytes, shared_secret.as_slice(), RESPONSE_INFO)?;
    open_payload(&response_key, sealed_payload)
}

pub fn generate_stream_key() -> [u8; AEAD_KEY_LEN] {
    let mut key = [0u8; AEAD_KEY_LEN];
    OsRng.fill_bytes(&mut key);
    key
}

pub fn seal_stream_init(
    response_public_key_base64: &str,
    stream_key: &[u8; AEAD_KEY_LEN],
) -> Result<String, E2eeError> {
    Ok(BASE64.encode(seal_response(response_public_key_base64, stream_key)?))
}

pub fn open_stream_init(
    client_secret: &ClientResponseSecret,
    stream_init_base64: &str,
) -> Result<[u8; AEAD_KEY_LEN], E2eeError> {
    let decrypted = open_response(
        client_secret,
        &BASE64.decode(stream_init_base64.as_bytes())?,
    )?;
    let stream_key: [u8; AEAD_KEY_LEN] = decrypted
        .as_slice()
        .try_into()
        .map_err(|_| E2eeError::InvalidBlob)?;
    Ok(stream_key)
}

pub fn seal_stream_chunk(
    stream_key: &[u8; AEAD_KEY_LEN],
    plaintext: &[u8],
) -> Result<String, E2eeError> {
    Ok(BASE64.encode(seal_payload(stream_key, plaintext)?))
}

pub fn open_stream_chunk(
    stream_key: &[u8; AEAD_KEY_LEN],
    stream_chunk_base64: &str,
) -> Result<Vec<u8>, E2eeError> {
    open_payload(stream_key, &BASE64.decode(stream_chunk_base64.as_bytes())?)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn default_request_method() -> String {
    "POST".to_string()
}

fn default_content_type() -> String {
    "application/json".to_string()
}

fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn derive_aead_key(
    ciphertext: &[u8],
    shared_secret: &[u8],
    info: &[u8],
) -> Result<[u8; AEAD_KEY_LEN], E2eeError> {
    let salt = &ciphertext[..ciphertext.len().min(HKDF_SALT_LEN)];
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret);
    let mut key = [0u8; AEAD_KEY_LEN];
    hkdf.expand(info, &mut key)
        .map_err(|_| E2eeError::EncryptFailed)?;
    Ok(key)
}

fn seal_payload(key: &[u8; AEAD_KEY_LEN], plaintext: &[u8]) -> Result<Vec<u8>, E2eeError> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| E2eeError::EncryptFailed)?;
    let mut nonce = [0u8; AEAD_NONCE_LEN];
    OsRng.fill_bytes(&mut nonce);

    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(|_| E2eeError::EncryptFailed)?;

    let mut sealed = Vec::with_capacity(AEAD_NONCE_LEN + ciphertext.len());
    sealed.extend_from_slice(&nonce);
    sealed.extend_from_slice(&ciphertext);
    Ok(sealed)
}

fn open_payload(key: &[u8; AEAD_KEY_LEN], sealed_payload: &[u8]) -> Result<Vec<u8>, E2eeError> {
    if sealed_payload.len() <= AEAD_NONCE_LEN {
        return Err(E2eeError::InvalidBlob);
    }

    let (nonce, ciphertext) = sealed_payload.split_at(AEAD_NONCE_LEN);
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| E2eeError::DecryptFailed)?;
    cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| E2eeError::DecryptFailed)
}

fn decode_key<T>(bytes: &[u8]) -> Result<T, E2eeError>
where
    T: EncodedSizeUser,
{
    let encoded = Encoded::<T>::try_from(bytes).map_err(|_| E2eeError::InvalidKeyMaterial)?;
    Ok(T::from_bytes(&encoded))
}

fn decode_ciphertext(bytes: &[u8]) -> Result<MlKemCiphertext, E2eeError> {
    MlKemCiphertext::try_from(bytes).map_err(|_| E2eeError::InvalidCiphertext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_response_round_trip() {
        let server = generate_server_keypair();
        let envelope = ConfidentialRequestEnvelope {
            request_path: "/v1/chat/completions".to_string(),
            request_method: "POST".to_string(),
            content_type: "application/json".to_string(),
            request_body_base64: BASE64.encode(br#"{"prompt":"night shade"}"#),
            response_public_key: String::new(),
            stream: false,
            headers: BTreeMap::from([("x-test".to_string(), "1".to_string())]),
        };

        let (blob, client_secret) = seal_request(&server.public_key_base64(), &envelope).unwrap();
        let decrypted = open_request(&server, &blob).unwrap();
        assert_eq!(decrypted.request_path, envelope.request_path);
        assert_eq!(
            decrypted.request_body_bytes().unwrap(),
            br#"{"prompt":"night shade"}"#
        );
        assert!(!decrypted.response_public_key.is_empty());

        let response_blob =
            seal_response(&decrypted.response_public_key, br#"{"ok":true}"#).unwrap();
        let plaintext = open_response(&client_secret, &response_blob).unwrap();
        assert_eq!(plaintext, br#"{"ok":true}"#);
    }

    #[test]
    fn stream_round_trip() {
        let server = generate_server_keypair();
        let envelope = ConfidentialRequestEnvelope {
            request_path: "/v1/responses".to_string(),
            request_method: "POST".to_string(),
            content_type: "application/json".to_string(),
            request_body_base64: BASE64.encode(br#"{"stream":true}"#),
            response_public_key: String::new(),
            stream: true,
            headers: BTreeMap::new(),
        };

        let (blob, client_secret) = seal_request(&server.public_key_base64(), &envelope).unwrap();
        let decrypted = open_request(&server, &blob).unwrap();
        let stream_key = generate_stream_key();
        let init = seal_stream_init(&decrypted.response_public_key, &stream_key).unwrap();
        let recovered_key = open_stream_init(&client_secret, &init).unwrap();
        assert_eq!(stream_key, recovered_key);

        let chunk = seal_stream_chunk(&stream_key, b"night").unwrap();
        let plaintext = open_stream_chunk(&recovered_key, &chunk).unwrap();
        assert_eq!(plaintext, b"night");
    }

    #[test]
    fn attestation_binding_verification() {
        let server = generate_server_keypair();
        let pubkey = server.public_key_base64();
        let evidence = AttestationEvidence {
            workload_id: "w1".to_string(),
            instance_id: "i1".to_string(),
            trust_tier: TrustTier::Enterprise,
            kind: AttestationKind::Synthetic,
            issued_at_unix_secs: now_unix_secs(),
            expires_at_unix_secs: now_unix_secs() + 60,
            quote_base64: BASE64.encode("synthetic-quote"),
            gpu_evidence_base64: None,
            binding: AttestationBinding {
                nonce: random_nonce_base64(),
                e2ee_public_key_sha256: server.public_key_sha256_hex(),
                tls_public_key_sha256: None,
            },
            claims: BTreeMap::new(),
        };
        assert!(evidence.verify_binding(&pubkey, 300).is_ok());
    }

    #[test]
    fn split_and_recover_secret_shares() {
        let policy = TpmKeyReleasePolicy {
            shares_required: 3,
            total_shares: 5,
            required_pcrs: vec![0, 7],
            allowed_measurements: vec!["abc".to_string()],
        };
        let bundle = split_secret_shares(b"night-shade-luks-key", policy).unwrap();
        let recovered = recover_secret_from_shares(&bundle, &bundle.shares[..3]).unwrap();
        assert_eq!(recovered, b"night-shade-luks-key");
    }
}
