use super::access_control::ValidatorAccessControl;
use super::types::{ValidatorAccess, ValidatorConfig, ValidatorId};
use anyhow::Result;
use common::journal::{log_validator_access_granted, log_validator_access_revoked};
use common::ssh::{SimpleSshKeys, SimpleSshUsers};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct ValidationSessionService {
    access_control: Arc<ValidatorAccessControl>,
    config: ValidatorConfig,
}

impl ValidationSessionService {
    pub fn new(config: ValidatorConfig) -> Result<Self> {
        let access_control = Arc::new(ValidatorAccessControl::new(config.access_config.clone()));

        Ok(Self {
            access_control,
            config,
        })
    }

    pub fn access_control(&self) -> &Arc<ValidatorAccessControl> {
        &self.access_control
    }

    pub async fn grant_ssh_access(
        &self,
        validator_id: &ValidatorId,
        public_key: &str,
    ) -> Result<()> {
        info!("Granting SSH access to validator: {}", validator_id);

        let username = SimpleSshUsers::validator_username(&validator_id.hotkey);

        SimpleSshUsers::create_user(&username).await?;

        let restrictions = if self.config.strict_ssh_restrictions {
            SimpleSshKeys::get_strict_restrictions()
        } else {
            SimpleSshKeys::get_default_restrictions()
        };

        SimpleSshKeys::add_key(&username, public_key, &restrictions).await?;

        let access = ValidatorAccess::new(validator_id.clone(), public_key);

        self.access_control.grant_access(&access).await?;

        log_validator_access_granted(&validator_id.hotkey, "ssh", 3600, HashMap::new());

        info!(
            "SSH access granted successfully to validator: {}",
            validator_id
        );
        Ok(())
    }

    pub async fn revoke_ssh_access(&self, validator_id: &ValidatorId) -> Result<()> {
        info!("Revoking SSH access for validator: {}", validator_id);

        let username = SimpleSshUsers::validator_username(&validator_id.hotkey);

        SimpleSshKeys::remove_key(&username).await?;

        self.access_control.revoke_access(validator_id).await?;

        log_validator_access_revoked(&validator_id.hotkey, "manual_revocation", HashMap::new());

        // Note: I haven't tested user removal, so it's commented out for now.
        // SimpleSshUsers::remove_user(&username).await?;

        info!(
            "SSH access revoked successfully for validator: {}",
            validator_id
        );
        Ok(())
    }

    pub async fn has_ssh_access(&self, validator_id: &ValidatorId) -> bool {
        let has_permission = self.access_control.has_access(validator_id).await;
        let username = SimpleSshUsers::validator_username(&validator_id.hotkey);
        let has_key = SimpleSshKeys::has_key(&username).await;

        has_permission && has_key
    }

    pub async fn list_active_access(&self) -> Result<Vec<ValidatorAccess>> {
        self.access_control.list_active_access().await
    }

    pub async fn cleanup_access(&self) -> Result<u32> {
        let mut cleaned = 0;
        let active_access = self.access_control.list_active_access().await?;

        for access in active_access {
            let username = SimpleSshUsers::validator_username(&access.validator_id.hotkey);

            if !SimpleSshKeys::has_key(&username).await {
                self.access_control
                    .revoke_access(&access.validator_id)
                    .await?;
                cleaned += 1;
                warn!(
                    "Cleaned up orphaned access record for: {}",
                    access.validator_id
                );
            }
        }

        if cleaned > 0 {
            info!("Cleaned up {} orphaned access records", cleaned);
        }

        Ok(cleaned)
    }
}
