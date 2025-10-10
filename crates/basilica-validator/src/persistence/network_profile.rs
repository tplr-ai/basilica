use sqlx::Row;
use tracing::info;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    #[allow(clippy::too_many_arguments)]
    pub async fn store_node_network_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        ip_address: Option<String>,
        hostname: Option<String>,
        city: Option<String>,
        region: Option<String>,
        country: Option<String>,
        location: Option<String>,
        organization: Option<String>,
        postal_code: Option<String>,
        timezone: Option<String>,
        test_timestamp: &str,
        full_result_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO node_network_profile
            (miner_uid, node_id, ip_address, hostname, city, region, country, location,
             organization, postal_code, timezone, test_timestamp, full_result_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                ip_address = excluded.ip_address,
                hostname = excluded.hostname,
                city = excluded.city,
                region = excluded.region,
                country = excluded.country,
                location = excluded.location,
                organization = excluded.organization,
                postal_code = excluded.postal_code,
                timezone = excluded.timezone,
                test_timestamp = excluded.test_timestamp,
                full_result_json = excluded.full_result_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(ip_address.clone())
        .bind(hostname)
        .bind(city.clone())
        .bind(region.clone())
        .bind(country.clone())
        .bind(location.clone())
        .bind(organization.clone())
        .bind(postal_code)
        .bind(timezone)
        .bind(test_timestamp)
        .bind(full_result_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            ip = ip_address.unwrap_or_else(|| "Unknown".to_string()),
            country = country.unwrap_or_else(|| "Unknown".to_string()),
            city = city.unwrap_or_else(|| "Unknown".to_string()),
            region = region.unwrap_or_else(|| "Unknown".to_string()),
            organization = organization.unwrap_or_else(|| "Unknown".to_string()),
            location = location.unwrap_or_else(|| "Unknown".to_string()),
            "Stored network profile for node"
        );

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub async fn get_node_network_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<
        Option<(
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            String,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT ip_address, hostname, city, region, country, location, organization,
                   postal_code, timezone, test_timestamp, full_result_json
            FROM node_network_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_result_json: String = row.get("full_result_json");
            let ip_address: Option<String> = row.get("ip_address");
            let hostname: Option<String> = row.get("hostname");
            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");
            let location: Option<String> = row.get("location");
            let organization: Option<String> = row.get("organization");
            let postal_code: Option<String> = row.get("postal_code");
            let timezone: Option<String> = row.get("timezone");
            let test_timestamp: String = row.get("test_timestamp");

            Ok(Some((
                full_result_json,
                ip_address,
                hostname,
                city,
                region,
                country,
                location,
                organization,
                postal_code,
                timezone,
                test_timestamp,
            )))
        } else {
            Ok(None)
        }
    }
}
