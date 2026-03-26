//! Test module for dynamic discovery functionality

#[cfg(test)]
mod tests {
    use crate::miner_prover::types::{MinerInfo, NodeInfo};
    use basilica_common::identity::{Hotkey, MinerUid, NodeId};

    #[tokio::test]
    async fn test_miner_discovery_ip_conversion() {
        // Test IPv4 conversion logic
        let ipv4: u128 = 0xC0A80101; // 192.168.1.1
        let ipv4_bytes = ipv4.to_be_bytes();
        let ipv4_addr = std::net::Ipv4Addr::new(
            ipv4_bytes[12],
            ipv4_bytes[13],
            ipv4_bytes[14],
            ipv4_bytes[15],
        );
        assert_eq!(ipv4_addr.to_string(), "192.168.1.1");

        // Test IPv6
        let ipv6: u128 = 0x20010db8000000000000000000000001;
        let ipv6_addr = std::net::Ipv6Addr::from(ipv6);
        assert_eq!(ipv6_addr.to_string(), "2001:db8::1");
    }

    #[test]
    fn test_miner_info_creation() {
        let uid = MinerUid::new(1);
        let endpoint = "http://127.0.0.1:8091".to_string();
        let hotkey =
            Hotkey::new("5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty".to_string()).unwrap();

        let miner_info = MinerInfo {
            uid,
            endpoint: endpoint.clone(),
            hotkey,
            is_validator: false,
            stake_tao: 0.0,
            last_verified: None,
            verification_score: 0.0,
        };

        assert_eq!(miner_info.uid.as_u16(), 1);
        assert_eq!(miner_info.endpoint, endpoint);
    }

    #[test]
    fn test_node_info_creation() {
        let node_info = NodeInfo {
            id: NodeId::new("test_node").unwrap(),
            miner_uid: MinerUid::new(1),
            node_ssh_endpoint: "root@127.0.0.1:50051".to_string(),
        };

        assert_eq!(node_info.miner_uid.as_u16(), 1);
        assert_eq!(node_info.node_ssh_endpoint, "root@127.0.0.1:50051");
    }
}
