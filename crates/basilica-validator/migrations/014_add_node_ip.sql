-- Add node_ip column for uniqueness enforcement (one node per physical host)
ALTER TABLE miner_nodes ADD COLUMN node_ip TEXT NOT NULL DEFAULT '';

-- Backfill node_ip from ssh_endpoint (expected format: user@ip:port)
-- Extract the substring between '@' and ':' as the IP
UPDATE miner_nodes
SET node_ip = substr(
  substr(ssh_endpoint, instr(ssh_endpoint, '@') + 1),
  1,
  instr(substr(ssh_endpoint, instr(ssh_endpoint, '@') + 1), ':') - 1
)
WHERE ssh_endpoint LIKE '%@%:%';

-- Remove rows where we couldn't parse a valid node_ip
DELETE FROM miner_nodes WHERE node_ip = '';

-- Unique index: only one node per IP across all miners
CREATE UNIQUE INDEX idx_miner_nodes_node_ip ON miner_nodes(node_ip) WHERE node_ip != '';
