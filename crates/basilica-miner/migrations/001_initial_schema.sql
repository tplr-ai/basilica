-- Initial miner database schema

-- Node identity (UUID) tracking
CREATE TABLE IF NOT EXISTS node_uuids (
    node_address TEXT NOT NULL UNIQUE,
    uuid TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
