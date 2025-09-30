-- Create API keys table for Personal Access Tokens
CREATE TABLE api_keys (
    kid VARCHAR(32) PRIMARY KEY,  -- 16 bytes = 32 hex chars, globally unique
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    hash TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ NULL,
    UNIQUE(user_id, name) -- Ensures unique names per user
);

-- Index for fast user key listing
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);