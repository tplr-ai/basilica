-- Add ssh_public_key column to all rental tables
-- This stores the actual public key used at rental creation, enabling SSH access
-- even after the user deletes their registered SSH key

-- 1. Secure cloud rentals (active)
ALTER TABLE secure_cloud_rentals ADD COLUMN ssh_public_key TEXT;

-- Backfill from ssh_keys table
UPDATE secure_cloud_rentals r
SET ssh_public_key = k.public_key
FROM ssh_keys k
WHERE r.ssh_key_id = k.id AND r.ssh_public_key IS NULL;

-- Drop the ssh_key_id column (no longer needed)
ALTER TABLE secure_cloud_rentals DROP COLUMN ssh_key_id;

-- 2. Terminated secure cloud rentals
ALTER TABLE terminated_secure_cloud_rentals ADD COLUMN ssh_public_key TEXT;

UPDATE terminated_secure_cloud_rentals r
SET ssh_public_key = k.public_key
FROM ssh_keys k
WHERE r.ssh_key_id = k.id AND r.ssh_public_key IS NULL;

ALTER TABLE terminated_secure_cloud_rentals DROP COLUMN ssh_key_id;

-- 3. Community cloud rentals (active)
ALTER TABLE user_rentals ADD COLUMN ssh_public_key TEXT;

-- 4. Terminated community cloud rentals
ALTER TABLE terminated_user_rentals ADD COLUMN ssh_public_key TEXT;
