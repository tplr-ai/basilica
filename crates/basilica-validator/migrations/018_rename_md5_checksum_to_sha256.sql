-- Rename checksum column to SHA-256 semantics and drop legacy-sized values.
ALTER TABLE collateral_status
RENAME COLUMN url_content_md5_checksum TO url_content_sha256;

UPDATE collateral_status
SET url_content_sha256 = NULL
WHERE url_content_sha256 IS NOT NULL
  AND length(url_content_sha256) != 64;
