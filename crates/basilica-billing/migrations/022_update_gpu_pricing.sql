-- Migration 022: Update GPU package pricing with expanded model support
-- Update existing packages and add new GPU families
-- Pricing strategy: Use lowest price among variants for each package ID

-- Update existing packages with reduced pricing
UPDATE billing.billing_packages
SET
  hourly_rate = 0.36,
  base_rate_per_hour = 0.36,
  updated_at = NOW()
WHERE
  package_id = 'a100';

UPDATE billing.billing_packages
SET
  hourly_rate = 1.11,
  base_rate_per_hour = 1.11,
  updated_at = NOW()
WHERE
  package_id = 'h100';

UPDATE billing.billing_packages
SET
  hourly_rate = 1.90,
  base_rate_per_hour = 1.90,
  updated_at = NOW()
WHERE
  package_id = 'h200';

UPDATE billing.billing_packages
SET
  hourly_rate = 2.99,
  base_rate_per_hour = 2.99,
  updated_at = NOW()
WHERE
  package_id = 'b200';

-- Add new GPU packages following the same pattern
INSERT INTO billing.billing_packages
  (package_id, name, description, hourly_rate, gpu_model,
   billing_period, priority, is_active, metadata,
   base_rate_per_hour, cpu_rate_per_hour, disk_iops_rate,
   storage_rate_per_gb_hour, network_rate_per_gb, disk_io_rate_per_gb,
   cpu_rate_per_core_hour, memory_rate_per_gb_hour,
   included_storage_gb_hours, included_network_gb, included_disk_io_gb,
   included_cpu_core_hours, included_memory_gb_hours,
   updated_at)
VALUES
  -- H800 family (Hopper generation)
  ('h800', 'H800 GPU Package', 'NVIDIA H800 80GB GPU compute instances',
   0.80, 'H800', 'Hourly', 105, true,
   '{"gpu_vram_gb": 80, "generation": "Hopper", "variants": ["H800 80GB HBM3", "H800 NVL", "H800 PCIe"]}'::jsonb,
   0.80, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- GeForce RTX 5090
  ('rtx_5090', 'RTX 5090 GPU Package', 'NVIDIA GeForce RTX 5090 compute instances',
   0.17, 'GeForce RTX 5090', 'Hourly', 95, true,
   '{"generation": "Ada Lovelace", "consumer_grade": true}'::jsonb,
   0.17, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- GeForce RTX 4090 family
  ('rtx_4090', 'RTX 4090 GPU Package', 'NVIDIA GeForce RTX 4090 compute instances',
   0.11, 'GeForce RTX 4090', 'Hourly', 90, true,
   '{"generation": "Ada Lovelace", "consumer_grade": true, "variants": ["RTX 4090", "RTX 4090 D"]}'::jsonb,
   0.11, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- RTX Ada Generation family
  ('rtx_ada_6000', 'RTX 6000 Ada Package', 'NVIDIA RTX 6000 Ada Generation compute instances',
   0.31, 'RTX 6000 Ada Generation', 'Hourly', 85, true,
   '{"generation": "Ada Lovelace", "professional": true}'::jsonb,
   0.31, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('rtx_ada_4000', 'RTX 4000 Ada Package', 'NVIDIA RTX 4000 Ada Generation compute instances',
   0.16, 'RTX 4000 Ada Generation', 'Hourly', 80, true,
   '{"generation": "Ada Lovelace", "professional": true}'::jsonb,
   0.16, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('rtx_ada_2000', 'RTX 2000 Ada Package', 'NVIDIA RTX 2000 Ada Generation compute instances',
   0.07, 'RTX 2000 Ada Generation', 'Hourly', 75, true,
   '{"generation": "Ada Lovelace", "professional": true}'::jsonb,
   0.07, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- L-series (Data Center)
  ('l40s', 'L40S GPU Package', 'NVIDIA L40S compute instances',
   0.34, 'L40S', 'Hourly', 70, true,
   '{"generation": "Ada Lovelace", "datacenter": true}'::jsonb,
   0.34, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('l40', 'L40 GPU Package', 'NVIDIA L40 compute instances',
   0.29, 'L40', 'Hourly', 65, true,
   '{"generation": "Ada Lovelace", "datacenter": true}'::jsonb,
   0.29, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('l4', 'L4 GPU Package', 'NVIDIA L4 compute instances',
   0.11, 'L4', 'Hourly', 60, true,
   '{"generation": "Ada Lovelace", "datacenter": true}'::jsonb,
   0.11, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- RTX A-series (Ampere generation professional)
  ('rtx_a6000', 'RTX A6000 Package', 'NVIDIA RTX A6000 compute instances',
   0.24, 'RTX A6000', 'Hourly', 55, true,
   '{"generation": "Ampere", "professional": true}'::jsonb,
   0.24, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('rtx_a5000', 'RTX A5000 Package', 'NVIDIA RTX A5000 compute instances',
   0.16, 'RTX A5000', 'Hourly', 50, true,
   '{"generation": "Ampere", "professional": true}'::jsonb,
   0.16, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('rtx_a4500', 'RTX A4500 Package', 'NVIDIA RTX A4500 compute instances',
   0.13, 'RTX A4500', 'Hourly', 45, true,
   '{"generation": "Ampere", "professional": true}'::jsonb,
   0.13, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('rtx_a4000', 'RTX A4000 Package', 'NVIDIA RTX A4000 compute instances',
   0.12, 'RTX A4000', 'Hourly', 40, true,
   '{"generation": "Ampere", "professional": true}'::jsonb,
   0.12, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- A-series data center (Ampere)
  ('a40', 'A40 GPU Package', 'NVIDIA A40 compute instances',
   0.12, 'A40', 'Hourly', 35, true,
   '{"generation": "Ampere", "datacenter": true}'::jsonb,
   0.12, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  ('a30', 'A30 GPU Package', 'NVIDIA A30 compute instances',
   0.10, 'A30', 'Hourly', 30, true,
   '{"generation": "Ampere", "datacenter": true}'::jsonb,
   0.10, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW()),

  -- GeForce RTX 3090 (Ampere consumer)
  ('rtx_3090', 'RTX 3090 GPU Package', 'NVIDIA GeForce RTX 3090 compute instances',
   0.13, 'GeForce RTX 3090', 'Hourly', 25, true,
   '{"generation": "Ampere", "consumer_grade": true}'::jsonb,
   0.13, 0.05, 0.0001,
   0.10, 0.05, 0.02, 0.05, 0.01,
   100, 50, 100, 10, 50,
   NOW())

ON CONFLICT (package_id) DO UPDATE SET
  name = EXCLUDED.name,
  description = EXCLUDED.description,
  hourly_rate = EXCLUDED.hourly_rate,
  gpu_model = EXCLUDED.gpu_model,
  priority = EXCLUDED.priority,
  is_active = EXCLUDED.is_active,
  metadata = EXCLUDED.metadata,
  base_rate_per_hour = EXCLUDED.base_rate_per_hour,
  cpu_rate_per_hour = EXCLUDED.cpu_rate_per_hour,
  disk_iops_rate = EXCLUDED.disk_iops_rate,
  storage_rate_per_gb_hour = EXCLUDED.storage_rate_per_gb_hour,
  network_rate_per_gb = EXCLUDED.network_rate_per_gb,
  disk_io_rate_per_gb = EXCLUDED.disk_io_rate_per_gb,
  cpu_rate_per_core_hour = EXCLUDED.cpu_rate_per_core_hour,
  memory_rate_per_gb_hour = EXCLUDED.memory_rate_per_gb_hour,
  included_storage_gb_hours = EXCLUDED.included_storage_gb_hours,
  included_network_gb = EXCLUDED.included_network_gb,
  included_disk_io_gb = EXCLUDED.included_disk_io_gb,
  included_cpu_core_hours = EXCLUDED.included_cpu_core_hours,
  included_memory_gb_hours = EXCLUDED.included_memory_gb_hours,
  updated_at = NOW();

COMMENT ON TABLE billing.billing_packages IS 'GPU billing packages - pricing updated 2025-10-22 (migration 022) - expanded GPU model support';
