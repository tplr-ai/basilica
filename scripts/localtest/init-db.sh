#!/bin/bash
# Initialize database directories for localtest

echo "Initializing database directories..."

# Create validator database directory
docker exec basilica-validator mkdir -p /var/lib/basilica/validator || true
docker exec basilica-validator touch /var/lib/basilica/validator/validator.db || true
docker exec basilica-validator chmod 666 /var/lib/basilica/validator/validator.db || true

# Create miner database directory
docker exec basilica-miner mkdir -p /var/lib/basilica/miner || true
docker exec basilica-miner touch /var/lib/basilica/miner/miner.db || true
docker exec basilica-miner chmod 666 /var/lib/basilica/miner/miner.db || true

echo "Database directories initialized"