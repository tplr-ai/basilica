# Basilica Localnet Test Report

## Test Date: 2025-07-02

## Overall Status: ✅ WORKING (with minor issues)

## Service Status

| Service | Status | Health | Notes |
|---------|--------|--------|-------|
| Subtensor (Alice) | ✅ Running | Healthy | Blockchain accessible on ws://localhost:9944 |
| Executor | ✅ Running | Healthy | gRPC on 50052, configured for GPU support |
| Miner | ✅ Running | Healthy | Successfully monitoring executor, skip_registration=true |
| Validator | ✅ Running | Healthy | Running with --local-test flag |
| Redis | ✅ Running | Healthy | Cache available on port 6379 |
| Prometheus | ✅ Running | Running | Metrics collection active |
| Grafana | ✅ Running | Healthy | Dashboard accessible on http://localhost:3000 |
| Public API | ❌ Restarting | Failed | Route conflict and metadata compatibility issues |

## Key Features Verified

1. **Registration Bypass**: Both miner and validator successfully bypass Bittensor registration
2. **Service Communication**: Miner successfully connects to and monitors executor health
3. **Database Initialization**: Both miner and validator successfully initialize SQLite databases
4. **GPU Support**: Executor configured with GPU support (will use if available)
5. **Network Isolation**: All services communicate within Docker network

## Known Issues

1. **Public API**: Crashes due to:
   - Overlapping OpenAPI routes (`/api-docs/openapi.json`)
   - Metadata incompatibility with local Subtensor
   
2. **Metrics Endpoints**: Some metrics endpoints not accessible from test script (but services are healthy)

## Test Commands Used

```bash
# Setup and start all services
./setup.sh

# Check service status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test service health
./test-services.sh

# Check individual service logs
docker logs basilica-miner-localnet --tail 20
docker logs basilica-validator-localnet --tail 20
docker logs basilica-executor-localnet --tail 20
```

## Conclusion

The localnet setup is working correctly for development and testing purposes. The core services (Executor, Miner, Validator) are all functioning properly with registration bypass enabled. The Public API requires code fixes to work with localnet but is not critical for basic testing.