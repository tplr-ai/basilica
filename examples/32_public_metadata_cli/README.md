# Public Deployment Metadata - CLI

Expose non-sensitive deployment metadata publicly for validator verification.

This enables Bittensor subnet validators to verify what miners have deployed
without requiring authentication.

## Setup

```bash
basilica login
# or
export BASILICA_API_TOKEN="your-token"
```

## 1. Deploy with Public Metadata Enabled

Enable metadata enrollment at deployment creation with `--public-metadata`:

```bash
basilica deploy hashicorp/http-echo:latest \
  --name my-verified-app \
  --port 5678 \
  --public-metadata
```

## 2. Check Enrollment Status

```bash
basilica deploy enroll-metadata my-verified-app
```

Output:

```
Public Metadata: Enrolled
  Metadata is publicly visible for validator verification.
```

## 3. View Public Metadata (No Auth Required)

Anyone can query public metadata for enrolled deployments:

```bash
basilica deploy metadata my-verified-app
```

Output:

```
Public Deployment Metadata: my-verified-app

  Image:    hashicorp/http-echo:latest
  ID:       dep-abc123
  State:    Active
  Replicas: 1/1
  Uptime:   2h 15m
```

JSON output for scripting:

```bash
basilica deploy metadata my-verified-app --json
```

## 4. Enroll an Existing Deployment

Enable metadata for a deployment that was created without `--public-metadata`:

```bash
basilica deploy enroll-metadata my-app --enable
```

## 5. Disable Enrollment

```bash
basilica deploy enroll-metadata my-app --disable
```

## 6. Interactive Selection

Omit the name to select from your active deployments:

```bash
basilica deploy enroll-metadata --enable
```

## Cleanup

```bash
basilica summon delete my-verified-app
```
