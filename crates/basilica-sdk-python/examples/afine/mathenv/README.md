# MathEnv Example

Simple arithmetic environment demonstrating basic AFINE SDK usage.

## Local Testing

```bash
# Set rental secret for local testing
export BASILICA_RENTAL_SECRET="test-secret-for-local-development"

# Run the service
python service.py
```

## Client Usage

```python
import basilica.afine as bs

with bs.create("user/mathenv:latest") as client:
    obs = client.reset()
    print(f"Initial state: {obs}")

    obs, reward, terminated, truncated, info = client.step(3)
    print(f"After step: {obs}, reward: {reward}")
```
