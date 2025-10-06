# CartPole Example

Gymnasium CartPole environment demonstrating integration with existing RL libraries
and proper state persistence handling.

## Local Testing

```bash
# Install dependencies
pip install basilica gymnasium[classic-control] numpy

# Set rental secret for local testing
export BASILICA_RENTAL_SECRET="test-secret-for-local-development"

# Run the service
python service.py
```

## Client Usage

```python
import basilica.afine as bs

with bs.create("user/cartpole:latest") as client:
    obs = client.reset(seed=42)
    print(f"Initial observation: {obs}")

    for step in range(100):
        action = 0 if obs[2] < 0 else 1
        obs, reward, terminated, truncated, info = client.step(action)

        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            break
```

## State Persistence

This example demonstrates proper handling of unpicklable objects (Gymnasium environments)
using `__getstate__` and `__setstate__` methods.
