# SATEnv Example

Simple SAT-style question environment demonstrating stateful interaction.

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

with bs.create("user/satenv:latest") as client:
    question = client.reset(seed=42)
    print(f"Question: {question}")

    _, reward, done, _, info = client.step(5)
    print(f"Reward: {reward}, Correct: {info['correct']}")
```
