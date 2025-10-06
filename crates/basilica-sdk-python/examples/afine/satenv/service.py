"""
SAT Solver Environment - Demonstrates stateful multi-turn interaction.
"""

import random

import basilica.afine as bs


class SATEnv(bs.Service[str, int]):
    """Simple arithmetic question environment."""

    def reset(self, seed: int | None = None) -> str:
        """Reset with a new random arithmetic question."""
        if seed is not None:
            random.seed(seed)
        self.a = random.randint(1, 10)
        self.b = random.randint(1, 10)
        return f"{self.a} + {self.b}"

    def step(self, answer: int) -> tuple[None, float, bool, bool, dict]:
        """Check if the answer is correct."""
        correct = (answer == self.a + self.b)
        reward = 1.0 if correct else -1.0
        return None, reward, True, False, {"correct": correct}


if __name__ == "__main__":
    bs.serve(SATEnv)
