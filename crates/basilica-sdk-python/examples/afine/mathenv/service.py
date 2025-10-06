"""
Simple Math Environment - Demonstrates basic AFINE SDK usage.
"""

import basilica.afine as bs


class MathEnv(bs.Service[tuple[int, int], int]):
    """Simple arithmetic environment for testing."""

    def reset(self, seed: int | None = None) -> tuple[int, int]:
        """Reset environment with initial values."""
        self.x, self.y = 1, 1
        return (self.x, self.y)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, bool, dict]:
        """Take a step by adding action to x."""
        self.x += action
        return (self.x, self.y), 1.0, False, False, {}


if __name__ == "__main__":
    bs.serve(MathEnv)
