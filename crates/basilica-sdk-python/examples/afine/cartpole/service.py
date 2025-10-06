"""
CartPole Environment - Demonstrates Gymnasium integration with state persistence.
"""

import gymnasium as gym
import numpy as np

import basilica.afine as bs


class CartPoleEnv(bs.Service):
    """Gymnasium CartPole environment wrapper."""

    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self, seed: int | None = None):
        """Reset the CartPole environment."""
        obs, info = self.env.reset(seed=seed)
        return obs.tolist()

    def step(self, action: int):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.tolist(), float(reward), terminated, truncated, info

    def close(self):
        """Clean up the environment."""
        self.env.close()

    def __getstate__(self):
        """Don't serialize Gymnasium environment."""
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __setstate__(self, state):
        """Recreate Gymnasium environment on deserialization."""
        self.__dict__.update(state)
        self.env = gym.make('CartPole-v1')


if __name__ == "__main__":
    bs.serve(CartPoleEnv)
