from gymnasium import Env
import numpy as np
from collections import defaultdict


class SimpleAgent():
    def __init__(self, environment: Env, epsilon: float = 0.0):
        self.env = environment
        self.q_values = defaultdict(lambda: np.zeros(
            (self.env.action_space.n), dtype=np.float64))
        self.epsilon = epsilon
        self.times = defaultdict(lambda: np.zeros(
            (self.env.action_space.n), dtype=np.float64))

    def get_action(self, observation: int) -> int:
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(self.q_values[observation]))
        return action

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        terminated: bool = False,
        truncated: bool = False,
    ):
        self.times[observation][action] += 1
        self.q_values[observation][action] += 1/self.times[observation][action] * \
            (reward - self.q_values[observation][action])
