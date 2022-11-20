import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple
import numpy.typing as npt


class k_BanditGame(gym.Env):
    def __init__(self, k_hands: int = 10):
        self.k_hands = k_hands
        self.observation_space = spaces.Discrete(1)  # always ready to play
        self.action_space = spaces.Discrete(self.k_hands)
        self.mean_rewards = (np.random.normal(size=self.k_hands))
        self.max_mean_reward = self.mean_rewards
        self.winning_arm = int(np.argmax(self.mean_rewards))
        self.max_mean_reward = float(np.max(self.mean_rewards))

    def get_mean_rewards(self) -> npt.NDArray[np.float64]:
        return self.mean_rewards

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        info = {}
        return self.observation_space.sample(), info

    def get_max_mean_reward(self) -> float:
        return self.max_mean_reward

    def get_winning_arm(self) -> int:
        return self.winning_arm

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        observation = self.observation_space.sample()
        reward = np.random.normal(loc=self.mean_rewards[action])
        terminated = False
        truncated = False
        info = {
            "k_bandits": self.k_hands,
            "arm": action,
            "mean reward": self.mean_rewards[action],
        }
        return observation, reward, terminated, truncated, info
