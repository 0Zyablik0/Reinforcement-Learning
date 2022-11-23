from gymnasium import Env
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy.special import softmax


class SimpleAgent():
    def __init__(self, environment: Env, **kwargs):
        self.env = environment
        self.q_values = defaultdict(lambda: np.zeros(
            (self.env.action_space.n), dtype=np.float64))
        self.times = defaultdict(lambda: np.zeros(
            (self.env.action_space.n), dtype=np.float64))
        self.n_step = 0
        
        if  "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]
        else:
            self.epsilon = 0.0

        if "step_size" in kwargs:
            self.step_size = kwargs["step_size"]
        else:
            self.step_size = 1.0

        if "step_policy" in kwargs:
            self.step_policy = kwargs["step_policy"]    
        else:
            self.step_policy = "average"

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
        self.n_step += 1
        self.times[observation][action] += 1
        match self.step_policy:
            case "average":
                step_size = 1/self.times[observation][action]
            case "constant":
                step_size = self.step_size
            case _:
                step_size = 1/self.times[observation][action]
        self.q_values[observation][action] = self.q_values[observation][action] + step_size * \
            (reward - self.q_values[observation][action])


class OptimisticSimpleAgent(SimpleAgent):
    def __init__(self, environment: Env, **kwargs):
        super().__init__(environment, **kwargs)
        self.q_values = defaultdict(lambda: kwargs["optimistic_value"] * np.ones(
            (self.env.action_space.n), dtype=np.float64))


class UCBSimpleAgent(SimpleAgent):
    def __init__(self, environment: Env, **kwargs):
        super().__init__(environment, **kwargs)
        if "ucb_c" in kwargs:
            self.ucb_c = kwargs["ucb_c"]
        else:
            self.ucb_c = 1.0
    
    def get_action(self, observation: int) -> int:
        if np.min(self.times[observation]) == 0:
            return np.argmin(self.times[observation])
        ucb_value = self.q_values[observation]+ self.ucb_c * np.sqrt(np.log(self.n_step)/self.times[observation])
        action = int(np.argmax(ucb_value))
        return action
    
class GradientAgent():
    def __init__(self, environment: Env, **kwargs):
        self.env = environment
        self.avg_reward = 0
        self.preferences = defaultdict(lambda: np.zeros((self.env.action_space.n), dtype=np.float64))
        self.n_step = 0
        
        self.baseline = False
        if  "baseline" in kwargs:
            self.baseline= kwargs["baseline"]
        if "step_size" in kwargs:
            self.step_size = kwargs["step_size"]
        else:
            self.step_size = 0.1

    def _get_action_probability(self, observation: int) -> npt.NDArray[np.float64]:
        return softmax(self.preferences[observation])
    
    def get_action(self, observation: int) -> int:
        probabilities = self._get_action_probability(observation)
        action = np.random.choice(self.env.action_space.n, p =probabilities)
        return action

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        terminated: bool = False,
        truncated: bool = False,
    ):
        
        self.n_step += 1
        if self.baseline:
            self.avg_reward += (reward - self.avg_reward)/self.n_step
        probabilities = self._get_action_probability(observation)
        self.preferences[observation] -= self.step_size*(reward - self.avg_reward)*probabilities[action]
        self.preferences[observation][action] += self.step_size*(reward - self.avg_reward)