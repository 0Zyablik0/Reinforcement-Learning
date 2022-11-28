import numpy as np


class GridWorld:
    __default_n_rows__ = 5
    __default_n_cols__ = 5
    __default_reward__ = 0
    __default_off_grid__reward__ = -1
    n_actions = 4
    n_rewards = 1
    actions = {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3,
    }

    def __init__(self, **kwargs) -> None:
        self.n_rows = kwargs["n_rows"] if "n_rows" in kwargs else self.__default_n_rows__
        self.n_cols = kwargs["n_cols"] if "n_cols" in kwargs else self.__default_n_cols__
        self.step_reward = kwargs["step_reward"] if "step_reward" in kwargs else self.__default_reward__
        self.off_grid_reward = kwargs["off_grid_reward"] if "off_grid_reward" in kwargs else self.__default_off_grid__reward__
        self.n_states = self.n_rows * self.n_cols
        self.special_transitions = kwargs["special_transitions"] if "special_transitions" in kwargs else {
        }
        self.probabilities = np.zeros(
            (self.n_states, self.n_actions, self.n_states, self.n_rewards), dtype=np.float64)
        self.rewards = np.zeros(
            (self.n_states, self.n_actions, self.n_states, self.n_rewards), dtype=np.float64)
        self._calculate_probabilities()
    
    def get_dynamics(self):
        return self.probabilities, self.rewards

    def _state_to_coords(self, state):
        if state >= self.n_states or state < 0:
            raise ValueError("This state={state} is outside of the GridWorld")

        x = state % self.n_cols
        y = state // self.n_cols
        return (x, y)

    def _is_top(self, x, y):
        return y == 0

    def _is_bottom(self, x, y):
        return y == self.n_rows - 1

    def _is_left_border(self, x, y):
        return x == 0

    def _is_right_border(self, x, y):
        return x == self.n_cols - 1

    def _coords_to_state(self, x, y):
        if (x >= self.n_cols or x < 0):
            raise ValueError("coordinate x={x} is outside of the GridWorld")
        if (y >= self.n_rows or y < 0):
            raise ValueError("coordinate y={y} is outside of the GridWorld")
        return y*self.n_cols + x

    def _calculate_probabilities(self):
        for state in range(self.n_states):
            coords = self._state_to_coords(state)
            if coords in self.special_transitions:
                new_state, reward = self.special_transitions[coords]
                new_state = self._coords_to_state(*new_state)
                for _, action in self.actions.items():
                    self.probabilities[state, action, new_state, 0] = 1
                    self.rewards[state, action, new_state, 0] = reward
                continue
            if self._is_top(*coords):
                self.probabilities[state, self.actions["up"], state, 0] = 1
                self.rewards[state, self.actions["up"], state, 0] = self.step_reward + self.off_grid_reward
            else:
                self.probabilities[state, self.actions["up"], state - self.n_cols, 0] = 1
                self.rewards[state, self.actions["up"], state - self.n_cols, 0] = self.step_reward

            if self._is_bottom(*coords):
                self.probabilities[state, self.actions["down"], state, 0] = 1
                self.rewards[state, self.actions["down"], state, 0] = self.step_reward + self.off_grid_reward
            else:
                self.probabilities[state, self.actions["down"], state + self.n_cols, 0] = 1
                self.rewards[state, self.actions["down"], state + self.n_cols, 0] = self.step_reward

            if self._is_left_border(*coords):
                self.probabilities[state, self.actions["left"], state, 0] = 1
                self.rewards[state, self.actions["left"], state, 0] = self.step_reward + self.off_grid_reward
            else:
                self.probabilities[state, self.actions["left"], state - 1, 0] = 1
                self.rewards[state, self.actions["left"], state - 1, 0] = self.step_reward

            if self._is_right_border(*coords):
                self.probabilities[state, self.actions["right"], state, 0] = 1
                self.rewards[state, self.actions["right"], state, 0] = self.step_reward + self.off_grid_reward
            else:
                self.probabilities[state, self.actions["right"], state + 1, 0] = 1
                self.rewards[state, self.actions["right"], state + 1, 0] = self.step_reward
