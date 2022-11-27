import numpy as np
import numpy.typing as npt


def policy_evaluation(
        policy: npt.NDArray[np.float64],
        transition_probabilities: npt.NDArray[np.float64],
        rewards: npt.NDArray[np.float64],
        gamma: float = 0.9,
        eps: float = 1e-4
) -> npt.NDArray[np.float64]:
    '''
        policy - policy of the agent, array of size (n_states, n_actions)
        transition_probabilities - dynamics of the environment, array of the size (n_states, n_actions, n_states, n_rewards )
        rewards - rewards for transition, size (n_states, n_actions, n_states, n_rewards )

    '''
    n_states, n_actions, _, n_rewards = transition_probabilities.shape
    state_values = np.zeros((n_states, 1))
    
    
    while True:
        delta = 0
        for i in range(n_states):
            old = state_values[i][0]
            state_values[i] = policy[i]@np.sum(transition_probabilities[i]*(rewards[i] + gamma * state_values), axis=(1, 2))
            delta += (state_values[i][0] - old)**2
        if delta < eps:
            print(delta)
            break
    return state_values
