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
    Calculate state-value function for the given policy

    Arguments:
        policy - policy of the agent, array of size (n_states, n_actions)
        transition_probabilities - dynamics of the environment, array of the size (n_states, n_actions, n_states, n_rewards )
        rewards - rewards for transition, array of the size (n_states, n_actions, n_states, n_rewards )
        gamma - discounting factor, float
        eps - accuracy of approximation

    Returns:
        state-value function

    '''
    n_states, _, _, _ = transition_probabilities.shape
    state_values = np.zeros((n_states, 1))

    while True:
        delta = 0
        for i in range(n_states):
            old = state_values[i][0]
            state_values[i] = policy[i]@np.sum(
                transition_probabilities[i]*(rewards[i] + gamma * state_values), axis=(1, 2)
            )
            delta += (state_values[i][0] - old)**2
        if delta < eps:
            break
    return state_values


def get_greedy_policy(
        state_values: npt.NDArray[np.float64],
        transition_probabilities: npt.NDArray[np.float64],
        rewards: npt.NDArray[np.float64],
        gamma: float = 0.9
) -> npt.NDArray[np.float64]:
    '''
    state_values - values of states, array of size (n_states, 1)
    transition_probabilities - dynamics of the environment, array of the size (n_states, n_actions, n_states, n_rewards )
    rewards - rewards for transition, size (n_states, n_actions, n_states, n_rewards )
    gamma - discounting factor, float
    '''
    n_states, n_actions, _, n_rewards = transition_probabilities.shape
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        q_value = np.sum(
            transition_probabilities[state]*(rewards[state] + gamma * state_values), axis=(1, 2))
        max_action = np.argmax(q_value, axis=0)
        policy[state][max_action] = 1
    return policy


def policy_iteration(
        transition_probabilities: npt.NDArray[np.float64],
        rewards: npt.NDArray[np.float64],
        gamma: float = 0.9,
        eps: float = 1e-4
) -> npt.NDArray[np.float64]:
    '''
    transition_probabilities - dynamics of the environment, array of the size (n_states, n_actions, n_states, n_rewards )
    rewards - rewards for transition, size (n_states, n_actions, n_states, n_rewards )
    gamma - discounting factor, float
    eps - accuracy of approximation

    '''
    n_states, n_actions, _, _ = transition_probabilities.shape

    policy = np.ones((n_states, n_actions), dtype=np.float64) / \
        n_actions  # We start from equiprobable policy

    state_values = policy_evaluation(
        policy=policy,
        transition_probabilities=transition_probabilities,
        rewards=rewards,
        gamma=gamma,
        eps=eps)

    while True:
        old_values = state_values
        policy = get_greedy_policy(transition_probabilities=transition_probabilities, rewards=rewards, gamma=gamma)

        state_values = policy_evaluation(
            policy=policy,
            transition_probabilities=transition_probabilities,
            rewards=rewards,
            gamma=gamma,
            eps=eps)
        if (np.linalg.norm(state_values - old_values) < eps):
            return state_values
