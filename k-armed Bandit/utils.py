import numpy as np

def experiment_run(env, agent, n_steps = 1000):
    rewards = np.zeros(n_steps, dtype=np.float64)
    actions = np.zeros(n_steps, dtype=np.float64)
    obs, info = env.reset()
    for step in range(0, n_steps):
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, truncated)
        rewards[step] = reward
        actions[step] = (action == env.get_winning_arm())
    return actions, rewards

def experiment(**kwargs):
    print(kwargs)
    n_step = kwargs["n_step"]
    n_runs = kwargs["n_runs"]
    total_rewards = np.zeros(n_step, dtype=np.float64)
    total_optimal_actions = np.zeros(n_step, dtype=np.float64)
    for run in range(n_runs):
        game = kwargs["env"]()
        agent = kwargs["agent"](game, **kwargs["agent_params"])
        optimal_actions, reward = experiment_run(game, agent, n_step)
        total_rewards += reward
        total_optimal_actions += optimal_actions
    return total_optimal_actions / n_runs,  total_rewards / n_runs