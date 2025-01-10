import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np


class FrozenAgent:
    def __init__(self, state_size: int,
                action_size: int,
                learning_rate :float = 0.05,
                discount_factor :float = 0.95,
                exploration_epsilon :float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.qtable = np.zeros((state_size, action_size))

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.exploration_epsilon: # z prawdopodobieństwem epsilon wybieramy akcje losową
            return np.random.randint(self.action_size)
        return np.argmax(self.qtable[state, :])

    def update_qtable(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next_action = np.max(self.qtable[next_state, :])
        delta = reward + self.discount_factor * best_next_action - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * delta

    def update_epsilon(self) -> None:
        self.exploration_epsilon = max(0.1, self.exploration_epsilon * 0.9995)


def Qlearing(env, agent: FrozenAgent, max_steps=200, num_episodes=1000):
    rewards = np.zeros(num_episodes)
    e = 0
    while e < num_episodes:
        state, _ = env.reset()
        total_reward = 0

        for i in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_qtable(state, action, reward, next_state)
            total_reward += reward
            state = next_state

            if done:
                break

        rewards[e] += total_reward
        agent.update_epsilon()
        if (e % 50 == 0):
            print(f"{e} epizod")
        e += 1
    return rewards


def count_averaged_reward(env, agent, max_steps=200, num_episodes=1000,  num_of_ind_runs=25):
    averaged_reward = np.zeros(num_episodes)
    for i in range(num_of_ind_runs):
        averaged_reward += Qlearing(env, agent, max_steps, num_episodes)
    return averaged_reward / num_of_ind_runs


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
    agent = FrozenAgent(env.observation_space.n, env.action_space.n, learning_rate=0.8, discount_factor=0.95, exploration_epsilon=1.0)
    #averaged_reward_base = count_averaged_reward(env, agent)
    rewards = Qlearing(env, agent)
    print(rewards)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.plot(averaged_reward_base, 'r')
    #plt.show()

if __name__ == "__main__":
    main()

