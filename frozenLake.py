import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import random

MAP = '''SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFFFHFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG'''

MAP = [ state for state in MAP if state in ['S', 'F', 'H', 'G']]


class FrozenAgent:
    def __init__(self,
                learning_rate :float = 0.1,
                discount_factor :float = 0.95,
                exploration_epsilon :float = 1.0):
        self.env =  gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.qtable = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state: int) -> int:
        if (random.uniform(0,1) < self.exploration_epsilon
            or np.max(self.qtable[state, :]) == 0): # z prawdopodobieństwem epsilon wybieramy akcje losową
            return self.env.action_space.sample()
        return np.argmax(self.qtable[state, :])

    def update_qtable(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next_action = np.max(self.qtable[next_state, :])
        delta = reward + self.discount_factor * best_next_action - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * delta

    def update_epsilon(self, episode :int) -> None:
        self.exploration_epsilon = max(0.1, self.exploration_epsilon - 0.005*episode)


def reward_default(next_state):
    if next_state == 63:
        return 1
    else:
        return 0


def reward_hole_minus(next_state):
    reward = 0
    if MAP[next_state] == 'H':
        reward = -2
    elif MAP[next_state] == 'G':
        reward = 10
    return reward


def Qlearing(agent: FrozenAgent, max_steps=200, num_episodes=1000, rewarding=reward_default):
    rewards = np.zeros(num_episodes)
    e = 0
    while e < num_episodes:
        state, _ = agent.env.reset()

        for i in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = agent.env.step(action)

            reward = rewarding(next_state)
            agent.update_qtable(state, action, reward, next_state)
            state = next_state

            if done:
                break

        agent.update_epsilon(e)
        rewards[e] += reward
        if (e % 50 == 0):
            print(f"{e} epizod")
        e += 1
    return rewards


def count_averaged_reward(max_steps=200, num_episodes=1000,  num_of_ind_runs=25):
    averaged_reward = np.zeros(num_episodes)
    for i in range(num_of_ind_runs):
        agent = FrozenAgent()
        averaged_reward += Qlearing(agent, max_steps, num_episodes, reward_hole_minus)
    return averaged_reward / num_of_ind_runs


def main():
    averaged_reward_base = count_averaged_reward()
    averaged_reward = count_averaged_reward()

    print(averaged_reward_base)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(averaged_reward_base, 'r')
    plt.plot(averaged_reward, 'b')
    plt.show()

if __name__ == "__main__":
    main()

