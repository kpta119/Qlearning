import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import random
from typing import Callable


class FrozenAgent:
    def __init__(self,
                learning_rate :float = 0.3,
                discount_factor :float = 0.9,
                epsilon :float = 1.0):
        self.env =  gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.qtable = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state: int) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        max_value = np.max(self.qtable[state, :])
        max_actions = [action for action in range(self.action_size) if self.qtable[state, action] == max_value]
        return random.choice(max_actions)


    def update_qtable(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next_action = np.max(self.qtable[next_state, :])
        delta = reward + self.discount_factor * best_next_action - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * delta


    def update_epsilon(self, episode :int) -> None:
        self.epsilon = max(0.001, self.epsilon - 0.001 * episode)


    def Qlearning(
            self,
            reward_function: Callable[[int, int, float, bool], int],
            rewards: np.array,
            num_of_episodes: int,
            num_of_steps: int
    ) -> np.ndarray:

        for episode in range(num_of_episodes):
            current_state, _ = self.env.reset()
            for _ in range(num_of_steps):
                action = self.choose_action(current_state)
                next_state, reward, done, _ , _ = self.env.step(action)

                reward = reward_function(current_state, next_state, reward, done)
                self.update_qtable(current_state, action, reward, next_state)
                current_state = next_state

                if done:
                    break

            self.update_epsilon(episode)
            if reward > 0:
                rewards[episode] += 1
        return rewards


def reward_default(state, next_state, reward, done) -> int:
    return reward


def reward_hole_minus(state, next_state, reward, done) -> int:
    if done and reward == 0:
        return -1
    elif done and reward == 1:
        return 100
    return 0


def minus_for_walking_into_walls_and_holes(state, next_state, reward, done) -> int:
    if done and reward == 0:
        return -2
    elif done and reward == 1:
        return 10
    elif next_state == state:
        return -2
    return 0



def count_averaged_reward(
    num_episodes: int,
    num_of_ind_runs: int = 25,
    learning_rate: float = 0.3,
    discount_factor: float = 0.9,
    max_steps: int = 200,
    rewarding: Callable[[int, int, float, bool], int] = reward_default
) -> np.ndarray:
    averaged_reward = np.zeros(num_episodes)
    for i in range(num_of_ind_runs):
        agent = FrozenAgent(learning_rate=learning_rate, discount_factor=discount_factor)
        agent.Qlearning(rewarding, averaged_reward, num_episodes, max_steps)
    return averaged_reward / num_of_ind_runs


def main():
    num_of_episodes = 1000
    rewarding=reward_hole_minus
    averaged_reward_base = count_averaged_reward(num_episodes=num_of_episodes)
    averaged_reward = count_averaged_reward(num_episodes=num_of_episodes, rewarding=rewarding)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    averaged_reward_base = np.convolve(averaged_reward_base, np.ones(20) / 20, 'valid')
    averaged_reward = np.convolve(averaged_reward, np.ones(20) / 20, 'valid')
    plt.xlim(0, num_of_episodes)
    plt.plot(averaged_reward_base, 'r',  label='Default rewarding')
    plt.plot(averaged_reward, 'b', label='Minus for walking into holes')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8)
    plt.show()

if __name__ == "__main__":
    main()

