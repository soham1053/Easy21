import numpy as np
from easy21 import Env
from collections import defaultdict
import utils
import dill as pickle


def mc_control(env, num_episodes, gamma=1.0, No=100):
    """
    Monte Carlo Control that finds the optimal epsilon greedy policy

    :param env: the environment in which the agent takes actions
    :param num_episodes: number of episodes to sample
    :param gamma: discount factor
    :param epsilon: probability that the policy chooses a random action at any given moment.
    :param No: Determines scale of epsilon and learning rate
    :return: Default dict representation of the Q table
    """

    returns_sum = defaultdict(float)
    sa_count_offline = defaultdict(float)
    s_count_online = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(len(env.action_space)))

    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        while not done:
            s_count_online[state] += 1
            epsilon = No / (No + s_count_online[state])

            probs = utils.epsilon_greedy(state, Q, epsilon, len(env.action_space))
            actionIdx = np.random.choice(np.arange(len(probs)), p=probs)
            action = env.action_space[actionIdx]

            old = state
            state, reward, done = env.step(action)

            episode.append((old, actionIdx, reward))

        G = 0
        for i, (state, actionIdx, reward) in enumerate(reversed(episode)):
            sa_pair = (state, actionIdx)

            G = G*gamma + reward
            returns_sum[sa_pair] += G

            sa_count_offline[sa_pair] += 1
            error = G - Q[state][actionIdx]
            alpha = 1/sa_count_offline[sa_pair]
            Q[state][actionIdx] += alpha * error

    return Q


if __name__ == "__main__":
    env = Env()
    print("Training.")
    Q = mc_control(env, num_episodes=1000000)
    print()

    iterations = 10000
    wins = 0
    print("Testing.")
    for _ in range(iterations):
        state = env.reset()
        while True:
            probs = utils.epsilon_greedy(state, Q, epsilon=0, nA=len(env.action_space))
            actionIdx = np.random.choice(np.arange(len(probs)), p=probs)
            action = env.action_space[actionIdx]

            nextState, reward, done = env.step(action)

            if done:
                wins += 1 if reward == 1 else 0
                break

            state = nextState

    score = wins / iterations * 100
    print()
    print(f"Percentage of wins against dealer over {iterations} games: {score}.")

    pickle.dump(Q, open('Q.dill', 'wb'))

    utils.plotQ(Q, np.arange(len(env.action_space)))
