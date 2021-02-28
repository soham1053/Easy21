import numpy as np
from collections import defaultdict
from easy21 import Env
import utils


def epsilon_greedy(state, Q, epsilon, nA):
    """
    Uses an epsilon-greedy policy based on a given Q-function and epsilon to choose probability distribution of
    actions based on a state

    :param state: the agent's view of the environment at one timestep
    :param Q: state-action to value map
    :param epsilon: probability that the selects a random action, else a greedy action
    :param nA: number of possible actions
    :return: probability distribution of all actions for the state
    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A


def mc_control(env, num_episodes, gamma=1.0, No=100):
    """
    Monte Carlo Control that finds the optimal epsilon greedy policy

    :param env: the environment in which the agent takes actions
    :param num_episodes: number of episodes to sample
    :param gamma: discount factor
    :param epsilon: probability that the policy chooses a random action at any given moment.
    :return: tuple containing final Q table and policy
    """

    returns_sum = defaultdict(float)
    sa_count_offline = defaultdict(float)
    s_count_online = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(len(env.action_space)))

    for e in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        while not done:
            s_count_online[state] += 1
            epsilon = No / (No + s_count_online[state])

            probs = epsilon_greedy(state, Q, epsilon, len(env.action_space))
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


env = Env()
print("Starting training.")
Q = mc_control(env, num_episodes=100000, gamma=0.7)
print("Training finished.\n")

iterations = 10000
wins = 0
print("Starting testing.")
for _ in range(iterations):
    state = env.reset()
    while True:
        probs = epsilon_greedy(state, Q, epsilon=0, nA=len(env.action_space))
        actionIdx = np.random.choice(np.arange(len(probs)), p=probs)
        action = env.action_space[actionIdx]

        nextState, reward, done = env.step(action)

        if done:
            wins += 1 if reward == 1 else 0
            break

        state = nextState

score = wins / iterations * 100
print("Testing finished.\n")
print(f"Percentage of wins against dealer over {iterations} games: {score}.")

utils.plot(Q, np.arange(len(env.action_space)))
