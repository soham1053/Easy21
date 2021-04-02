import numpy as np
from easy21 import Env
from collections import defaultdict
import utils
import dill as pickle

NUM_EPISODES = 10000
GAMMA = 1.0
NO = 100
MSE_UPDATE = 1

def sarsa_lambda(env, lambd, QStar, num_episodes=NUM_EPISODES, gamma=GAMMA, No=NO, mse_update=MSE_UPDATE):
    """
    Sarsa Lambda Control that finds the optimal epsilon greedy policy

    :param env: the environment in which the agent takes actions
    :param lambd: Lambda in Sarsa lambda algorithm
    :param QStar: Optimal Q table
    :param num_episodes: number of episodes to sample
    :param gamma: discount factor
    :param No: Determines scale of epsilon and learning rate
    :return: Default dict representation of the Q table, and the MSE's between Q and QStar for each episode
    """

    s_count = defaultdict(float)
    sa_count = defaultdict(lambda: np.zeros(len(env.action_space)))

    Q = defaultdict(lambda: np.zeros(len(env.action_space)))

    QStar_sa_count = len(QStar) * len(env.action_space)
    mse = np.zeros(num_episodes // mse_update)

    for i in range(num_episodes):
        e = defaultdict(lambda: np.zeros(len(env.action_space)))

        state = env.reset()

        s_count[state] += 1
        epsilon = No / (No + s_count[state])

        probs = utils.epsilon_greedy_table(state, Q, epsilon, len(env.action_space))
        actionIdx = np.random.choice(np.arange(len(probs)), p=probs)
        action = env.action_space[actionIdx]

        sa_count[state][actionIdx] += 1

        done = False
        while not done:
            nextState, reward, done = env.step(action)

            s_count[nextState] += 1
            epsilon = No / (No + s_count[nextState])

            probs = utils.epsilon_greedy_table(nextState, Q, epsilon, len(env.action_space))
            nextActionIdx = np.random.choice(np.arange(len(probs)), p=probs)
            nextAction = env.action_space[nextActionIdx]

            sa_count[nextState][nextActionIdx] += 1

            if not done:
                tdError = reward + gamma*Q[nextState][nextActionIdx] - Q[state][actionIdx]
            else:
                tdError = reward - Q[state][actionIdx]

            e[state][actionIdx] += 1

            for s in Q.keys():
                for a in range(len(Q[s])):
                    if sa_count[s][a] == 0:
                        continue

                    alpha = 1 / sa_count[s][a]
                    Q[s][a] += alpha * tdError * e[s][a]

                    e[s][a] *= (gamma * lambd)

            if done:
                break

            state = nextState
            action = nextAction
            actionIdx = nextActionIdx
        if i % mse_update == 0:
            for s in QStar.keys():
                for a in range(len(QStar[s])):
                    mse[i // mse_update] += (Q[s][a] - QStar[s][a]) ** 2
            mse[i // mse_update] /= QStar_sa_count

    return Q, mse


if __name__ == "__main__":
    env = Env()

    QStar = pickle.load(open('Q.dill', 'rb'))

    lambds = list(np.arange(0, 1.1, 0.1))
    mseLambdas = np.zeros((len(lambds), NUM_EPISODES // MSE_UPDATE))
    finalMSE = np.zeros(len(lambds))

    print("Training.")
    mse = defaultdict(float)
    for i, lambd in enumerate(lambds):
        Q, mse = sarsa_lambda(env, lambd, QStar)
        mseLambdas[i] = mse
        finalMSE[i] = mse[-1]

        print(f"Lambda {lambd}: Final MSE {mse[-1]}")
    print()

    print("Plotting.")
    utils.plotMseLambdas(finalMSE, lambds)
    utils.plotMseEpisodesLambdas(mseLambdas)
