import numpy as np
from easy21 import Env
from collections import defaultdict
import utils
import dill as pickle

EPSILON = 0.05
ALPHA = 0.01
NUM_EPISODES = 10000
GAMMA = 1.0
MSE_UPDATE = 1
dealer_features = [[1, 4], [4, 7], [7, 10]]
agent_features = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]


def phi(state, action):
    """
    :param state: current state of environment
    :param action: action -- hit or stick
    :return: one-hot encoded features of the state-action pair for linear function Q approximation
    """
    d_sum = state[0]
    a_sum = state[1]

    features = np.zeros((3, 6, 2), dtype=np.int)

    d_features = np.array([x[0] <= d_sum <= x[1] for x in dealer_features])
    a_features = np.array([x[0] <= a_sum <= x[1] for x in agent_features])

    action = 1 if action == 'hit' else 0
    for i in np.where(d_features):
        for j in np.where(a_features):
            features[i, j, action] = 1

    return features.flatten()

class QLinear:
    def __init__(self, input_size):
        self.w = np.random.randn(input_size) / 100
    def forward(self, feature):
        """
        Calculates the Q values of a feature with a linear approximation

        :param feature: The features of a state in the environment
        :return: The Q value of the feature, meaning how much long term the agent will be expected to get
        """
        return np.sum(self.w * feature)

    def take_action(self, state, epsilon, action_space):
        """
        Given an epsilon greedy policy, takes a weighted random action

        :param state: State of an environment
        :param epsilon: Epsilon in epsilon greedy exploration
        :action_space: All possible actions in the environment
        """
        probs = self.epsilon_greedy_approx(state, epsilon, action_space)
        actionIdx = np.random.choice(np.arange(len(probs)), p=probs)
        action = action_space[actionIdx]
        return action

    def epsilon_greedy_approx(self, state, epsilon, action_space):
        """
        Uses an epsilon-greedy policy based on the Q function approximations and epsilon to choose probability
        distribution of actions based on a state

        :param state: current state of the environment
        :param epsilon: probability that the selects a random action, else a greedy action
        :param action_space: all possible actions
        :return: probability distribution of all actions for the state
        """
        nA = len(action_space)
        A = np.ones(nA, dtype=float) * epsilon / nA
        features = [phi(state, action) for action in action_space]
        q_values = np.array([self.forward(feature) for feature in features])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def update(self, alpha, delta, E):
        """
        Update weights towards a better Q function approximation

        :param alpha: Learning rate
        :param delta: TD Error
        :param E: Eligibility traces of the features
        """
        self.w += alpha * delta * E


def func_approx(env, lambd, QStar, epsilon=EPSILON, alpha=ALPHA, num_episodes=NUM_EPISODES, gamma=GAMMA, mse_update=MSE_UPDATE):
    """
    Function Approximation with Sarsa Lambda Control that finds the optimal epsilon greedy policy

    :param env: the environment in which the agent takes actions
    :param lambd: Lambda in Sarsa lambda algorithm
    :param QStar: Optimal Q table
    :param epsilon: Probability the agent takes a random action while training
    :param alpha: Training step size
    :param num_episodes: number of episodes to sample
    :param gamma: discount factor
    :param mse_update: how often to update the mean squared error cache
    :return: Linear function representation of the Q table, and the MSE's between Q and QStar for each episode
    """
    Q = QLinear(36)
    mse = np.zeros(num_episodes // mse_update)

    QStar_sa_count = len(QStar) * len(env.action_space)

    for i in range(num_episodes):
        # initial eligibility traces
        E = np.zeros([36,])
        state = env.reset()
        action = Q.take_action(state, epsilon, env.action_space)

        done = False
        # sample until terminal
        while not done:
            # run one step
            nextState, reward, done = env.step(action)

            if done:
                # if the episode is in terminal state, Q[s', a'] is 0
                q0 = Q.forward(phi(state, action))
                delta = reward - q0
                nextAction = 'hit'

            else:
                nextAction = Q.take_action(nextState, epsilon, env.action_space)
                q0 = Q.forward(phi(state, action))
                q1 = Q.forward(phi(nextState, nextAction))
                delta = reward + gamma * q1 - q0

            E = E * (gamma * lambd) + phi(state, action)
            Q.update(alpha, delta, E)

            state, action = nextState, nextAction

        if i % mse_update == 0:
            for s in QStar.keys():
                for a in range(len(QStar[s])):
                    mse[i//mse_update] += (Q.forward(phi(s, env.action_space[a])) - QStar[s][a]) ** 2
            mse[i//mse_update] /= QStar_sa_count

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
        Q, mse = func_approx(env, lambd, QStar)
        mseLambdas[i] = mse
        finalMSE[i] = mse[-1]

        print(f"Lambda {lambd}: Final MSE {mse[-1]}")
    print()

    print("Plotting.")
    utils.plotMseLambdas(finalMSE, lambds)
    utils.plotMseEpisodesLambdas(mseLambdas)
