import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def epsilon_greedy_table(state, Q, epsilon, nA):
    """
    Uses an epsilon-greedy policy based on a given Q-table and epsilon to choose probability distribution of
    actions based on a state

    :param state: the agent's view of the environment at one timestep
    :param Q: state-action to value table
    :param epsilon: probability that the selects a random action, else a greedy action
    :param nA: number of possible actions
    :return: probability distribution of all actions for the state
    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A

def plotQ(Q, actions):
    """
    Plots given Q table of Easy21 environment
    Credits to timbmg (https://github.com/timbmg/easy21-rl)

    :param Q: Default dict representation of Q table
    :param actions: Possible actions in environment
    """
    pRange = list(range(1,22))
    dRange = list(range(1,11))
    vStar = list()
    for p in pRange:
        for d in dRange:
            vStar.append([p, d, np.max([Q[(d, p)][a] for a in actions])])

    df = pd.DataFrame(vStar, columns=['player', 'dealer', 'value'])

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Dealer First Card')
    ax.set_ylabel('Player End Total')
    ax.set_zlabel('Expected Return')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()

    # to Add a color bar which maps values to colors.
    surf=ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)
    plt.show()

    # Rotate it
    ax.view_init(30, 45)
    plt.show()

    # Other palette
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()


def plotMseLambdas(mse, lambds):
    """
    Plots lambda vs mean squared error of Sarsa(lambda) value functions

    :param mse: mean squared error of each lambda's value function
    :param lambds: each lambda value
    """

    plt.plot(lambds, mse)
    plt.title("Lambda vs MSE")
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error")
    plt.show()


def plotMseEpisodesLambdas(arr):
    """
    Plots training MSE's for all episodes and lambdas with the Sarsa(lambda) algorithm
    Credits to timbmg (https://github.com/timbmg/easy21-rl)

    :param arr: for each lambda, a history of MSE's per episode
    """

    # https://stackoverflow.com/questions/45857465/create-a-2d-array-from-another-array-and-its-indices-with-numpy
    m,n = arr.shape
    I,J = np.ogrid[:m,:n]
    out = np.empty((m,n,3), dtype=arr.dtype)
    out[...,0] = I
    out[...,1] = J
    out[...,2] = arr
    out.shape = (-1,3)

    df = pd.DataFrame(out, columns=['lambda', 'Episode', 'MSE'])
    df['lambda'] = df['lambda'] / 10
    #df = df.loc[df.index % 100 == 0]
    g = sns.FacetGrid(df, hue="lambda", height=8, legend_out=True)
    #g.map(plt.scatter, "episode", "mse")
    g = g.map(plt.plot, "Episode", "MSE").add_legend()

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Mean Squared Error per Episode')

    plt.show()
