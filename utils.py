import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot(Q, actions):
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