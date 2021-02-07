from typing import Any, Union

import numpy as np


def start():
    return {'dealerFirstCard': np.random.randint(1, 11), 'playerSum': np.random.randint(1, 11)}, 0


def draw():
    color = np.random.choice(['black', 'red'], p=(2/3, 1/3))
    if color == 'black':
        card_val = np.random.randint(1, 11)
    elif color == 'red':
        card_val = -np.random.randint(1, 11)

    return card_val


def step(s, a):
    """
    :param s: state -- dealer's first card 1-10 and player's sum 1-21
    :param a: action -- hit or stick
    :return: sample of next state (which may be terminal if game is finished) and reward
    """
    if s == 'terminal':
        print('The game has already ended.')
        return 'terminal', 0

    if a == 'hit':
        card_val = draw()
        player_sum = s['playerSum'] + card_val
        if player_sum > 21 or player_sum < 1:
            return 'terminal', -1
        else:
            return {'dealerFirstCard': s['dealerFirstCard'], 'playerSum': player_sum}, 0

    elif a == 'stick':
        dealer_sum = s['dealerFirstCard']
        while 1 <= dealer_sum < 17:
            card_val = draw()
            dealer_sum += card_val

        if dealer_sum > 21 or dealer_sum < 1:
            return 'terminal', 1

        player_sum = s['playerSum']
        if player_sum > dealer_sum:
            return 'terminal', 1
        elif player_sum == dealer_sum:
            return 'terminal', 0
        elif player_sum < dealer_sum:
            return 'terminal', -1
