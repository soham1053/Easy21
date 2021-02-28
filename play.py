"""
For human play
"""
from easy21 import Env
env = Env()


def get_input_action(state, player_cards):
    print(f'Current State: Dealer\'s First Card = {state[0]}, Your Total = {state[1]}')
    action = input('What\'s your move? (hit or stick)')
    while action not in env.action_space:
        action = input('Either say hit or stick please!')
    print('\n')
    return action

while True:
    state = env.reset()
    print('Let\'s start!\n')
    player_cards = [env.cur_card]
    print(f'Your deck: {player_cards}')
    while True:
        action = get_input_action(state, player_cards)
        nextState, reward, done = env.step(action)
        if action == 'hit':
            player_cards.append(env.cur_card)
            print(f'Your deck: {player_cards}')
        if done:
            break
        state = nextState

    print(f'Dealer\'s Total - {nextState[0]}, Your Total - {nextState[1]}\n')

    if reward == -1:
        print("You lost")
    elif reward == 0:
        print("Draw")
    else:
        print("You won")

    if input('Play again? [yes, y]') not in ('yes', 'y'):
        break
