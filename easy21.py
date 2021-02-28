import numpy as np


class Env:
    def __init__(self):
        self.action_space = ('hit', 'stick')

    def reset(self):
        """
        :return: Resets the environment, initializing the state and reward
        """
        player_start = np.random.randint(1, 11)
        self.state = (np.random.randint(1, 11), player_start)
        self.reward = 0
        self.cur_card = 'black ' + str(player_start)
        return self.state

    def draw(self, player_turn):
        """
        :return: random card value(with replacement)
        """
        color = np.random.choice(['black', 'red'], p=(2/3, 1/3))
        if color == 'black':
            card_val = np.random.randint(1, 11)
        elif color == 'red':
            card_val = -np.random.randint(1, 11)

        if player_turn:
            self.cur_card = color + ' ' + str(card_val)

        return card_val

    def step(self, action):
        """
        :param s: state -- dealer's first card 1-10 and player's sum 1-21
        :param a: action -- hit or stick
        :return: sample of next state (which may be terminal if game is finished) and reward
        """
        if self.state == 'terminal':
            print('The game has already ended.')
            return 'terminal', 0

        if action == 'hit':
            card_val = self.draw(player_turn=True)
            player_sum = self.state[1] + card_val
            self.state = (self.state[0], player_sum)
            if player_sum > 21 or player_sum < 1:
                return self.state, -1, 'terminal'
            else:
                return self.state, 0, False

        elif action == 'stick':
            dealer_sum = self.state[0]
            while 1 <= dealer_sum < 17:
                card_val = self.draw(player_turn=False)
                dealer_sum += card_val

            self.state = (dealer_sum, self.state[1])

            if dealer_sum > 21 or dealer_sum < 1:
                return self.state, 1, 'terminal'

            player_sum = self.state[1]
            if player_sum > dealer_sum:
                return self.state, 1, 'terminal'
            elif player_sum == dealer_sum:
                return self.state, 0, 'terminal'
            elif player_sum < dealer_sum:
                return self.state, -1, 'terminal'
