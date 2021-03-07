import numpy as np


class Env:
    def __init__(self):
        self.action_space = ('hit', 'stick')

    def reset(self):
        """
        Resets the environment, initializing the state and reward

        :return: The initial state
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
        :param action: action -- hit or stick
        :return: sample of next state, reward, and whether the episode terminated
        """

        if action == 'hit':
            card_val = self.draw(player_turn=True)
            player_sum = self.state[1] + card_val
            self.state = (self.state[0], player_sum)
            if player_sum > 21 or player_sum < 1:
                return self.state, -1, True
            else:
                return self.state, 0, False

        elif action == 'stick':
            dealer_sum = self.state[0]
            while 1 <= dealer_sum < 17:
                card_val = self.draw(player_turn=False)
                dealer_sum += card_val

            self.state = (dealer_sum, self.state[1])

            if dealer_sum > 21 or dealer_sum < 1:
                return self.state, 1, True

            player_sum = self.state[1]
            if player_sum > dealer_sum:
                return self.state, 1, True
            elif player_sum == dealer_sum:
                return self.state, 0, True
            elif player_sum < dealer_sum:
                return self.state, -1, True
