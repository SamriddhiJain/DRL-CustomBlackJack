import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


def sum_hand(card_values):
    sum = np.sum(card_values)
    if 1 in card_values and sum + 10 <= 21:
        return sum + 10  # use ace as 11
    return sum


class BlackjackEnv(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self, one_card_dealer=False, card_values=None):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((spaces.MultiBinary(52), spaces.Discrete(52)))
        self._card_values = np.asarray(card_values)
        if card_values is None:
            self._card_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4)
        assert len(self._card_values) == 52
        self._one_card_dealer = one_card_dealer
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._deck = np.arange(52)
        self.np_random.shuffle(self._deck)
        self._player_cards, self._deck = self._deck[:2], self._deck[2:]
        self._dealer_cards, self._deck = self._deck[:2], self._deck[2:]
        return self._get_obs()

    def _get_obs(self):
        obs = np.asarray([False] * 52)
        obs[self._player_cards] = True
        return obs, self._dealer_cards[0]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        player_sum = sum_hand(self._card_values[self._player_cards])
        info = {'player hand sum before': player_sum}
        if action:  # hit
            self._player_cards = np.append(self._player_cards, self._deck[0])
            self._deck = self._deck[1:]
            player_sum = sum_hand(self._card_values[self._player_cards])
        else:
            if self._one_card_dealer:
                reward = float(player_sum > self._card_values[self._dealer_cards[0]])
            else:
                while sum_hand(self._card_values[self._dealer_cards]) < 17:
                    self._dealer_cards = np.append(self._dealer_cards, self._deck[0])
                    self._deck = self._deck[1:]
                dealer_sum = sum_hand(self._card_values[self._dealer_cards])
                info.update({'dealer hand sum': dealer_sum})
                if dealer_sum > 21:
                    reward = 1
                else:
                    reward = float(player_sum > dealer_sum)
        info.update({'action played': 'Hit' if action else 'Stand', 'player hand sum now': player_sum})
        done = player_sum > 21 or not action
        return self._get_obs(), reward, done, info

    def render(self):
        pass
