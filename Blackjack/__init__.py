from gym.envs.registration import register
from .blackjack import BlackjackEnv

register(
    id='Blackjack-v1',
    entry_point='Blackjack:BlackjackEnv'
)
