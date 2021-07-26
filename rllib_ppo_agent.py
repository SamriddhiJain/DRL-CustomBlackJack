from Blackjack import BlackjackEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


tune.run(PPOTrainer, stop={"timesteps_total": 100000}, config={"env": BlackjackEnv,
                                                               "env_config": {"one_card_dealer": True}})
