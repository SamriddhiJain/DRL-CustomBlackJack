from Blackjack import BlackjackEnv

environment_configurations = [{"one_card_dealer": True},
                              {},
                              {"card_values": [2] * 52},
                              {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
                                               4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
                                               2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
                                               5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
achieved_win_rates = []
for env_config in environment_configurations:
    print(env_config)
    tune_results = tune.run(PPOTrainer,
                            stop={"timesteps_total": 100000},
                            config={"env": BlackjackEnv, "env_config": env_config})
    achieved_win_rates.append(tune_results.trials[0].last_result["episode_reward_mean"])

print('PPO achieved the following win rates: ', achieved_win_rates)
# OUTPUT:
# PPO achieved the following win rates:
# [0.9949523275378576, 0.4225774225774226, 0.05228758169934641, 0.22347417840375586]
