## Update: Submission
The results are discussed [here](report.md)

#########################################################

# Deep Reinforcement Learning Seminar '20 Coding Challenge

This repository contains a Blackjack environment as coding challenge for the deep reinforcement learning seminar 2020.

The challenge is setup in 4 difficulty levels:

1. Get familiar with the frameworks and learn an easy version of Blackjack, where the dealer only has one card which is revealed to you. That is, your algorithm should be able to learn to collect cards until the value of the dealers card is exceeded, without going above 21.
2. Let your agent learn Blackjack (Dealer hits on 16, Dealer wins draws).
3. Investigate, whether your agent is also able to learn Blackjack if the carde values are changed.
4. We will provide a new version of the game shortly before the deadline to question your agents capabilities further.

## Setup

Requirements: Conda (https://docs.conda.io/en/latest/)

Clone and enter the Git repository:

```bash
git clone https://gitlab.ethz.ch/disco/teaching/deep-reinforcement-learning-seminar.git
cd deep-reinforcement-learning-seminar
```

Install and activate the conda environment

```bash
conda create -f conda_env.yml
conda activate drl-seminar
```

Run the random agent to see whether everything is working

```bash
python random_agent.py
```

The repository also provides a code snippet to train a PPO agent from RLlib (https://ray.readthedocs.io/en/latest/rllib.html) for 100000 steps on the easy Blackjack task, where the dealer only uses one card.


## Update 12.05.20
The repository now includes an evaluation script (evaluation.py), which specifies 4 environment configurations, on which you should train and evaluate your agent. These 4 evaluation configurations can be described as follows:
1. Easy Blackjack - this is the easy version introduced earlier, where the dealer has only one card.
2. Normal Blackjack - the game as one typically knows it (with Dealer hitting on 16 and Dealer winning draws)
3. All those 2's - A variation of Blackjack where all card values are set to 2. This is a conceptually easy game, as the dealer always gets 18 and the player therefore can achieve a 100% win rate by drawing until he/she reaches 20. Note however that this is a particularly hard problem from a learning perspective, as the feedback is sparse.
4. Random Blackjack - this environment configuration provides a randomly determined assignment of values to the cards. This version cannot be solved with too much game specific knowledge build into the algorithm.
Please train and evaluate your algorithm on these 4 variations for your submission.
