# Deep Reinforcement Learning Seminar '20 Coding Challenge

## Implementation details
For the challenge, I tried implementing two algorithms: DQN and PPO. The motivation behind this is to compare the behavior of both the algorithms on the given modified Blackjack game environments. Fir both the algorithms I modified the input state representation to concatenation of arrays representing player and dealer hands.
- DQN: I reused my existing implementation of DQN from [here](https://github.com/SamriddhiJain/RL-practise/tree/master/Cart). The Q-Network is Fully connected network with two hidden layers each containing 128 nodes and relu activation. The other tuned parameters are as below:
```
BUFFER_SIZE = 1e5       # Replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # Learning rates
EPS_DECAY = 0.995       # Epsilon decay
```
All the 4 game configs were trained for 30000 steps, till which the average reward curve gave a plateau.

- PPO: The actor and critic network were both implemented as FC networks, with two hidden layers with 256 nodes each and Tanh as activation. Last layer of actor network is softmax layer, output from which is used to create a categorical distribution from which actions are sampled. The other parameters are as below:
```
GAMMA = 0.99            # discount factor
GAE_LAMBDA = 0.95       # lambda for gae update
LR = 0.005              # learning rate
BETAS = (0.9, 0.999)    # betas used for adam optimizer
UPDATE_TIMES = 6        # number of ppo updates
CLIP = 0.2              # ppo loss clip factor
C1 = 1                  # policy loss
C2 = 0.5                # value loss
C3 = -0.01              # entropy term
BATCH_SIZE = 128        # batch size for update
UPDATE_AFTER = 2000     # number of episodes to collect before ppo update step
```
PPO training required all the games to be run longer than DQN, approximately 100000 episodes. With the above hyperparameters, PPO agent was able to learn well only on the first two games. The agent was not able to learn Game 3, all 2s, with the above set of hyper-parameters. Then I added gradient norm clipping till 10 and reduced the learning rate to 5e-5, which led to better results for game 3 and game 4 as well.

## Results
Winning Rates for the 4 tasks calculated over 1000 episodes:
- DQN (after 30000 episodes): **0.996, 0.352, 1.0, 0.431**
- PPO (after 100000 episodes): **0.984, 0.392, 0.957, 0.423**

### DQN
The **average score variation** (over 100 episodes) with timesteps are shown below for each of the games:

Game 1:
![game1](/DQN/dqn1.png)

Game 2:
![game2](/DQN/dqn2.png)

Game 3:
![game3](/DQN/dqn3.png)

Game 4:
![game4](/DQN/dqn5.png)

- As seen from the curves, the training was most stable for game 1, single card for dealer. The agent learnt quite fast and the average score plateaued very early.
- The learning curve is similar for game 3, with all card values as 2, but the learning is gradual and not as sudden as game 1. At first when the Q network hidden layer sizes were [64, 64], the agent was not able to learn at all and the average reward was all 0. But increasing the hidden layer sizes to [128, 128] led to a gradual learning within 10000 steps.
- Performance on Game 2 and game 4 was slightly better than the rllib PPO results. Both the curves have high variance and flatten at scores around 0.4.  

### PPO
At first, the PPO implementation seemed to performed worse than the DQN one, but after adding gradient clipping and lowering learning rate the results were comparable.
The **average score variation** (over 100 episodes) with timesteps are shown below for each of the games:

Game 1:
![game1](/PPO/ppo5.png)

Game 2:
![game2](/PPO/ppo6.png)

Game 3:
![game3](/PPO/ppo7.png)

Game 4:
![game4](/PPO/ppo8.png)

- PPO agent was able to learn game 1 and game 2 within 50000 steps, teh curves here are extended till 100000 for uniformity with other games.
- Game 3 was the toughest to learn and the agent was not able to make any progress initially. Then after 20000 steps, we see gradual increase in rewards.
- Similar to DQN, the learning curves showed high variance for game 2 and 4. The agent required much more steps to stabilize for game 4.
