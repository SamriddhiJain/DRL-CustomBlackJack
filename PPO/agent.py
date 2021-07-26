import numpy as np
import random
from collections import namedtuple, deque

from model import ActorCritic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
GAE_LAMBDA = 0.95
LR = 0.00005              # learning rate
UPDATE_TIMES = 6        # how often to update the network
BETAS = (0.9, 0.999)
CLIP = 0.2
GRAD_CLIP = 10
C1 = 1
C2 = 0.5
C3 = -0.01
BATCH_SIZE = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.done = []
        self.values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.done[:]
        del self.values[:]

    def add(self, state, action, reward, next_state, done, value):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.done.append(done)
        self.values.append(value)

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.network = ActorCritic(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR, betas=BETAS)

        self.memory = Memory()

    def step(self, state, action, reward, next_state, done, value):
        self.memory.add(state, action, reward, next_state, done, value)

    def act(self, state):
        return self.network.act(state)

    def update(self):
        states = torch.stack(self.memory.states).to(device)
        actions = torch.tensor(np.array(self.memory.actions)).to(device)

        oldProb, _, _ = self.network.eval(states, actions)

        rewards = self.gaeUpdated(self.memory.rewards, self.memory.done, self.memory.values)

        for _ in range(UPDATE_TIMES):
            for statesB, actionsB, oldProbB, rewardsB in self.generateBatches(states, actions, oldProb, rewards):

                logprobs, entropy, values = self.network.eval(statesB, actionsB)

                rTheta = torch.exp(logprobs - oldProbB.detach())

                adv = rewardsB - values.squeeze(-1)
                # normalize
                adv = torch.tensor(adv).to(device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

                policyLoss = -torch.min(rTheta*adv, torch.clamp(rTheta, 1-CLIP, 1+CLIP)*adv)

                valLoss = F.mse_loss(values.squeeze(-1), rewardsB)

                loss = C1*policyLoss + C2*valLoss + C3*entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), GRAD_CLIP)
                self.optimizer.step()

        self.memory.clear()
        return policyLoss.mean().item(), valLoss.item()

    def generateBatches(self, states, actions, oldProb, rewards):
        size = states.size(0)

        for _ in range(size // BATCH_SIZE):
            rand_ids = np.random.randint(0, size, BATCH_SIZE)
            yield states[rand_ids, :], actions[rand_ids], oldProb[rand_ids], rewards[rand_ids]

    def gaeUpdated(self, rewardsOriginal, dones, values):
        run_add = 0
        returns = []
        returns.insert(0, rewardsOriginal[-1])

        for t in reversed(range(len(values)-1)):
            if dones[t]:
                run_add = rewardsOriginal[t] - values[t]
            else:
                sigma = rewardsOriginal[t] + GAMMA*values[t+1] - values[t]
                run_add = sigma + run_add*GAMMA*GAE_LAMBDA
            returns.insert(0, run_add + values[t])

        # returns = torch.tensor(returns).to(device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return torch.tensor(returns).to(device)
