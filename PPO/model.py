import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, n_actions, seed):
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1))

        self.value_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1))

        self.policy_net.apply(self.init_weights)
        self.value_net.apply(self.init_weights)

    def forward(self, state):
        pass

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def act(self, state):
        logits = self.policy_net(state.float())
        dist = Categorical(logits)
        action = dist.sample().cpu().numpy()
        value = self.value_net(state.float())

        return action, value

    def eval(self, state, action):
        logits = self.policy_net(state.float())
        dist = Categorical(logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = self.value_net(state.float())

        return log_prob, entropy, value
