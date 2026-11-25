"""The Q-network used by the DDQN agent.

A small fully-connected ReLU MLP.  Deliberately modest: the action space is
discrete and only 101 wide, so 3 layers with 256 hidden units are entirely
sufficient and train quickly on CPU.
"""

import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self,state_dim,action_dim,nhidden=256):#action_dim=0-99 and add mask 
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(state_dim,nhidden,bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(nhidden,nhidden,bias=True)
        self.linear3 = nn.Linear(nhidden,action_dim,bias=True)
    def forward(self, x):#input x=obs_state
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x#Q(s,a) obs_dim=ninput act_dim=noutput
