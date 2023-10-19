import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

import os

class Linear_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="net.pth"):
        model_path = './models'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

class Trainer:
    def __init__(self, model, linreg, gamma):
        self.linreg = linreg
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.linreg)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done): #Can be one or more values per
        state = numpy.array(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = numpy.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # Turns it into a tuple

        #Bellman Equation
        #1: Pred Q with current state
        pred = self.model(state)
        # pred = torch.tensor(pred)
        target = torch.clone(pred)

        # 2: Q_new r + gamma*max(next_pred_Q) --> if not done
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        #3: Loss Function
        self.optim.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optim.step()