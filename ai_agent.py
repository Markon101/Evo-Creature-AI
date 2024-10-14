import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, activation_levels, hidden_size=24):
        self.state_size = state_size
        self.action_size = action_size
        self.activation_levels = activation_levels
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.model = NeuralNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.step_counter = 0  # To track steps for saving

        # Load existing model if available
        if os.path.exists("dqn_model.pth"):
            try:
                self.load("dqn_model.pth")
                print("Loaded existing model.")
            except Exception as e:
                print("Failed to load existing model:", e)
                print("Starting with a new model.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.model(next_state_tensor.unsqueeze(0)), dim=1)[0].item()
            target_f = self.model(state_tensor)
            target_f = target_f.clone().detach()
            target_f[action] = target
            states.append(state_tensor)
            targets.append(target_f)

        states = torch.stack(states)
        targets = torch.stack(targets)

        outputs = self.model(states)
        loss = self.criterion(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_counter += 1

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device, weights_only=True))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)