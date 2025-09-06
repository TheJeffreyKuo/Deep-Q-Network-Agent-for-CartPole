import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, config, state_dim, action_dim):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        self.q_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)

        self.memory = ReplayMemory(config.buffer_size)
        self.batch_size = config.batch_size
        self.action_dim = action_dim
        self.step_count = 0

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.step_count / self.epsilon_decay)
        self.step_count += 1
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_v = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_v).max(1)[1].item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).long().unsqueeze(1).to(self.device)
        reward = torch.from_numpy(reward).float().unsqueeze(1).to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().unsqueeze(1).to(self.device)

        q_values = self.q_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_q_net(next_state).max(1, keepdim=True)[0]
            expected_q = reward + self.gamma * next_q * (1 - done)
        loss = nn.functional.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
