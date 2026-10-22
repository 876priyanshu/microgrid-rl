import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Maps a state (11 numbers) to one Q-value per action (6 numbers)."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size memory of (state, action, reward, next_state, done)."""
    def __init__(self, size, obs_dim):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.d[self.i] = s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, gamma=0.99, lr=1e-3, buffer_size=50_000,
                 batch=64, target_every=500):
        self.q = QNet(obs_dim, n_actions)
        self.q_target = QNet(obs_dim, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer_size, obs_dim)
        self.gamma, self.batch, self.target_every = gamma, batch, target_every
        self.n_actions, self.steps = n_actions, 0

    def act(self, obs, eps):
        if np.random.rand() < eps:                       # explore: random move
            return np.random.randint(self.n_actions)
        with torch.no_grad():                            # exploit: best known move
            return int(self.q(torch.from_numpy(obs).unsqueeze(0)).argmax())

    def learn(self):
        if self.buf.n < self.batch:
            return None
        s, a, r, s2, d = self.buf.sample(self.batch)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)   # Q(s, a) we predicted
        with torch.no_grad():
            best_next = self.q_target(s2).max(1).values         # best Q at next state
            target = r + self.gamma * best_next * (1 - d)       # Bellman target
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def greedy_policy(self):
        return lambda obs: self.act(obs, eps=0.0)