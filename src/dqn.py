import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """State (11 numbers) -> one Q-value per action (6 numbers)."""
    def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
            self.value = nn.Linear(hidden, 1)          # "how good is this situation?"
            self.adv = nn.Linear(hidden, n_actions)    # "how much better is each action?"
        else:
            self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_actions))

    def forward(self, x):
        if self.dueling:
            z = self.body(x)
            adv = self.adv(z)
            return self.value(z) + adv - adv.mean(dim=1, keepdim=True)
        return self.net(x)


class ReplayBuffer:
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


class PrioritizedBuffer(ReplayBuffer):
    """Re-visits surprising experiences more often."""
    def __init__(self, size, obs_dim, alpha=0.6):
        super().__init__(size, obs_dim)
        self.p = np.zeros(size, np.float64)
        self.alpha, self.max_p = alpha, 1.0

    def add(self, s, a, r, s2, d):
        self.p[self.i] = self.max_p                  # new data gets top priority
        super().add(s, a, r, s2, d)

    def sample(self, batch, beta=0.4):
        pr = self.p[:self.n] ** self.alpha
        probs = pr / pr.sum()
        idx = np.random.choice(self.n, batch, p=probs)
        w = (self.n * probs[idx]) ** (-beta)
        w = (w / w.max()).astype(np.float32)         # importance weights fix the bias
        tensors = (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                   torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                   torch.from_numpy(self.d[idx]))
        return tensors, idx, torch.from_numpy(w)

    def update(self, idx, td_abs):
        self.p[idx] = td_abs + 1e-3
        self.max_p = max(self.max_p, float(self.p[idx].max()))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, double=False, dueling=False, per=False,
                 gamma=0.99, lr=1e-3, buffer_size=50_000, batch=64, target_every=500):
        self.q = QNet(obs_dim, n_actions, dueling=dueling)
        self.q_target = QNet(obs_dim, n_actions, dueling=dueling)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = PrioritizedBuffer(buffer_size, obs_dim) if per else ReplayBuffer(buffer_size, obs_dim)
        self.double, self.per, self.beta = double, per, 0.4
        self.gamma, self.batch, self.target_every = gamma, batch, target_every
        self.n_actions, self.steps = n_actions, 0

    def act(self, obs, eps):
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            return int(self.q(torch.from_numpy(obs).unsqueeze(0)).argmax())

    def learn(self):
        if self.buf.n < self.batch:
            return None
        if self.per:
            (s, a, r, s2, d), idx, w = self.buf.sample(self.batch, self.beta)
        else:
            s, a, r, s2, d = self.buf.sample(self.batch)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                best_a = self.q(s2).argmax(1, keepdim=True)                  # online net PICKS
                best_next = self.q_target(s2).gather(1, best_a).squeeze(1)   # target net SCORES
            else:
                best_next = self.q_target(s2).max(1).values
            target = r + self.gamma * best_next * (1 - d)
        loss_each = nn.functional.smooth_l1_loss(q_sa, target, reduction="none")
        if self.per:
            loss = (w * loss_each).mean()
            self.buf.update(idx, (target - q_sa).detach().abs().numpy())
        else:
            loss = loss_each.mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def greedy_policy(self):
        return lambda obs: self.act(obs, eps=0.0)
        