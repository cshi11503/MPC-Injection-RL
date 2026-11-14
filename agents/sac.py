"""
Soft Actor-Critic (SAC), Haarnoja et al. 2018 (with automatic temperature tuning).

Pieces (each maps to one idea you will present):
    ReplayBuffer   stores transitions (s, a, r, s', terminated); off-policy => any source
    sample_mixed   builds a minibatch with a fraction rho from the MPC buffer (MPC-Injection)
    Actor          squashed Gaussian policy  a = tanh(mu + sigma * eps)
    Critic         Q(s, a) network; SAC uses two of them (+ slow target copies)
    SAC.update     one step of "soft policy iteration":
                     1) critic  = soft policy EVALUATION   (Bellman regression)
                     2) actor   = soft policy IMPROVEMENT  (move pi toward high Q)
                     3) alpha   = keep policy entropy near the target
                     4) targets = Polyak averaging of critic weights

Actions are always in [-1, 1] (the env scales them by a_max).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Replay buffer
# =========================================================================
class ReplayBuffer:
    """
    Fixed-size circular buffer of transitions, stored in preallocated NumPy arrays.
    Row i holds one transition:
        obs[i]       (obs_dim,)  s_t
        action[i]    (act_dim,)  a_t in [-1, 1]
        reward[i]    ()          r_t
        next_obs[i]  (obs_dim,)  s_{t+1}
        terminated[i] ()         1.0 if s_{t+1} is a true terminal (goal/collision),
                                 0.0 otherwise (including time-limit truncation)
    When full, the oldest rows are overwritten.
    """

    FIELDS = ("obs", "action", "reward", "next_obs", "terminated")

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000, seed: int | None = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0          # next row to write
        self.size = 0         # number of valid rows
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> dict:
        """n transitions drawn uniformly at random (with replacement)."""
        idx = self.rng.integers(0, self.size, size=n)
        return {f: getattr(self, f)[idx] for f in self.FIELDS}

    # --- saving / loading (the MPC dataset uses this same .npz format) ---
    def save_npz(self, path):
        np.savez_compressed(path, **{f: getattr(self, f)[: self.size] for f in self.FIELDS})

    @classmethod
    def from_npz(cls, path, seed: int | None = None):
        data = np.load(path)
        n = data["obs"].shape[0]
        buf = cls(data["obs"].shape[1], data["action"].shape[1], capacity=n, seed=seed)
        for f in cls.FIELDS:
            getattr(buf, f)[:] = data[f]
        buf.size = n
        buf.ptr = 0
        return buf


def sample_mixed(rl_buffer: ReplayBuffer, mpc_buffer: ReplayBuffer | None,
                 batch_size: int, rho: float) -> dict:
    """
    MPC-Injection minibatch: round(rho * B) rows from the MPC buffer,
    the rest from the agent's own buffer. rho = 0 -> plain SAC.
    """
    n_mpc = int(round(rho * batch_size)) if (mpc_buffer is not None and rho > 0) else 0
    rl = rl_buffer.sample(batch_size - n_mpc)
    if n_mpc == 0:
        return rl
    mpc = mpc_buffer.sample(n_mpc)
    return {f: np.concatenate([rl[f], mpc[f]], axis=0) for f in ReplayBuffer.FIELDS}


# =========================================================================
# Networks
# =========================================================================
def mlp(in_dim, out_dim, hidden=(256, 256)):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Squashed Gaussian policy.
        u   = mu(s) + sigma(s) * eps,  eps ~ N(0, I)      (reparameterization trick)
        a   = tanh(u)                                      (bounded to [-1, 1])
        log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2)
    The second term is the change-of-variables correction for the tanh squashing.
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)   # outputs [mu, log_std]
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_log_prob=True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            u = mu                                   # evaluation: use the mean
        else:
            u = mu + std * torch.randn_like(mu)      # rsample -> gradients flow to mu, std
        a = torch.tanh(u)

        if not with_log_prob:
            return a, None
        # Gaussian log-density, summed over action dimensions:
        #   log N(u; mu, std) = -0.5 ((u - mu)/std)^2 - log std - 0.5 log(2 pi)
        # (written out instead of torch.distributions.Normal, whose per-call argument
        #  validation was the single most expensive part of an update, esp. on GPU)
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2.0 * np.pi)).sum(-1)
        # tanh correction, numerically stable form of log(1 - tanh(u)^2):
        #   log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        log_prob -= (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return a, log_prob


class Critic(nn.Module):
    """Q(s, a): concatenate state and action, output one number."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# =========================================================================
# SAC agent
# =========================================================================
class SAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden=(256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 1.0,
        target_entropy: float | None = None,
        auto_alpha: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        # Heuristic from the SAC paper: target entropy = -|A|
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy

        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.q1 = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, hidden).to(self.device)
        # Target critics: slow-moving copies, never trained by gradient descent
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        # Optimize log(alpha) so alpha stays positive
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)

        # fused=True runs the whole Adam step as one GPU kernel (CUDA only)
        fused = self.device.type == "cuda"
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, fused=fused)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                           lr=lr, fused=fused)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, fused=fused)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    # ----------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """obs (obs_dim,) numpy -> action (act_dim,) numpy in [-1, 1]."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(obs_t, deterministic=deterministic, with_log_prob=False)
        return a.squeeze(0).cpu().numpy()

    # ----------------------------------------------------------- learning
    def _to_tensors(self, batch: dict):
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return (t(batch["obs"]), t(batch["action"]), t(batch["reward"]),
                t(batch["next_obs"]), t(batch["terminated"]))

    @torch.no_grad()
    def critic_target(self, reward, next_obs, terminated):
        """
        Soft Bellman target:
            a' ~ pi(.|s')
            y  = r + gamma * (1 - terminated) * [ min(Q1', Q2')(s', a') - alpha * log pi(a'|s') ]
        Note a' comes from the CURRENT policy, not from whoever generated the data:
        this is what makes SAC off-policy (and lets it learn from MPC transitions).
        """
        next_a, next_logp = self.actor(next_obs)
        q_next = torch.min(self.q1_target(next_obs, next_a), self.q2_target(next_obs, next_a))
        soft_value = q_next - self.log_alpha.exp() * next_logp
        return reward + self.gamma * (1.0 - terminated) * soft_value

    def update(self, batch: dict) -> dict:
        obs, action, reward, next_obs, terminated = self._to_tensors(batch)

        # ---- 1) Critic update: soft policy evaluation ----
        y = self.critic_target(reward, next_obs, terminated)
        q1, q2 = self.q1(obs, action), self.q2(obs, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- 2) Actor update: soft policy improvement ----
        # minimize E[ alpha * log pi(a|s) - min Q(s, a) ],  a = tanh(mu + sigma*eps)
        new_a, logp = self.actor(obs)
        q_new = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        # (actor_loss.backward() also wrote gradients into the critics; they are
        #  cleared by critic_opt.zero_grad() before the next critic step.)

        # ---- 3) Temperature update: keep entropy near target_entropy ----
        # entropy too low (logp large) -> alpha increases -> more exploration
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        if self.auto_alpha:
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ---- 4) Target networks: Polyak averaging ----
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_mean": q1.mean().item(),
            "entropy": (-logp.mean()).item(),
        }

    @torch.no_grad()
    def _soft_update(self, net, target):
        # theta_target <- tau * theta + (1 - tau) * theta_target
        for p, p_t in zip(net.parameters(), target.parameters()):
            p_t.mul_(1.0 - self.tau).add_(self.tau * p)

    # ----------------------------------------------------------- checkpoints
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
