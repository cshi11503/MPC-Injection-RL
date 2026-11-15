"""
Unit tests for agents/sac.py.
Run from repo root:   python -m pytest tests -q      (or)   python tests/test_sac.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC, ReplayBuffer, sample_mixed  # noqa: E402

OBS, ACT = 9, 2


def _fill(buf, n, reward=0.0, terminated=0.0):
    for _ in range(n):
        buf.add(np.random.randn(OBS), np.random.uniform(-1, 1, ACT), reward,
                np.random.randn(OBS), terminated)


def test_buffer_circular_overwrite():
    buf = ReplayBuffer(OBS, ACT, capacity=5)
    for i in range(7):
        buf.add(np.full(OBS, i), np.zeros(ACT), float(i), np.zeros(OBS), False)
    assert len(buf) == 5 and buf.ptr == 2
    # rows 0,1 were overwritten by transitions 5,6
    assert sorted(buf.reward.tolist()) == [2, 3, 4, 5, 6]


def test_buffer_npz_roundtrip():
    buf = ReplayBuffer(OBS, ACT, capacity=100)
    _fill(buf, 30)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.npz")
        buf.save_npz(path)
        buf2 = ReplayBuffer.from_npz(path)
    assert len(buf2) == 30
    assert np.allclose(buf2.obs, buf.obs[:30]) and np.allclose(buf2.reward, buf.reward[:30])


def test_sample_mixed_ratio():
    rl = ReplayBuffer(OBS, ACT, capacity=100)
    mpc = ReplayBuffer(OBS, ACT, capacity=100)
    _fill(rl, 50, reward=0.0)
    _fill(mpc, 50, reward=1.0)       # mark MPC rows by reward = 1
    for rho, expected in [(0.0, 0), (0.25, 64), (0.5, 128), (0.75, 192)]:
        batch = sample_mixed(rl, mpc, 256, rho)
        assert batch["obs"].shape == (256, OBS)
        assert int(batch["reward"].sum()) == expected


def test_actions_bounded_and_deterministic_mode():
    agent = SAC(OBS, ACT, seed=0)
    obs = np.random.randn(OBS)
    for _ in range(100):
        a = agent.act(obs)
        assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0)
    assert np.allclose(agent.act(obs, deterministic=True), agent.act(obs, deterministic=True))


def test_log_prob_matches_torch_tanh_distribution():
    """Our tanh-corrected log-prob must equal PyTorch's TransformedDistribution."""
    from torch.distributions import Normal, TransformedDistribution
    from torch.distributions.transforms import TanhTransform

    agent = SAC(OBS, ACT, seed=0)
    obs = torch.randn(64, OBS)
    torch.manual_seed(1)
    a, logp = agent.actor(obs)
    mu, log_std = agent.actor.net(obs).chunk(2, dim=-1)
    dist = TransformedDistribution(Normal(mu, log_std.clamp(-20, 2).exp()), [TanhTransform()])
    a_clamped = a.clamp(-1 + 1e-6, 1 - 1e-6)
    expected = dist.log_prob(a_clamped).sum(-1)
    assert torch.allclose(logp, expected, atol=1e-3)


def test_terminal_transitions_do_not_bootstrap():
    agent = SAC(OBS, ACT, gamma=0.99, seed=0)
    r = torch.tensor([10.0, 10.0])
    s2 = torch.randn(2, OBS)
    y = agent.critic_target(r, s2, terminated=torch.tensor([1.0, 0.0]))
    assert torch.isclose(y[0], torch.tensor(10.0))          # terminal: y = r exactly
    assert not torch.isclose(y[1], torch.tensor(10.0))      # non-terminal: y = r + gamma*V(s')


def test_update_runs_and_targets_move_slowly():
    agent = SAC(OBS, ACT, tau=0.005, seed=0)
    buf = ReplayBuffer(OBS, ACT, capacity=1000)
    _fill(buf, 500)
    p_before = [p.clone() for p in agent.q1_target.parameters()]
    q_before = [p.clone() for p in agent.q1.parameters()]
    metrics = agent.update(buf.sample(256))
    assert all(np.isfinite(v) for v in metrics.values())
    # Polyak: target moved by exactly tau * (new online - old target); here old target == old online
    for pt_old, pt_new, q_new in zip(p_before, agent.q1_target.parameters(), agent.q1.parameters()):
        expected = (1 - 0.005) * pt_old + 0.005 * q_new
        assert torch.allclose(pt_new, expected, atol=1e-6)
    # online critic actually changed
    assert any(not torch.allclose(a, b) for a, b in zip(q_before, agent.q1.parameters()))


def test_alpha_increases_when_entropy_too_low():
    """If the target entropy is very high, alpha should grow (more exploration)."""
    agent = SAC(OBS, ACT, target_entropy=10.0, seed=0)
    buf = ReplayBuffer(OBS, ACT, capacity=1000)
    _fill(buf, 500)
    a0 = agent.alpha
    for _ in range(20):
        agent.update(buf.sample(64))
    assert agent.alpha > a0


def test_save_load_roundtrip():
    agent = SAC(OBS, ACT, seed=0)
    obs = np.random.randn(OBS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        agent2 = SAC(OBS, ACT, seed=123)
        agent2.load(path)
    assert np.allclose(agent.act(obs, True), agent2.act(obs, True))
    assert np.isclose(agent.alpha, agent2.alpha)



def test_cuda_graph_update_learns_like_eager():
    """Graph replay must apply every update to fresh batches: Q -> r on terminal transitions."""
    if not torch.cuda.is_available():
        return
    buf = ReplayBuffer(OBS, ACT, capacity=1000, seed=0)
    _fill(buf, 1000, reward=1.0, terminated=1.0)
    agent = SAC(OBS, ACT, device="cuda", seed=0, cuda_graph=True)
    eager = SAC(OBS, ACT, device="cuda", seed=0)
    for _ in range(300):
        batch = buf.sample(256)
        assert agent.update(batch) is None
        eager.update(batch)
    m = agent.pop_metrics()
    assert all(np.isfinite(v) for v in m.values())
    obs = torch.as_tensor(buf.obs[:256], device="cuda")
    act = torch.as_tensor(buf.action[:256], device="cuda")
    with torch.no_grad():
        q_graph = agent.q1(obs, act).mean().item()
        q_eager = eager.q1(obs, act).mean().item()
    assert abs(q_graph - 1.0) < 0.1 and abs(q_eager - 1.0) < 0.1, (q_graph, q_eager)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
