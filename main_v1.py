import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ==========================================================
# 1. ENVIRONMENT
# ==========================================================
class WheelTruingEnv:
    """
    Observation: (5, N)
        0: lateral runout (mean-centred, mm)
        1: radial runout  (mean-centred, mm)
        2: tension relative to the side's target window  ((T - mid) / half_range)
        3: flange side (-1 left / +1 right)
        4: fraction of the episode elapsed
    Action: [0, N-1] tighten spoke i 1/4 turn; [N, 2N-1] loosen spoke i 1/4 turn
    """
    N_CHANNELS = 5
    T_SAFE_MIN, T_SAFE_MAX = 300.0, 1600.0   # N, outside this = unsafe -> terminate

    def __init__(self, specs, goals, max_steps=400, gamma=0.99, seed=None):
        self.n = int(specs[9])
        assert self.n % 2 == 0, "Even/odd left/right alternation needs an even spoke count"
        self.action_dim = 2 * self.n
        self.goals = goals.astype(np.float32)
        self.max_steps = max_steps
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.difficulty = 1.0   # curriculum knob in [0, 1]

        even = (np.arange(self.n) % 2 == 0)
        self.side = np.where(even, -1.0, 1.0).astype(np.float32)
        self.t_lo = np.where(even, goals[2], goals[4]).astype(np.float32)
        self.t_hi = np.where(even, goals[3], goals[5]).astype(np.float32)
        self.t_mid = 0.5 * (self.t_lo + self.t_hi)
        self.t_half = 0.5 * (self.t_hi - self.t_lo)

        self.offsets = np.arange(-2, 3)
        self.influence = np.exp(-np.abs(self.offsets)).astype(np.float32)
        self.max_scramble = 150

    # ---------- helpers ----------
    def _metrics(self):
        p_lat, p_rad = np.ptp(self.lat), np.ptp(self.rad)
        viol = np.maximum(0, self.t_lo - self.T) + np.maximum(0, self.T - self.t_hi)
        return p_lat, p_rad, viol
    
    def _apply(self, spoke, turn):
        idx = (spoke + self.offsets) % self.n
        self.T[spoke] += turn * self.k_T
        self.lat[idx] += turn * self.side[spoke] * self.k_lat * self.influence
        self.rad[idx] -= turn * self.k_rad * self.influence

    def _is_success(self):
        p_lat, p_rad, viol = self._metrics()
        return (p_lat <= self.goals[0]) and (p_rad <= self.goals[1]) and np.all(viol == 0)

    def reset(self):
        self.t = 0
        self.k_T = 80.0 * self.rng.uniform(0.85, 1.15)
        self.k_lat = 0.15 * self.rng.uniform(0.7, 1.3)
        self.k_rad = 0.10 * self.rng.uniform(0.7, 1.3)
        n_scr = max(30, int(round(self.difficulty * self.max_scramble)))
        for _ in range(20):                       # retry if already solved
            self.lat = np.zeros(self.n, np.float32)
            self.rad = np.zeros(self.n, np.float32)
            self.T = self.t_mid.copy()
            for _ in range(n_scr):
                self._apply(int(self.rng.integers(self.n)),
                            float(self.rng.choice([-0.25, 0.25])))
            if not self._is_success():
                break
        self.phi = self._potential()
        return self._obs()

    def _potential(self):
        p_lat, p_rad, viol = self._metrics()
        gl, gr = self.goals[0], self.goals[1]      # measure runout in tolerance units
        return -(p_lat / gl + 0.5 * self.lat.std() / gl
                 + p_rad / gr + 0.5 * self.rad.std() / gr
                 + viol.mean() / 50.0)

    def _obs(self):
        lat = (self.lat - self.lat.mean()) / self.goals[0]
        rad = (self.rad - self.rad.mean()) / self.goals[1]
        ten = (self.T - self.t_mid) / self.t_half
        prog = np.full(self.n, self.t / self.max_steps, dtype=np.float32)
        return np.stack([lat, rad, ten, self.side, prog]).astype(np.float32)
    
    def step(self, action: int):
        self.t += 1
        spoke = action % self.n
        turn = 0.25 if action < self.n else -0.25
        side = self.side[spoke]

        self.T[spoke] += turn * self.k_T
        idx = (spoke + self.offsets) % self.n
        self.lat[idx] += turn * side * self.k_lat * self.influence
        self.rad[idx] -= turn * self.k_rad * self.influence

        p_lat, p_rad, viol = self._metrics()
        success = (p_lat <= self.goals[0]) and (p_rad <= self.goals[1]) and np.all(viol == 0)
        unsafe = bool(self.T.min() < self.T_SAFE_MIN or self.T.max() > self.T_SAFE_MAX)

        new_phi = self._potential()
        reward = self.gamma * new_phi - self.phi - 0.01      # shaping - step cost
        self.phi = new_phi
        if success:
            reward += 5.0
        if unsafe:
            reward -= 5.0

        terminated = bool(success or unsafe)
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"p2p_lat": float(p_lat), "p2p_rad": float(p_rad),
                "success": bool(success), "unsafe": unsafe}
        return self._obs(), float(reward), terminated, truncated, info


class SyncVectorEnv:
    """Minimal synchronous vector env with auto-reset."""
    def __init__(self, envs):
        self.envs = envs

    def set_difficulty(self, d):
        for e in self.envs:
            e.difficulty = d

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rews, terms, truncs, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(int(a))
            if te or tr:
                info["final_obs"] = o
                info["episode"] = {"success": info["success"], "length": e.t,
                                   "p2p_lat": info["p2p_lat"], "p2p_rad": info["p2p_rad"]}
                o = e.reset()
            obs.append(o); rews.append(r); terms.append(te); truncs.append(tr); infos.append(info)
        return (np.stack(obs), np.array(rews, np.float32),
                np.array(terms), np.array(truncs), infos)


# ==========================================================
# 2. MODEL
# ==========================================================
class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, padding_mode="circular")
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class WheelTruingActorCritic(nn.Module):
    """
    Fully-convolutional on the circular spoke axis:
      * actor emits per-spoke logits for [tighten, loosen] -> rotation-equivariant
      * global context (mean/max pooling) lets each spoke 'know' where the wheel's
        global peaks are (needed because p2p runout is a global quantity)
      * parameter count is independent of the number of spokes
    """
    def __init__(self, in_ch=5, hidden=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, d) for d in dilations])
        self.ctx = nn.Linear(2 * hidden, hidden)
        self.actor_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, 2, 1))
        self.critic_head = nn.Sequential(
            nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Linear(128, 1))

        self.apply(self._init)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)   # near-uniform start policy
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    def forward(self, x):                                   # x: (B, C, N)
        h = self.blocks(self.stem(x))
        g = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)     # (B, 2H)
        h = h + self.ctx(g).unsqueeze(-1)                   # inject global context
        logits = self.actor_head(h).flatten(1)              # (B, 2N): [tighten | loosen]
        value = self.critic_head(g).squeeze(-1)             # (B,)
        return logits, value


# ==========================================================
# 3. TRAINING
# ==========================================================
@dataclass
class Config:
    n_envs: int = 16
    n_steps: int = 32
    updates: int = 4000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ent_start: float = 0.02
    ent_end: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_every: int = 100
    eval_episodes: int = 20
    curriculum_success: float = 0.8
    curriculum_step: float = 0.15
    seed: int = 0
    save_path: str = "best_truing_agent.pt"


@torch.no_grad()
def evaluate(model, env, device, n_episodes=20):
    model.eval()
    env.difficulty = 1.0
    succ, lens, lats, rads = [], [], [], []
    for _ in range(n_episodes):
        obs, done = env.reset(), False
        while not done:
            logits, _ = model(torch.as_tensor(obs, device=device).unsqueeze(0))
            obs, _, te, tr, info = env.step(int(logits.argmax(-1)))
            done = te or tr
        succ.append(info["success"]); lens.append(env.t)
        lats.append(info["p2p_lat"]); rads.append(info["p2p_rad"])
    model.train()
    return np.mean(succ), np.mean(lens), np.mean(lats), np.mean(rads)


def train_truing_agent(specs, goals, cfg: Config = Config()):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    venv = SyncVectorEnv([WheelTruingEnv(specs, goals, gamma=cfg.gamma, seed=cfg.seed + i)
                          for i in range(cfg.n_envs)])
    eval_env = WheelTruingEnv(specs, goals, gamma=cfg.gamma, seed=10_000 + cfg.seed)

    model = WheelTruingActorCritic(in_ch=WheelTruingEnv.N_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T, E = cfg.n_steps, cfg.n_envs
    obs_shape = (WheelTruingEnv.N_CHANNELS, venv.envs[0].n)
    difficulty = 0.2
    venv.set_difficulty(difficulty)
    obs = venv.reset()
    recent = deque(maxlen=100)
    best_score = -1.0

    for upd in range(1, cfg.updates + 1):
        frac = 1.0 - (upd - 1) / cfg.updates
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * max(frac, 0.1)
        ent_coef = cfg.ent_end + (cfg.ent_start - cfg.ent_end) * frac

        obs_buf = torch.zeros((T, E, *obs_shape), device=device)
        act_buf = torch.zeros((T, E), dtype=torch.long, device=device)
        rew_buf = torch.zeros((T, E), device=device)
        done_buf = torch.zeros((T, E), device=device)
        val_buf = torch.zeros((T, E), device=device)

        # ---------- rollout ----------
        for t in range(T):
            obs_t = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                logits, value = model(obs_t)
                action = Categorical(logits=logits).sample()

            next_obs, rew, term, trunc, infos = venv.step(action.cpu().numpy())
            rew = torch.as_tensor(rew, device=device)

            # Bootstrap value on time-limit truncation (not a real terminal state)
            if trunc.any():
                idx = np.flatnonzero(trunc)
                final = torch.as_tensor(np.stack([infos[i]["final_obs"] for i in idx]), device=device)
                with torch.no_grad():
                    _, v_final = model(final)
                rew[torch.as_tensor(idx, device=device)] += cfg.gamma * v_final

            for info in infos:
                if "episode" in info:
                    recent.append(info["episode"])

            obs_buf[t], act_buf[t], rew_buf[t], val_buf[t] = obs_t, action, rew, value
            done_buf[t] = torch.as_tensor(term | trunc, dtype=torch.float32, device=device)
            obs = next_obs

        # ---------- GAE ----------
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(obs, device=device))
        adv = torch.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_v * nonterm - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * last_gae
            adv[t] = last_gae
        returns = adv + val_buf

        # ---------- update ----------
        b_obs = obs_buf.reshape(T * E, *obs_shape)
        b_act, b_ret, b_adv = act_buf.reshape(-1), returns.reshape(-1), adv.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        logits, values = model(b_obs)
        dist = Categorical(logits=logits)
        policy_loss = -(dist.log_prob(b_act) * b_adv).mean()
        value_loss = F.smooth_l1_loss(values, b_ret)
        entropy = dist.entropy().mean()
        loss = policy_loss + cfg.vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        # ---------- curriculum ----------
        if len(recent) >= 50:
            sr = np.mean([e["success"] for e in recent])
            if sr >= cfg.curriculum_success and difficulty < 1.0:
                difficulty = min(1.0, difficulty + cfg.curriculum_step)
                venv.set_difficulty(difficulty)
                recent.clear()
                print(f"  -> curriculum difficulty raised to {difficulty:.2f}")

        # ---------- logging / eval ----------
        if upd % cfg.eval_every == 0:
            train_sr = np.mean([e["success"] for e in recent]) if recent else 0.0
            sr, ln, lat, rad = evaluate(model, eval_env, device, cfg.eval_episodes)
            print(f"[Upd {upd:5d}] diff={difficulty:.2f} | train SR={train_sr:.2f} | "
                  f"EVAL SR={sr:.2f} len={ln:.0f} lat={lat:.3f}mm rad={rad:.3f}mm | "
                  f"ent={entropy.item():.2f} vloss={value_loss.item():.3f}")
            score = sr - 1e-3 * (lat + rad)      # success first, tie-break on final runout
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), cfg.save_path)

    model.load_state_dict(torch.load(cfg.save_path, map_location=device))
    return model

def advise(model, lat, rad, ten, side, step, max_steps,
           goals, t_mid, t_half, top_k=3):
    lat = np.asarray(lat) - np.mean(lat)
    rad = np.asarray(rad) - np.mean(rad)
    obs = np.stack([lat / goals[0],
                    rad / goals[1],
                    (np.asarray(ten) - t_mid) / t_half,
                    side,
                    np.full(len(lat), step / max_steps)]).astype(np.float32)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(obs)[None])
    logits = logits[0].clone()

    # mask moves that would push tension outside the safe window (T_MAX/T_MIN, k_T from calibration)
    # logits[unsafe_action_ids] = -1e9

    for a in torch.topk(logits, top_k).indices.tolist():
        spoke, direction = divmod(a, 2)   # adjust to your action encoding
        print(f"Spoke {spoke} ({'R' if side[spoke] > 0 else 'L'}): "
              f"{'TIGHTEN' if direction == 0 else 'LOOSEN'} 1/4 turn")


if __name__ == "__main__":
    # specs = np.array([602.0, 142.0, 32.5, 35.0, 19.0, 48.0, 58.0, 58.0,
                      # 2.5, 32.0, 3.0, 2.0, 0.0])
    # goals = np.array([0.2, 0.3, 900.0, 1200.0, 700.0, 1000.0])
    
    # 32-spoke rear disc wheel example specifications
    specs = np.array([
        602.0,  # ERD
        142.0,  # Axle width
        32.5,   # Left Locknut->flange
        35.0,   # Left Flange->center
        19.0,   # Right Flange->center
        48.0,   # Right Locknut->flange
        58.0,   # Left flange diam
        58.0,   # Right flange diam
        2.5,    # Spoke hole diam
        32.0,   # Number of spokes
        3.0,    # Lacing pattern (3-cross)
        2.0,    # Spoke diameter
        0.0     # 0.0=round, 1.0=flattened
    ])

    # Goal tolerances
    goals = np.array([
        0.1,    # Peak-to-peak Lateral Runout <= 0.1 mm
        0.1,    # Peak-to-peak Radial Runout <= 0.1 mm
        900.0,  # Left Spoke Min Tension (N)
        1200.0, # Left Spoke Max Tension (N)
        700.0,  # Right Spoke Min Tension (N)
        1000.0  # Right Spoke Max Tension (N)
    ])
    
    model = train_truing_agent(specs, goals, Config())