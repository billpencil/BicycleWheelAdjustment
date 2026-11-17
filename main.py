import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. WHEEL ENVIRONMENT (SIMULATOR / INTERFACE)
# ==========================================
class WheelTruingEnv:
    def __init__(self, specs: np.ndarray, goals: np.ndarray):
        """
        specs: shape (13,) - Geometric wheel specs
        goals: shape (6,)  - [max_p2p_lat, max_p2p_rad, min_T_L, max_T_L, min_T_R, max_T_R]
        """
        self.specs = specs.astype(np.float32)
        self.goals = goals.astype(np.float32)
        
        self.num_spokes = int(specs[9])
        # Action space: 2 * N (Spoke i tighten 1/4 turn, or loosen 1/4 turn)
        # Action index: [0, N-1] -> tighten 1/4 turn; [N, 2N-1] -> loosen 1/4 turn
        self.action_dim = 2 * self.num_spokes
        
        # State: 3 channels x N spokes
        # Channel 0: Lateral Runout (mm)
        # Channel 1: Radial Runout (mm)
        # Channel 2: Spoke Tension (N or kgf)
        self.state = np.zeros((3, self.num_spokes), dtype=np.float32)
        
        self.prev_p2p_lateral = 0.0
        self.prev_p2p_radial = 0.0
        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        
        # Initialize an untrued wheel state (random initial imperfections)
        # In practice, this will be read from physical dial indicators/tensiometers.
        init_lateral = np.random.normal(0, 1.2, size=self.num_spokes) # ~1.2mm runout
        init_radial = np.random.normal(0, 0.8, size=self.num_spokes)  # ~0.8mm runout
        init_tensions = np.random.uniform(700, 1100, size=self.num_spokes) # tension in N

        self.state[0] = init_lateral
        self.state[1] = init_radial
        self.state[2] = init_tensions

        self.prev_p2p_lateral = np.ptp(self.state[0])
        self.prev_p2p_radial = np.ptp(self.state[1])

        return self._get_obs()

    def _get_obs(self):
        return self.state.copy()

    def step(self, action: int):
        self.current_step += 1
        
        spoke_idx = action % self.num_spokes
        turn = 0.25 if action < self.num_spokes else -0.25 # +0.25 = tight, -0.25 = loosen
        
        # --- PHYSICAL RESPONSE (Simulated or Real Hardware Interface) ---
        # Spoke side: Even = Left flange, Odd = Right flange
        side = -1.0 if (spoke_idx % 2 == 0) else 1.0 
        
        # 1. Tension response (~80 N per 1/4 turn on a standard 2.0mm spoke)
        delta_tension = turn * 80.0
        self.state[2, spoke_idx] += delta_tension
        
        # 2. Local Rim deflection response (Influence function across neighbors)
        for offset in [-2, -1, 0, 1, 2]:
            idx = (spoke_idx + offset) % self.num_spokes
            influence = np.exp(-abs(offset))
            # Lateral deflection depends on which flange the spoke belongs to
            self.state[0, idx] += turn * side * 0.15 * influence
            # Tightening always pulls radial runout inward (-), loosening pushes outward (+)
            self.state[1, idx] -= turn * 0.10 * influence

        # --- CALCULATE METRICS ---
        curr_p2p_lateral = np.ptp(self.state[0])
        curr_p2p_radial = np.ptp(self.state[1])

        # --- REWARD FUNCTION (Treat / Punish) ---
        reward = 0.0
        
        # 1. Lateral runout evaluation
        delta_lat = self.prev_p2p_lateral - curr_p2p_lateral
        if delta_lat > 0:
            reward += 10.0 * delta_lat   # Treat: lateral runout reduced
        else:
            reward += 15.0 * delta_lat   # Punish: lateral runout increased
            
        # 2. Radial runout evaluation
        delta_rad = self.prev_p2p_radial - curr_p2p_radial
        if delta_rad > 0:
            reward += 10.0 * delta_rad   # Treat: radial runout reduced
        else:
            reward += 15.0 * delta_rad   # Punish: radial runout increased

        # 3. Tension boundaries evaluation
        left_tensions = self.state[2, 0::2]
        right_tensions = self.state[2, 1::2]
        
        t_penalty = 0.0
        # Left side check
        t_penalty += np.sum(np.maximum(0, self.goals[2] - left_tensions))
        t_penalty += np.sum(np.maximum(0, left_tensions - self.goals[3]))
        # Right side check
        t_penalty += np.sum(np.maximum(0, self.goals[4] - right_tensions))
        t_penalty += np.sum(np.maximum(0, right_tensions - self.goals[5]))
        reward -= 0.01 * t_penalty

        # --- CHECK TERMINATION (Goal Criterion) ---
        lat_met = curr_p2p_lateral <= self.goals[0]
        rad_met = curr_p2p_radial <= self.goals[1]
        tension_met = (
            np.all(left_tensions >= self.goals[2]) and np.all(left_tensions <= self.goals[3]) and
            np.all(right_tensions >= self.goals[4]) and np.all(right_tensions <= self.goals[5])
        )
        
        success = lat_met and rad_met and tension_met
        done = success or (self.current_step >= self.max_steps)

        if success:
            reward += 100.0  # Goal reached bonus

        # Update cache
        self.prev_p2p_lateral = curr_p2p_lateral
        self.prev_p2p_radial = curr_p2p_radial

        info = {
            "p2p_lat": curr_p2p_lateral,
            "p2p_rad": curr_p2p_radial,
            "success": success
        }

        return self._get_obs(), reward, done, info


# ==========================================
# 2. ACTOR-CRITIC REINFORCEMENT LEARNING MODEL
# ==========================================
class WheelTruingActorCritic(nn.Module):
    def __init__(self, num_spokes: int):
        super().__init__()
        self.num_spokes = num_spokes
        self.action_dim = 2 * num_spokes
        
        # 1D Circular Convolution to leverage circumferential rim structure
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten()
        )
        
        feat_size = 64 * num_spokes
        
        # Policy Head (Actor): selects spoke & direction
        self.actor = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # Value Head (Critic): estimates expected cumulative reward
        self.critic = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, 3, num_spokes]
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


# ==========================================
# 3. TRAINING PROCEDURE
# ==========================================
def train_truing_agent(env: WheelTruingEnv, episodes: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WheelTruingActorCritic(num_spokes=env.num_spokes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(episodes):
        state = env.reset()
        done = False
        
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device) # (1, 3, N)
            logits, value = model(state_tensor)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            
            if info["success"]:
                print(f"[Episode {ep}] SUCCESS! Lat: {info['p2p_lat']:.3f}mm, Rad: {info['p2p_rad']:.3f}mm")
                break

        # Compute discounted returns (A2C update)
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = (returns - values).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | Steps: {env.current_step} | Last P2P Lat: {info['p2p_lat']:.3f}mm, Rad: {info['p2p_rad']:.3f}mm")

    return model
    
if __name__ == "__main__":
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
        0.2,    # Peak-to-peak Lateral Runout <= 0.2 mm
        0.3,    # Peak-to-peak Radial Runout <= 0.3 mm
        900.0,  # Left Spoke Min Tension (N)
        1200.0, # Left Spoke Max Tension (N)
        700.0,  # Right Spoke Min Tension (N)
        1000.0  # Right Spoke Max Tension (N)
    ])

    env = WheelTruingEnv(specs, goals)
    trained_model = train_truing_agent(env, episodes=1000)