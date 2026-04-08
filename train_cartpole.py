import gymnasium as gym
import numpy as np
import os
import json
import math
from datetime import datetime
from stable_baselines3 import PPO, SAC
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import shutil
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def select_log_directory():
    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('cartpole_shift_logs_')]
    dirs.sort(reverse=True)

    print("\n=== Select Log Directory ===")
    for i, d in enumerate(dirs):
        print(f"[{i}] {d}")
    print("============================")

    while True:
        try:
            choice = input(f"Enter number (0-{len(dirs)-1}): ")
            idx = int(choice)
            if 0 <= idx < len(dirs):
                selected = dirs[idx]
                print(f"Selected: {selected}\n")
                return os.path.join(".", selected)
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

class PADNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=32, action_dim=1):
        super().__init__()
        # 1. Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.idm_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.decoder_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_idm(self, feat_t, feat_next):
        combined = torch.cat([feat_t, feat_next], dim=-1)
        return self.idm_head(combined)
    
    def forward_decoder(self, feat):
        return self.decoder_head(feat)

class PADCartPoleWrapper(gym.Wrapper):
    def __init__(self, env, feature_dim=32, learning_rate=1e-3, is_test_mode=False,load_path=None):
        super().__init__(env)
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]    
        self.feature_dim = feature_dim
        self.is_test_mode = is_test_mode

        self.device = torch.device("cpu") 
        self.pad_net = PADNetwork(self.input_dim, self.feature_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.pad_net.parameters(), lr=learning_rate)

        if load_path is not None and os.path.exists(load_path):
            self.pad_net.load_state_dict(torch.load(load_path))
            print(f"[PAD] Loaded encoder weights from {load_path}")
        elif is_test_mode:
            print("[Warning] PAD wrapper is in test mode but NO weights loaded! Result will be random.")
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(feature_dim,), dtype=np.float32
        )
        
        self.last_obs_tensor = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.last_obs_tensor = obs_tensor
        
        with torch.no_grad():
            feat = self.pad_net.forward_encoder(obs_tensor)
            
        return feat.cpu().numpy().flatten(), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        feat_t = self.pad_net.forward_encoder(self.last_obs_tensor)
        feat_next = self.pad_net.forward_encoder(next_obs_tensor)
        
        pred_action = self.pad_net.forward_idm(feat_t, feat_next)
        loss_idm = F.mse_loss(pred_action, action_tensor)
        pred_obs = self.pad_net.forward_decoder(feat_t)
        loss_rec = F.mse_loss(pred_obs, self.last_obs_tensor)
        loss = loss_idm + 0.1 * loss_rec 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.last_obs_tensor = next_obs_tensor
        return feat_next.detach().cpu().numpy().flatten(), reward, terminated, truncated, info

class RMANetwork(nn.Module):
    def __init__(self, input_dim, output_dim, history_len=50):
        super().__init__()
        self.input_flat_dim = input_dim * history_len
        
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class RMAInferenceWrapper(gym.Wrapper):
    def __init__(self, env, adaptor_path, history_len=50):
        super().__init__(env)
        self.history_len = history_len
        self.base_env = env.unwrapped
        self.obs_dim = 4
        self.act_dim = 1
        self.priv_dim = 1
        
        self.device = torch.device("cpu")
        self.adaptor = RMANetwork(self.obs_dim + self.act_dim, self.priv_dim, history_len).to(self.device)
        if os.path.exists(adaptor_path):
            self.adaptor.load_state_dict(torch.load(adaptor_path))
            print(f"[RMA] Loaded adaptor from {adaptor_path}")
        else:
            print(f"[Warning] Adaptor not found at {adaptor_path}")
        self.adaptor.eval()
        
        self.history = deque(maxlen=history_len)
        low = np.concatenate([env.observation_space.low, np.full(self.priv_dim, -np.inf)]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, np.full(self.priv_dim, np.inf)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.last_obs = None

    def _get_history_tensor(self):
        flat = []
        current_items = list(self.history)
        
        pad_len = self.history_len - len(current_items)
        for _ in range(pad_len):
            flat.extend([0.0] * (self.obs_dim + self.act_dim))
            
        for o, a in current_items:
            flat.extend(list(o) + list(a))
            
        return torch.FloatTensor([flat]).to(self.device)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        self.last_obs = obs
        with torch.no_grad():
            h = self._get_history_tensor()
            z_pred = self.adaptor(h).cpu().numpy()[0]
            
        return np.concatenate([obs, z_pred]).astype(np.float32), info

    def step(self, action):
        self.history.append((self.last_obs, action))
        next_obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = next_obs
        
        with torch.no_grad():
            h = self._get_history_tensor()
            z_pred = self.adaptor(h).cpu().numpy()[0]
            
        combined_obs = np.concatenate([next_obs, z_pred]).astype(np.float32)
        return combined_obs, reward, term, trunc, info

class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        
        return noisy_obs, info
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class ESN_FORCE:
    def __init__(self, input_dim, res_dim=300, spectral_radius=0.9, leak_rate=0.3, method="online",density=1,Q=100):
        self.input_dim = input_dim
        self.res_dim = res_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.method = method
        self.Q = Q
        self.log_pred = []
        self.log_target = []
        self.density = density
        self._init_weights()

    def _init_weights(self):
        rng = np.random.RandomState(42)
        self.Win = rng.uniform(-0.1, 0.1, (self.res_dim, self.input_dim))
        W = rng.normal(0, 1, (self.res_dim, self.res_dim))
        mask = rng.rand(self.res_dim, self.res_dim) < self.density
        W = W * mask
        self.W = W * (self.spectral_radius / max(abs(np.linalg.eigvals(W))))
        
        self.Wout = np.zeros((self.input_dim, self.res_dim))

        if "No pre-study" in self.method:
            print(f"[ESN Init] {self.method}: Initializing Wout with Zeros.")
            self.Wout = np.zeros((self.input_dim, self.res_dim))
        elif any(x in self.method for x in ["static", "Pre-study"]):
            filename = "fixed_esn_weights.npz"
            
            if os.path.exists(filename):
                try:
                    data = np.load(filename)
                    self.Win = data['Win']
                    self.W = data['W']
                    self.Wout = data['Wout']
                    print(f"[ESN Init] {self.method}: Successfully loaded Win, W, Wout from {filename}")
                except Exception as e:
                    print(f"[Error] Failed to load {filename}: {e}")
                    print("[Warning] Fallback to random initialization (Performance will be bad!)")
            else:
                print(f"[Warning] {self.method}: {filename} not found. Running with un-trained random weights.")

        self.P = np.eye(self.res_dim) * self.Q
        self.state = np.zeros(self.res_dim)

    def reset_state(self):
        self.state = np.zeros(self.res_dim)

    def reset_weights(self):
        self._init_weights()
        self.reset_state()
    
    def forward(self, u):
        u = np.clip(u, -5.0, 5.0)
        pre = np.dot(self.Win, u) + np.dot(self.W, self.state)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre)
        return self.state.copy()
    
    def predict(self): 
        pred = np.dot(self.Wout, self.state)
        self.log_pred.append(pred.copy())
        return np.clip(pred, -10.0, 10.0)
    
    def force_update(self, target, modulation=1.0):
        target = np.clip(target, -10.0, 10.0)
        self.log_target.append(target.copy())
        lambda_forget = 0.9995
        
        lambda_forget = max(0.95, lambda_forget) 
        lambda_forget = min(0.9999, lambda_forget) 
        
        z = self.state.reshape(-1, 1)
        err = target - np.dot(self.Wout, self.state)
        
        Pr = np.dot(self.P, z)
        rPr = np.dot(z.T, Pr)
        c = 1.0 / (lambda_forget + rPr + 1e-6)
        
        self.P = (1.0/lambda_forget) * (self.P - c * np.dot(Pr, Pr.T))
        
        if np.max(np.abs(self.P)) > 1e10: 
            self.P = np.eye(self.res_dim) * self.Q
            
        self.Wout += np.dot(err.reshape(-1, 1), np.dot(self.P, z).T)
        self.Wout = np.clip(self.Wout, -5.0, 5.0)
        return np.linalg.norm(err)
    
class ShiftCartPoleRender(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, is_test_mode=False, wind_bias=0.0, render_mode=None):
        super().__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([self.x_threshold * 2, np.inf, 
                         self.theta_threshold_radians * 2, np.inf], dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.state = None
        self.steps = 0
        self.max_steps = 1000
        self.is_test_mode = is_test_mode
        self.wind_bias = wind_bias 
        self.is_hell_mode = True if is_test_mode else False
        self.render_mode = render_mode
        
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action[0]) * self.force_mag
        if self.is_hell_mode: 
            force += self.wind_bias * math.cos(0.04 * self.steps)
            
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        self.steps += 1
        
        terminated = bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)
        truncated = self.steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return None

        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.screen.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0
        carty = 100 
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        
        gfxdraw.aapolygon(self.screen, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.screen, cart_coords, (0, 0, 0))

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)

        gfxdraw.aapolygon(self.screen, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.screen, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(self.screen, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(self.screen, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))

        gfxdraw.hline(self.screen, 0, self.screen_width, int(carty), (0, 0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self): 
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class ESNCartPoleWrapper(gym.Wrapper):
    def __init__(self, env, esn, train_esn=False, washout_steps=50):
        super().__init__(env)
        self.esn = esn
        self.train_esn = train_esn
        self.washout_steps = washout_steps
        self.episode_steps = 0
        
        obs_dim = self.env.observation_space.shape[0] # 4
        
        low = np.concatenate([self.env.observation_space.low, np.full(obs_dim, -10.0)]).astype(np.float32)
        high = np.concatenate([self.env.observation_space.high, np.full(obs_dim, 10.0)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.esn.reset_weights()
        self.episode_steps = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self.esn.forward(obs)
        if self.washout_steps > 0:
            pred = np.zeros_like(obs)
        else:
            pred = self.esn.predict()
        combined_obs = np.concatenate([obs, pred]).astype(np.float32)
        return combined_obs, info

    def step(self, action):
        self.episode_steps += 1
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.train_esn:
            self.esn.force_update(target=next_obs)
        
        self.esn.forward(next_obs)
        if self.episode_steps < self.washout_steps:
            pred = np.zeros_like(next_obs)
        else:
            pred = self.esn.predict()
        
        combined_obs = np.concatenate([next_obs, pred]).astype(np.float32)
        return combined_obs, reward, terminated, truncated, info

class OracleCartPoleWrapper(gym.Wrapper):
    def __init__(self, env, randomize_training=False, wind_range=(0.0, 5.0)):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.randomize_training = randomize_training
        self.wind_range = wind_range
        
        obs_dim = env.observation_space.shape[0]
        self.priv_dim = 1
        
        low = np.concatenate([env.observation_space.low, np.full(self.priv_dim, -10.0)]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, np.full(self.priv_dim, 10.0)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_privileged_info(self):
        return np.array([self.base_env.wind_bias], dtype=np.float32)

    def _randomize_env(self):
        rng = self.np_random
        new_wind = rng.uniform(self.wind_range[0], self.wind_range[1])
        
        self.base_env.wind_bias = new_wind
        self.base_env.is_hell_mode = True

    def reset(self, seed=None, options=None):
        if seed is not None:
             self.base_env.reset(seed=seed)

        if self.randomize_training:
            self._randomize_env()

        obs, info = self.env.reset(seed=seed, options=options)
        
        priv_info = self._get_privileged_info()
        combined_obs = np.concatenate([obs, priv_info]).astype(np.float32)
        return combined_obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        priv_info = self._get_privileged_info()
        combined_obs = np.concatenate([next_obs, priv_info]).astype(np.float32)
        return combined_obs, reward, terminated, truncated, info

class DRCartPoleWrapper(gym.Wrapper):
    def __init__(self, env, wind_range=(0.0, 5.0)):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.wind_range = wind_range

    def reset(self, seed=None, options=None):
        rng = self.np_random
        new_wind = rng.uniform(self.wind_range[0], self.wind_range[1])
        
        self.base_env.wind_bias = new_wind
        self.base_env.is_hell_mode = True

        return self.env.reset(seed=seed, options=options)

def train_rma_adaptor(oracle_model_path, log_parent_dir, history_len=50, n_samples=50000):
    if log_parent_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_parent_dir = f"./cartpole_rma_logs_{timestamp}/"
    
    os.makedirs(log_parent_dir, exist_ok=True)
    print(f"Directory prepared: {log_parent_dir}")
    target_name = "SAC-RMA" 
    extensions = [".zip", "_vecnormalize.pkl", "_info.json"]
    
    for ext in extensions:
        src_file = oracle_model_path + ext
        if os.path.exists(src_file):
            dst_file = os.path.join(log_parent_dir, target_name + ext)
            shutil.copy2(src_file, dst_file)
            print(f"Copied & Renamed: {os.path.basename(src_file)} -> {target_name}{ext}")
    if not os.path.exists(oracle_model_path + ".zip"):
        print(f"Error: Oracle model not found at {oracle_model_path}.zip")
        return
    
    oracle_policy = SAC.load(oracle_model_path)
    print(f"Loaded Oracle Policy from: {oracle_model_path}")
    env = ShiftCartPoleRender(is_test_mode=False) 
    env = TimeLimit(env, max_episode_steps=1000)
    env = OracleCartPoleWrapper(env, randomize_training=True, wind_range=(0, 11.0))
    env = DummyVecEnv([lambda: env]) 
    vec_norm_path = oracle_model_path + "_vecnormalize.pkl"
    if os.path.exists(vec_norm_path):
        print("Loading VecNormalize stats for data collection...")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False 
        env.norm_reward = False
    else:
        print("VecNormalize not found. Using raw observations.")
    adaptor = RMANetwork(input_dim=5, output_dim=1, history_len=history_len)
    optimizer = optim.Adam(adaptor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history_buffer = deque(maxlen=history_len)
    obs = env.reset()
    last_state = obs[0][:4] 
    
    losses = []
    for i in range(n_samples):
        action, _ = oracle_policy.predict(obs, deterministic=True)
        true_wind = torch.FloatTensor([obs[0][4]]).unsqueeze(0)
        history_buffer.append(np.concatenate([last_state, action[0]]))
        obs, reward, done, info = env.step(action)
        last_state = obs[0][:4]
        if len(history_buffer) == history_len:
            flat_hist = np.concatenate(history_buffer)
            input_tensor = torch.FloatTensor(flat_hist).unsqueeze(0)
            
            pred_wind = adaptor(input_tensor)
            loss = loss_fn(pred_wind, true_wind)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        if (i+1) % 5000 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"RMA Step {i+1}/{n_samples} | Loss: {avg_loss:.6f}")
    save_path = os.path.join(log_parent_dir, "SAC-RMA_adaptor.pth")
    torch.save(adaptor.state_dict(), save_path)
    print(f"RMA Adaptor saved to: {save_path}")
    env.close()

def train_and_save_cartpole(total_timesteps=100000, early_stop_threshold=1000,model_type="PPO_ESN", train_esn_online=False, log_parent_dir=None, learning_rate=3e-4,n_steps=2048,batch_size=256,n_epochs=10,gamma=0.99,gae_lambda=0.95,clip_range=0.2,ent_coef=0.0,max_grad_norm=0.5,net_arch=[64, 64],esn_res_dim=300,esn_spectral_radius=0.9,esn_leak_rate=0.3,max_episode_steps=1000,washout_steps=0):
    
    if log_parent_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"./cartpole_shift_logs_{timestamp}/"
    else:
        log_dir = log_parent_dir

    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, model_type)

    print(f"\n=== Start Training: {model_type} ===")
    
    hyperparams = {
        "model_type": model_type,
        "total_timesteps": total_timesteps,
        "train_esn_online": train_esn_online,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "net_arch": net_arch,
        "esn_res_dim": esn_res_dim,
        "esn_spectral_radius": esn_spectral_radius,
        "esn_leak_rate": esn_leak_rate,
        "washout_steps": washout_steps,
        "max_episode_steps": max_episode_steps,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm
    }
    with open(os.path.join(log_dir, f"{model_type}_info.json"), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    def make_env():
        env = ShiftCartPoleRender(is_test_mode=False, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)

        if "Oracle" in model_type or "RMA" in model_type:
            print("Wrapper: OracleCartPoleWrapper (Phase 1 for RMA / Oracle)")
            env = OracleCartPoleWrapper(env, randomize_training=True, wind_range=(0, 9.0))
        
        elif "ESN" in model_type:
            input_dim = env.observation_space.shape[0]
            esn = ESN_FORCE(input_dim=input_dim, res_dim=esn_res_dim, spectral_radius=esn_spectral_radius, leak_rate=esn_leak_rate, method=model_type)
            env = ESNCartPoleWrapper(env, esn, train_esn=train_esn_online, washout_steps=washout_steps)

        elif "SAC-DR" in model_type:
            print("Wrapper: DRCartPoleWrapper (Random Wind, No Privileged Info)")
            env = DRCartPoleWrapper(env, wind_range=(0, 9.0))

        elif "PAD" in model_type:
            print("Wrapper: PADCartPoleWrapper (Fixed Encoder Mode)")
            pretrained_path = "pretrained_models/cartpole_pad_encoder.pth"
            
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Run pretrain_pad.py first! Cannot find {pretrained_path}")

            env = ShiftCartPoleRender(is_test_mode=False, render_mode="rgb_array")
            env = ObservationNoiseWrapper(env, noise_std=0.01)
            env = PADCartPoleWrapper(
                env, 
                feature_dim=32, 
                learning_rate=0.0,
                load_path=pretrained_path
            )
            
        return env
    
    env = DummyVecEnv([make_env])    
    if "PAD" in model_type:
        print("PAD Mode: Observation Normalization OFF, Reward Normalization ON")
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    if model_type == "LSTM":
        print("Using RecurrentPPO (LSTM policy)...")
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, learning_rate=linear_schedule(learning_rate),
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, max_grad_norm=max_grad_norm
        )
    elif "PPO" in model_type:
        print(f"Using Standard PPO ({model_type})...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=linear_schedule(hyperparams["learning_rate"]),
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            clip_range=hyperparams["clip_range"],
            max_grad_norm=hyperparams["max_grad_norm"],
            policy_kwargs=dict(net_arch=[dict(pi=hyperparams["net_arch"], vf=hyperparams["net_arch"])]),
            ent_coef=hyperparams["ent_coef"]
        )
    elif "SAC" in model_type:
        print(f"Using Standard SAC ({model_type})...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hyperparams["learning_rate"],
            batch_size=hyperparams["batch_size"],
            buffer_size=100000,
            learning_starts=hyperparams["total_timesteps"]*0.05,
            ent_coef='auto',
            gamma=hyperparams["gamma"],
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.save(save_path)
        if "PAD" not in model_type:
            env.save(os.path.join(log_dir, f"{model_type}_vecnormalize.pkl"))

        if "PAD" in model_type:
            current_env = env.envs[0]
            
            while not hasattr(current_env, 'pad_net'):
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                else:
                    break
            
            if hasattr(current_env, 'pad_net'):
                pad_weight_path = os.path.join(log_dir, f"{model_type}_encoder.pth")
                torch.save(current_env.pad_net.state_dict(), pad_weight_path)
                print(f"Saved PAD encoder weights to: {pad_weight_path}")

        print(f"Model and stats saved to: {save_path}")
        if "RMA" in model_type:
            train_rma_adaptor(
                oracle_model_path=save_path, 
                log_parent_dir=log_dir
            )

    env.close()
if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_log_dir = f"./cartpole_shift_logs_{timestamp}/"
    print(f"New experiment logs will be saved in: {current_log_dir}")
    STEPS =18000  
    train_and_save_cartpole(
        total_timesteps=25000, 
        model_type="SAC-ESN(No pre-study)", 
        train_esn_online=True,
        log_parent_dir=current_log_dir,
        esn_res_dim=300,
        esn_spectral_radius=0.9
    )
    train_and_save_cartpole(
        total_timesteps=18000, 
        model_type="SAC",
        log_parent_dir=current_log_dir
    )

    train_and_save_cartpole(
        total_timesteps=30000, 
        model_type="SAC-Oracle", # SACベースのOracle
        log_parent_dir=current_log_dir
    )

    oracle_model_path_base = os.path.join(current_log_dir, "SAC-Oracle")
    
    train_rma_adaptor(
        oracle_model_path=oracle_model_path_base, 
        log_parent_dir=current_log_dir,
        history_len=50,
        n_samples=100000 
    )  
    
    train_and_save_cartpole(
        total_timesteps=30000, 
        model_type="SAC-PAD", 
        log_parent_dir=current_log_dir
    )
    
    train_and_save_cartpole(
        total_timesteps=30000, 
        model_type="SAC-ESN(Pre-study)", 
        train_esn_online=True,
        log_parent_dir=current_log_dir,
        esn_res_dim=300,
        esn_spectral_radius=0.9
    )

    train_and_save_cartpole(
        total_timesteps=30000, 
        model_type="SAC-DR", 
        log_parent_dir=current_log_dir
    )
    
    print("\nAll training finished!")