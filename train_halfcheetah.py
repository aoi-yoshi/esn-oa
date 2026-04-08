import gymnasium as gym
import numpy as np
import os
import json
import shutil
from collections import deque
from datetime import datetime
from stable_baselines3 import PPO, SAC
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def select_log_directory():
    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('halfcheetah_logs_')]
    dirs.sort(reverse=True)
    if not dirs:
        print("No log directories found!")
        return "./halfcheetah_logs"
    print("\n=== Select Log Directory ===")
    for i, d in enumerate(dirs):
        print(f"[{i}] {d}")
    while True:
        try:
            choice = input(f"Enter number (0-{len(dirs)-1}): ")
            idx = int(choice)
            if 0 <= idx < len(dirs):
                return os.path.join(".", dirs[idx])
        except ValueError: pass

class ESN_FORCE:
    def __init__(self, input_dim, res_dim=500, spectral_radius=0.9, leak_rate=0.3, method="online",density=1,Q=100):
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
            filename = "cheetah_fixed_esn_weights.npz"
            
            if os.path.exists(filename):
                try:
                    data = np.load(filename)
                    self.Win = data['Win']
                    self.W = data['W']
                    self.Wout = data['Wout']
                    print(f"[ESN Init] {self.method}: Successfully loaded Win, W, Wout from {filename}")
                except Exception as e:
                    print(f"[Error] Failed to load {filename}: {e}")
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

class CrippledHalfCheetah(HalfCheetahEnv):
    def __init__(self, cripple_joint=None, **kwargs):
        super().__init__(**kwargs)
        self.cripple_joint = cripple_joint

    def step(self, action):
        if self.cripple_joint is not None:
            action = np.array(action)
            action[self.cripple_joint] = 0.0
        return super().step(action)

class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_std=0.01):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise, info

class ESNCheetahWrapper(gym.Wrapper):
    def __init__(self, env, esn, train_esn=False, washout_steps=200):
        super().__init__(env)
        self.esn = esn
        self.train_esn = train_esn
        self.washout_steps = washout_steps
        self.episode_steps = 0
        obs_dim = self.env.observation_space.shape[0]
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
        return np.concatenate([obs, pred]).astype(np.float32), info

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
        return np.concatenate([next_obs, pred]).astype(np.float32), reward, terminated, truncated, info

class OracleCheetahWrapper(gym.Wrapper):
    def __init__(self, env, randomize_training=True):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.randomize_training = randomize_training
        self._dummy_priv = self._get_privileged_info()
        self.priv_dim = self._dummy_priv.shape[0]
        
        obs_dim = env.observation_space.shape[0]
        low = np.concatenate([env.observation_space.low, np.full(self.priv_dim, -np.inf)]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, np.full(self.priv_dim, np.inf)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.default_friction = self.base_env.model.geom_friction.copy()

    def _get_privileged_info(self):
        friction = self.base_env.model.geom_friction.flat[0]
        return np.array([friction], dtype=np.float32)

    def _randomize_env(self):
        rng = self.np_random
        f_scale = rng.uniform(0.5, 10.0)
        new_friction = self.default_friction * f_scale
        self.base_env.model.geom_friction[:] = new_friction

    def reset(self, seed=None, options=None):
        if seed is not None: self.base_env.reset(seed=seed)
        if self.randomize_training:
            self._randomize_env()
        obs, info = self.env.reset(seed=seed, options=options)
        priv_info = self._get_privileged_info()
        return np.concatenate([obs, priv_info]).astype(np.float32), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        priv_info = self._get_privileged_info()
        return np.concatenate([next_obs, priv_info]).astype(np.float32), reward, terminated, truncated, info

class DRCheetahWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.default_friction = self.base_env.model.geom_friction.copy()

    def _randomize_env(self):
        rng = self.np_random
        f_scale = rng.uniform(1, 5.0)
        new_friction = self.default_friction * f_scale
        self.base_env.model.geom_friction[:] = new_friction

    def reset(self, seed=None, options=None):
        if seed is not None: self.base_env.reset(seed=seed)
        self._randomize_env()
        return self.env.reset(seed=seed, options=options)

class PADNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=32, action_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.idm_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_idm(self, feat_t, feat_next):
        combined = torch.cat([feat_t, feat_next], dim=-1)
        return self.idm_head(combined)
    
    def forward_decoder(self, feat):
        return self.decoder_head(feat)

class PADCheetahWrapper(gym.Wrapper):
    def __init__(self, env, feature_dim=32, learning_rate=1e-3, is_test_mode=False, load_path=None):
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
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(feature_dim,), dtype=np.float32)
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
            nn.Linear(self.input_flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class RMAInferenceWrapper(gym.Wrapper):
    def __init__(self, env, adaptor_path, history_len=50):
        super().__init__(env)
        self.history_len = history_len
        self.base_env = env.unwrapped
        self.obs_dim = env.observation_space.shape[0] 
        self.act_dim = env.action_space.shape[0]      
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
        return np.concatenate([next_obs, z_pred]).astype(np.float32), reward, term, trunc, info

def train_rma_adaptor_cheetah(oracle_model_path, log_parent_dir, history_len=50, n_samples=100000):
    print("\n=== Start RMA Phase 2: Training Adaptor (Cheetah) ===")
    
    if log_parent_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_parent_dir = f"./halfcheetah_rma_logs_{timestamp}/"
    os.makedirs(log_parent_dir, exist_ok=True)
    model_name = os.path.basename(oracle_model_path)
    target_name = "SAC-RMA"
    for ext in [".zip", "_vecnormalize.pkl", "_info.json"]:
        src = oracle_model_path + ext
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(log_parent_dir, target_name + ext))
    if not os.path.exists(oracle_model_path + ".zip"):
        print("Oracle model not found!")
        return
    oracle_policy = SAC.load(oracle_model_path)
    env = CrippledHalfCheetah(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=1000)
    env = OracleCheetahWrapper(env, randomize_training=True)
    env = DummyVecEnv([lambda: env])

    vec_norm_path = oracle_model_path + "_vecnormalize.pkl"
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    obs_dim = env.observation_space.shape[0] - 1
    act_dim = env.action_space.shape[0]
    
    adaptor = RMANetwork(input_dim=obs_dim + act_dim, output_dim=1, history_len=history_len)
    optimizer = optim.Adam(adaptor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    history_buffer = deque(maxlen=history_len)
    obs = env.reset()
    last_state = obs[0][:-1]
    
    losses = []
    print(f"Collecting {n_samples} samples...")
    
    for i in range(n_samples):
        action, _ = oracle_policy.predict(obs, deterministic=True)
        true_priv = torch.FloatTensor([obs[0][-1]]).unsqueeze(0)
        
        history_buffer.append(np.concatenate([last_state, action[0]]))
        obs, reward, done, info = env.step(action)
        last_state = obs[0][:-1]
        
        if len(history_buffer) == history_len:
            flat_hist = np.concatenate(history_buffer)
            input_tensor = torch.FloatTensor(flat_hist).unsqueeze(0)
            pred_priv = adaptor(input_tensor)
            loss = loss_fn(pred_priv, true_priv)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if (i+1) % 5000 == 0:
            avg = np.mean(losses[-100:]) if losses else 0
            print(f"RMA Step {i+1}/{n_samples} | Loss: {avg:.6f}")
            
    save_path = os.path.join(log_parent_dir, "SAC-RMA_adaptor.pth")
    torch.save(adaptor.state_dict(), save_path)
    print(f"RMA Adaptor saved to: {save_path}")
    env.close()
def train_and_save_cheetah(total_timesteps=500000, model_type="SAC", train_esn_online=True, log_parent_dir=None, learning_rate=3e-4,n_steps=2048,batch_size=256,n_epochs=10,gamma=0.99,gae_lambda=0.95,clip_range=0.2,ent_coef=0.0,max_grad_norm=0.5,net_arch=[256, 256],esn_res_dim=500,esn_spectral_radius=0.9,esn_leak_rate=0.3,max_episode_steps=1000,washout_steps=0,density=1 ):
    
    if log_parent_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"./halfcheetah_logs_{timestamp}/"
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
        "gamma": gamma,
        "net_arch": net_arch,
        "esn_res_dim": esn_res_dim,
        "esn_spectral_radius": esn_spectral_radius,
        "esn_leak_rate": esn_leak_rate,
        "max_episode_steps": max_episode_steps,
        "washout_steps": washout_steps,
        "density" : density
    }
    with open(os.path.join(log_dir, f"{model_type}_info.json"), 'w') as f:
        json.dump(hyperparams, f, indent=4)

    def make_env():
        env = CrippledHalfCheetah(cripple_joint=None, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)

        if "Oracle" in model_type or "RMA" in model_type:
            env = OracleCheetahWrapper(env, randomize_training=True)

        elif "DR" in model_type:
            env = DRCheetahWrapper(env)
        
        elif "PAD" in model_type:
            env = DRCheetahWrapper(env)
            env = ObservationNoiseWrapper(env, noise_std=0.01)
            pretrained_path = "pretrained_models/cheetah_pad_encoder.pth"
            lr = 0.0 
            if not os.path.exists(pretrained_path):
                print("[Warning] Pretrained encoder not found! Using random initialization.")
            
            env = PADCheetahWrapper(env, feature_dim=32, learning_rate=lr, load_path=pretrained_path)
            
        elif "ESN" in model_type:
            env = DRCheetahWrapper(env)
            input_dim = env.observation_space.shape[0]
            esn = ESN_FORCE(input_dim=input_dim, res_dim=esn_res_dim, spectral_radius=esn_spectral_radius, leak_rate=esn_leak_rate, method=model_type, density=density)
            env = ESNCheetahWrapper(env, esn, train_esn=train_esn_online, washout_steps=washout_steps)
            
        return env
    
    env = DummyVecEnv([make_env])
    if "PAD" in model_type:
        print("PAD Mode: Observation Normalization OFF, Reward Normalization ON")
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    if model_type == "LSTM":
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=linear_schedule(learning_rate), n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, max_grad_norm=max_grad_norm)
    elif "PPO" in model_type:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=linear_schedule(learning_rate), n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, clip_range=clip_range, max_grad_norm=max_grad_norm, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=net_arch)]), ent_coef=ent_coef)
    elif "SAC" in model_type:
        print(f"Using SAC ({model_type})...")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate, batch_size=batch_size, buffer_size=500000, learning_starts=total_timesteps*0.03, ent_coef='auto', gamma=gamma, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256]))
        
    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        model.save(save_path)
        if "PAD" not in model_type:
            env.save(os.path.join(log_dir, f"{model_type}_vecnormalize.pkl"))
        
        if "PAD" in model_type:
            current_env = env.envs[0]
            while not hasattr(current_env, 'pad_net'):
                if hasattr(current_env, 'env'): current_env = current_env.env
                else: break
            if hasattr(current_env, 'pad_net'):
                torch.save(current_env.pad_net.state_dict(), os.path.join(log_dir, f"{model_type}_encoder.pth"))
        if "RMA" in model_type:
            train_rma_adaptor_cheetah(save_path, log_parent_dir)

    print(f"Model saved: {save_path}")
    env.close()

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_log_dir = f"./halfcheetah_logs_{timestamp}/"
    print(f"All results for this run will be saved in: {current_log_dir}")
    STEPS=200000
    train_and_save_cheetah(
        total_timesteps=STEPS, 
        model_type="SAC-ESN(Pre-study)", 
        train_esn_online=True,
        log_parent_dir=current_log_dir,
        esn_res_dim=500,
        esn_spectral_radius=0.9,
        esn_leak_rate=0.3,
        max_episode_steps=1000,
        washout_steps=0
    )
    
    train_and_save_cheetah(
        total_timesteps=STEPS, 
        model_type="SAC-ESN(No pre-study)", 
        train_esn_online=True,
        log_parent_dir=current_log_dir,
        esn_res_dim=500,
        esn_spectral_radius=0.9,
        esn_leak_rate=0.3,
        max_episode_steps=1000,
        washout_steps=0
    )

    train_and_save_cheetah(total_timesteps=STEPS, model_type="SAC-DR", log_parent_dir=current_log_dir)

    train_and_save_cheetah(
        total_timesteps=STEPS, 
        model_type="SAC",
        log_parent_dir=current_log_dir,
        max_episode_steps=1000,
    )

    # 3. PAD (※事前に pretrain_pad_cheetah.py を実行しておく必要あり)
    train_and_save_cheetah(total_timesteps=STEPS, model_type="SAC-PAD", log_parent_dir=current_log_dir)
    

    # 2. SAC
    train_and_save_cheetah(
        total_timesteps=STEPS, 
        model_type="SAC",
        log_parent_dir=current_log_dir, # 同じフォルダに保存
        max_episode_steps=1000
    )

    

    print("\nAll training finished!")