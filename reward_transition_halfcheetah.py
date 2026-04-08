import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_halfcheetah import ESN_FORCE, ESNCheetahWrapper, OracleCheetahWrapper, RMAInferenceWrapper, PADCheetahWrapper, DRCheetahWrapper
from eval_halfcheetah import get_esn_info
import json
from matplotlib.ticker import MultipleLocator

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def select_log_directory():
    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('halfcheetah_logs_')]
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
                print(f"Selected: {dirs[idx]}\n")
                return os.path.join(".", dirs[idx])
        except ValueError:
            pass
class VariableFrictionCheetah(HalfCheetahEnv):
    def __init__(self, change_step=500, friction_factor=0.3, **kwargs):
        super().__init__(**kwargs)
        self.change_step = change_step
        self.friction_factor = friction_factor
        self.steps = 0
        self.changed = False
        self.original_friction = self.model.geom_friction.copy()

    def step(self, action):
        self.steps += 1
        
        if self.steps >= self.change_step and not self.changed:
            self.model.geom_friction[:, 0] = self.original_friction[:, 0] * self.friction_factor
            self.changed = True
            
        return super().step(action)
        
    def reset(self, **kwargs):
        self.steps = 0
        self.changed = False
        self.model.geom_friction[:] = self.original_friction
        return super().reset(**kwargs)

LOG_DIR = "./halfcheetah_logs_fixed/"

def get_esn_res_dim(log_dir):
    json_path = os.path.join(log_dir, "Online_Cheetah_info.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"esn_res_dim": 500, "esn_spectral_radius": 0.9, "esn_leak_rate": 0.3}

def run_cheetah_history(method, friction_factor, max_steps=1000):

    info = get_esn_info(LOG_DIR, method)
    def make_test_env():
        env = VariableFrictionCheetah(change_step=500, friction_factor=friction_factor)
        env = TimeLimit(env, max_episode_steps=info.get("max_episode_steps", 1000))
        
        if "Oracle" in method:
            env = OracleCheetahWrapper(env, randomize_training=False)

        elif "RMA" in method:
            adaptor_path = os.path.join(LOG_DIR, "SAC-RMA_adaptor.pth")
            env = RMAInferenceWrapper(env, adaptor_path=adaptor_path)

        elif "PAD" in method:
            encoder_path = os.path.join(LOG_DIR, "SAC-PAD_encoder.pth")
            env = PADCheetahWrapper(env, feature_dim=32, is_test_mode=True, load_path=encoder_path)

        elif "DR" in method:
            pass

        elif "ESN" in method:
            input_dim = env.observation_space.shape[0]
            esn = ESN_FORCE(input_dim=input_dim, 
                            res_dim=info.get("esn_res_dim", 500), 
                            spectral_radius=info.get("esn_spectral_radius", 0.5), 
                            leak_rate=info.get("esn_leak_rate", 0.3),
                            method=method)
            
            if any(x in method for x in ["static", "Pre-study"]):
                npy_path = os.path.join("./", f"cheetah_fixed_wout.npy")
                if os.path.exists(npy_path):
                    esn.Wout = np.load(npy_path)
                else:
                    print(f"Warning: {npy_path} not found.")

                train_esn = False if "static" in method else True
            elif "No pre-study" in method:
                esn.reset_weights()
                train_esn = True
            
            env = ESNCheetahWrapper(env, esn, train_esn=train_esn, washout_steps=info.get("washout_steps", 200))
            
        return env

    env = DummyVecEnv([make_test_env])
    if "PAD" not in method:
        stats_path = os.path.join(LOG_DIR, f"{method}_vecnormalize.pkl")
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False 
            env.norm_reward = False
        else:
            print(f"Warning: {stats_path} not found for {method}.")


    model_path = os.path.join(LOG_DIR, method + ".zip")

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        env.close()
        return 0.0

    if "LSTM" in method:
        model = RecurrentPPO.load(model_path)
    elif "SAC" in method:
        model = SAC.load(model_path)
    elif "PPO" in method:
        model = PPO.load(model_path)
    else:
        print(f"Unknown model architecture for {method}")
        return 0.0

    obs = env.reset()
    done = False
    lstm_states = None
    episode_start = True
    reward_history = []

    for _ in range(max_steps):
        if method == "LSTM":
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        reward_history.append(reward[0])
        
        if done[0]:
            break
            
    env.close()
    if len(reward_history) < max_steps:
        reward_history.extend([0.0] * (max_steps - len(reward_history)))
        
    return np.array(reward_history)

def plot_reward_time_series(results, friction_factor):
    print("\n=== Generating Time-Series Graph ===")
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    FS_TITLE = 20
    FS_LABEL = 21
    FS_TICK = 21
    FS_LEGEND = 16

    legend_names = {
        "SAC": "SAC",
        "SAC-ESN(No pre-study)": "+ESN-OA", 
        "SAC-ESN(Pre-study)": "+ESN-OA-PT",
        "SAC-RMA": "+RMA",
        "SAC-DR": "+DR",
        "SAC-Oracle": "Ground Truth"
    }
    
    colors = {
        "SAC": "#1f77b4",
        "SAC-ESN(static)": "#2ca02c",
        "SAC-ESN(No pre-study)": "#e377c2",
        "SAC-ESN(Pre-study)": "#7f7f7f",
        "SAC-LSTM": "#8c564b"
    }

    styles = {
        "SAC":                   {"c": "#1f77b4", "m": "o"}, 
        "SAC-DR":                {"c": "#2ca02c", "m": "H"}, 
        "SAC-RMA":               {"c": "#d62728", "m": "s"}, 
        "SAC-ESN(Pre-study)":    {"c": "#7f7f7f", "m": "*"}, 
        "SAC-Oracle":            {"c": "#17becf", "m": "v"}, 
        "SAC-ESN(No pre-study)": {"c": "#e377c2", "m": "^"}, 
    }

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    WINDOW = 0 

    for method, histories in results.items():
        if len(histories) == 0: continue
        data = np.array(histories)
        mean_history = np.mean(data, axis=0)
        std_history = np.std(data, axis=0)
        if WINDOW > 1:
            smooth_mean = moving_average(mean_history, WINDOW)
            smooth_std = moving_average(std_history, WINDOW)
            x = np.arange(len(smooth_mean)) + (WINDOW // 2)
        else:
            smooth_mean = mean_history
            smooth_std = std_history
            x = np.arange(len(smooth_mean))
        
        x = np.arange(len(smooth_mean)) + (WINDOW // 2)
        label_name = legend_names.get(method, method)
        color = colors.get(method, "black")
        
        ax.plot(x, smooth_mean, label=label_name, color=color, 
                linewidth=2.0, linestyle="-")
        ax.fill_between(x, smooth_mean - smooth_std, smooth_mean + smooth_std, 
                        color=color, alpha=0.15, edgecolor=None)
    ax.axvline(x=500, color='#d62728', linestyle=':', linewidth=2.0, 
               label=f"Friction Change")
    
    ax.set_xlabel("Steps", fontsize=FS_LABEL)
    ax.set_ylabel("Reward", fontsize=FS_LABEL)
    
    ax.set_xlim(0, 1000)

    ax.tick_params(axis='both', labelsize=FS_TICK)
    xtick_interval = 100
    if xtick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xtick_interval))
    
    ax.legend(fontsize=FS_LEGEND, loc='upper right', frameon=True, edgecolor='black')
    
    plt.tight_layout()
    
    save_path = "Fig_Reward_TimeSeries.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved graph to {save_path}")
if __name__ == "__main__":
    LOG_DIR = select_log_directory()

    methods = [
        "SAC", 
        "SAC-DR", 
        "SAC-RMA", 
        "SAC-Oracle",
        "SAC-ESN(No pre-study)",
        "SAC-ESN(Pre-study)"
    ]

    target_friction = 10
    n_seeds = 10
    info = get_esn_info(LOG_DIR, methods[0]) 
    max_steps = info.get("max_episode_steps", 1000)
    results = {m: [] for m in methods}
    
    print(f"Starting Time-Series Experiment: Friction x{target_friction}, {max_steps} steps, {n_seeds} seeds")
    
    for method in methods:
        print(f"\nTesting {method} ", end="")
        
        for i in range(n_seeds):
            history = run_cheetah_history(method, friction_factor=target_friction, max_steps=max_steps)
            
            if history is not None:
                results[method].append(history)
                print(f".", end="", flush=True)
            else:
                print("x", end="", flush=True)

    plot_reward_time_series(results, friction_factor=target_friction)
    print("\nAll Done!")