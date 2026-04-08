import gymnasium as gym
import numpy as np
import os
import csv
import json
import cv2
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit, RecordVideo
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import math

from train_halfcheetah import (
    ESN_FORCE, ESNCheetahWrapper, 
    OracleCheetahWrapper, 
    PADCheetahWrapper, PADNetwork,
    RMAInferenceWrapper, RMANetwork,
    DRCheetahWrapper
)

os.environ["MUJOCO_GL"] = "glfw"

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
                return os.path.join(".", dirs[idx])
        except ValueError: pass

class VariableFrictionCheetah(HalfCheetahEnv):
    def __init__(self, change_step=500, friction_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.change_step = change_step
        self.friction_factor = friction_factor
        self.steps = 0
        self.changed = False
        
        self.model.geom_size[0][0] = 400.0  

        floor_mat_id = self.model.geom_matid[0]
        self.model.mat_texrepeat[floor_mat_id, 0] = 400.0
        self.model.mat_texrepeat[floor_mat_id, 1] = 20.0 
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

class TextOverlayWrapper(gym.Wrapper):
    def __init__(self, env, trigger_step=500, text="Friction Changed!", color=(255, 50, 50)):
        super().__init__(env)
        self.trigger_step = trigger_step
        self.text = text
        self.color = color 
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        return self.env.step(action)

    def render(self):
        frame = self.env.render()
        if frame is None or frame.size == 0: return frame

        env_steps = getattr(self.env.unwrapped, 'steps', self.current_step)

        if env_steps >= self.trigger_step:
            frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(self.text, font, font_scale, thickness)
            x, y = 30, 50
            
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 10), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, self.text, (x, y), font, font_scale, self.color, thickness, cv2.LINE_AA)
            
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), self.color, 12) 

        return frame
    
def get_esn_info(log_dir, method):
    json_path = os.path.join(log_dir, f"{method}_info.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except: pass
    return {}

def create_test_env(method, friction_factor, log_dir, render_mode=None, save_video=False, video_dir="videos", video_prefix="cheetah"):
    info = get_esn_info(log_dir, method)
    
    def make_test_env():
        env = VariableFrictionCheetah(change_step=500, friction_factor=friction_factor, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=info.get("max_episode_steps", 1000))
        
        if save_video:
            env = TextOverlayWrapper(env, trigger_step=500, text=f"Friction x{friction_factor}", color=(255, 0, 0))
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env, video_folder=video_dir,
                name_prefix=f"{video_prefix}_{method}_f{friction_factor}",
                episode_trigger=lambda x: True, disable_logger=True
            )
            
        if "Oracle" in method:
            env = OracleCheetahWrapper(env, randomize_training=False)
        elif "RMA" in method:
            adaptor_path = os.path.join(log_dir, "SAC-RMA_adaptor.pth")
            env = RMAInferenceWrapper(env, adaptor_path=adaptor_path)
        elif "PAD" in method:
            encoder_path = os.path.join(log_dir, "SAC-PAD_encoder.pth")
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
            env = ESNCheetahWrapper(env, esn, train_esn=True, washout_steps=info.get("washout_steps", 200))
            
        return env

    env = DummyVecEnv([make_test_env])

    if "PAD" not in method:
        stats_path = os.path.join(log_dir, f"{method}_vecnormalize.pkl")
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False 
            env.norm_reward = False
        else:
            print(f"Warning: {stats_path} not found for {method}.")

    return env


def run_evaluation(env, model, method, render=False):
    obs = env.reset()
    total_reward = 0.0
    lstm_states = None
    episode_start = True

    while True:
        if render:
            env.render()

        if "LSTM" in method:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        total_reward += reward[0]
        if dones[0]:
            break
        
    return total_reward


def plot_robustness(results, x_values, xlabel="Friction Multiplier", legend_map=None, font_conf=None):
    print(f"\n=== Generating {xlabel} Robustness Graph (Academic Style) ===")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    default_fonts = {
        'title': 22,
        'label': 21,
        'tick': 21,
        'legend': 16
    }
    if font_conf:
        default_fonts.update(font_conf)
    
    styles = {
        "SAC":                   {"c": "#1f77b4", "m": "o"}, 
        "SAC-DR":                {"c": "#2ca02c", "m": "H"}, 
        "SAC-RMA":               {"c": "#d62728", "m": "s"}, 
        "SAC-ESN(Pre-study)":    {"c": "#7f7f7f", "m": "*"}, 
        "SAC-Oracle":            {"c": "#17becf", "m": "v"}, 
        "SAC-ESN(No pre-study)": {"c": "#e377c2", "m": "^"}, 
    }

    legend_map = {
        "SAC":                   "SAC", 
        "SAC-DR":                "+DR", 
        "SAC-RMA":               "+RMA", 
        "SAC-ESN(Pre-study)":    "+ESN(Pre)", 
        "SAC-Oracle":            "Ground Truth", 
        "SAC-ESN(No pre-study)": "+ESN (No pre)", 
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for method, data in results.items():
        if not data["mean"]: continue
        
        means = np.array(data["mean"])
        stds = np.array(data["std"])
        x = np.array(x_values)

        style = styles.get(method, {"c": "black", "m": "x"})
        display_label = method
        if legend_map and method in legend_map:
            display_label = legend_map[method]

        ax.plot(x, means, label=display_label, color=style["c"], marker=style["m"], 
                linewidth=2.0, markersize=8, markerfacecolor='white', markeredgewidth=1.5)
        ax.fill_between(x, means - stds, means + stds, color=style["c"], alpha=0.15)

    ax.set_xlabel(xlabel, fontsize=default_fonts['label'])
    ax.set_ylabel("Reward", fontsize=default_fonts['label'])
    ax.tick_params(axis='both', which='major', labelsize=default_fonts['tick'])
    
    min_x, max_x = ax.get_xlim()
    if len(x_values) > 1:
        ax.set_xticks(np.arange(math.floor(min_x), math.ceil(max_x)+1, 1)) 
        
    ax.legend(fontsize=default_fonts['legend'], loc='best', frameon=True, edgecolor='black', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"Fig_{xlabel}_Robustness.png", bbox_inches='tight')
    print(f"Saved graph to Fig_{xlabel}_Robustness.png")


if __name__ == "__main__":
    LOG_DIR = select_log_directory()    
    
    methods = [
        "SAC",
        "SAC-ESN(No pre-study)",
        "SAC-ESN(Pre-study)",
        "SAC-DR",
        "SAC-RMA",
    ]

    friction_factors = np.arange(1, 10.1, 0.5)
    n_seeds = 10
    results = {m: {"mean": [], "std": []} for m in methods}
    print(f"Starting Cheetah Friction Experiment: {len(friction_factors)} conditions x {n_seeds} seeds")
    
    for method in methods:
        print(f"\nTesting {method} ", end="")
        means, stds = [], []
        
        model_path = os.path.join(LOG_DIR, method + ".zip")
        if not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            continue

        if "LSTM" in method:
            model = RecurrentPPO.load(model_path)
        elif "SAC" in method:
            model = SAC.load(model_path)
        elif "PPO" in method:
            model = PPO.load(model_path)
        else:
            print(f"Unknown model architecture for {method}")
            continue
        
        for f_factor in friction_factors:
            rewards = []
            
            env = create_test_env(
                method=method, 
                friction_factor=f_factor, 
                log_dir=LOG_DIR, 
                save_video=False
            )
            
            for _ in range(n_seeds):
                r = run_evaluation(env, model, method, render=False)
                rewards.append(r)
            
            env.close() 
            
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            means.append(mean_r)
            stds.append(std_r)
            print(f"[x{f_factor}:{int(mean_r)}]", end=" ", flush=True)
            
        results[method]["mean"] = means
        results[method]["std"] = stds
        
    plot_robustness(results, friction_factors, xlabel="Friction Multiplier")
    
    with open("robustness_results_cheetah_friction.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Friction"]
        for m in methods: header.extend([f"{m}_Mean", f"{m}_Std"])
        writer.writerow(header)
        
        for i, f_val in enumerate(friction_factors):
            row = [f_val]
            for m in methods:
                if results[m]["mean"]:
                    row.extend([results[m]["mean"][i], results[m]["std"][i]])
                else:
                    row.extend(["NaN", "NaN"])
            writer.writerow(row)
            
    print("\nAll Done!")