import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import csv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from train_cartpole import ESN_FORCE, ShiftCartPoleRender, ESNCartPoleWrapper, DRCartPoleWrapper, OracleCartPoleWrapper, PADCartPoleWrapper, ObservationNoiseWrapper, RMAInferenceWrapper
from train_cartpole import select_log_directory
from gymnasium.wrappers import TimeLimit, RecordVideo
import json


def get_esn_res_dim(log_dir):
    json_path = os.path.join(log_dir, "Online_Cheetah_info.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data
        except:
            print("Error reading JSON.")
            pass
    return {"esn_res_dim": 300, "esn_spectral_radius": 0.5, "esn_leak_rate": 0.3, "max_episode_steps": 1000}

def run_cartpole_episode(method, wind_bias, render=False, save_video=False, video_dir="videos", video_prefix="catpole"):
    render_mode = "rgb_array" if save_video else None
    
    def make_test_env():
        env = ShiftCartPoleRender(is_test_mode=True, wind_bias=wind_bias, render_mode=render_mode)
        
        info = get_esn_res_dim(LOG_DIR)
        env = TimeLimit(env, max_episode_steps=info.get("max_episode_steps", 1000))
        if save_video:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_dir,
                name_prefix=f"{video_prefix}_{method}_idx{wind_bias}",
                episode_trigger=lambda x: True,
                disable_logger=True
            )

        if "Oracle" in method:
            env = OracleCartPoleWrapper(env, randomize_training=False)
        
        elif "PAD" in method:
            pad_weight_path = os.path.join(LOG_DIR, f"{method}_encoder.pth")
            env = ObservationNoiseWrapper(env, noise_std=0.01)
            env = PADCartPoleWrapper(
                env, 
                feature_dim=32, 
                is_test_mode=True, 
                learning_rate=1e-3, 
                load_path=pad_weight_path
            )
        elif "RMA" in method:
            adaptor_path = os.path.join(LOG_DIR, "SAC-RMA_adaptor.pth")
            env = RMAInferenceWrapper(env, adaptor_path=adaptor_path)

        elif "ESN" in method:
            input_dim = env.observation_space.shape[0] 
            esn = ESN_FORCE(input_dim=input_dim, 
                            res_dim=info.get("esn_res_dim", 300), 
                            spectral_radius=info.get("esn_spectral_radius", 0.9), 
                            leak_rate=info.get("esn_leak_rate", 0.2),
                            method=method)
            
            if "No pre-study" in method:
                print(f"[Init] {method}: Reset Weights, Train=True")
                esn.reset_weights()
                train_esn = True
                
            elif "static" in method:
                if os.path.exists("fixed_wout.npy"):
                    esn.Wout = np.load("fixed_wout.npy")
                    print(f"[Init] {method}: Loaded Wout, Train=False")
                else:
                    print("[Warning] fixed_wout.npy not found for static mode.")
                train_esn = False

            elif "Pre-study" in method:
                if os.path.exists("fixed_wout.npy"):
                    esn.Wout = np.load("fixed_wout.npy")
                    print(f"[Init] {method}: Loaded Wout, Train=True")
                else:
                    print("[Warning] fixed_wout.npy not found for Pre-study mode.")
                train_esn = True
                
            else:
                print(f"[Warning] Unknown ESN method: {method}. Defaulting to Train=True.")
                train_esn = True

            env = ESNCartPoleWrapper(env, esn, train_esn=train_esn, washout_steps=info.get("washout_steps", 0))

        return env
        
    env = DummyVecEnv([make_test_env])
    if "PAD" not in method:
        stats_path = os.path.join(LOG_DIR, f"{method}_vecnormalize.pkl")   
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
        else:
            print(f"Warning: {stats_path} not found.")
    else:
        print("Skipping VecNormalize load for PAD.")

    model = None
    model_path = os.path.join(LOG_DIR, method + ".zip")

    try:
        if "LSTM" in method:
            model = RecurrentPPO.load(model_path)
        elif "PPO" in method:
            model = PPO.load(model_path)
        elif "SAC" in method:
            model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"Model file for {method} ({model_path}) not found. Skipping.")
        env.close()
        return 0.0

    # エピソード実行
    obs = env.reset()
    done = False
    total_reward = 0.0
    lstm_states = None
    episode_start = True
    
    while not done:
        if render:
            env.render()

        if "LSTM" in method:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if done[0]:
            break
            
    env.close()
    return total_reward

def plot_robustness(results, x_values, xlabel="Friction Multiplier", 
                    legend_map=None, font_conf=None):
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
    methods = ["SAC",
               "SAC-ESN(No pre-study)", 
               "SAC-ESN(Pre-study)",
               "SAC-PAD",
               "SAC-DR",
               "SAC-RMA"]

    disturbance_values = np.arange(1, 10.1,0.5)
    n_seeds = 10
    results = {m: {"mean": [], "std": []} for m in methods}
    print(f"Starting CartPole Robustness Experiment: {len(disturbance_values)} points x {n_seeds} seeds")
    
    for method in methods:
        print(f"\nTesting {method} ", end="")
        method_means = []
        method_stds = []
        for distrubance in disturbance_values:
            rewards = []
            for _ in range(n_seeds):
                r = run_cartpole_episode(method, wind_bias=distrubance, render=False, save_video=False)
                rewards.append(r)
            
            if rewards:
                mean_r = np.mean(rewards)
                std_r = np.std(rewards)
            else:
                mean_r = 0.0
                std_r = 0.0

            method_means.append(mean_r)
            method_stds.append(std_r)

            print(f"[W{distrubance:.1f}:{int(mean_r)}]", end=" ", flush=True)

        results[method]["mean"] = method_means
        results[method]["std"] = method_stds

    plot_robustness(results, gravity_values=disturbance_values)
    csv_path = "robustness_results_cartpole.csv"
    print(f"\nSaving results to {csv_path} ...")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        header = ["Joint_Index"]
        for method in methods:
            header.append(f"{method}_Mean")
            header.append(f"{method}_Std")
        writer.writerow(header)
        
        for i, j_idx in enumerate(disturbance_values):
            row = [j_idx]
            for method in methods:
                if results[method]["mean"]:
                    row.append(results[method]["mean"][i])
                    row.append(results[method]["std"][i])
                else:
                    row.extend(["NaN", "NaN"])
            writer.writerow(row)
    