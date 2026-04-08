import gymnasium as gym
import numpy as np
import os
import csv
import json
import time
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from train_cartpole import (
        ESN_FORCE, ShiftCartPoleRender, 
        ESNCartPoleWrapper, DRCartPoleWrapper, 
        OracleCartPoleWrapper, PADCartPoleWrapper, 
        ObservationNoiseWrapper, RMAInferenceWrapper,
        select_log_directory
    )
except ImportError:
    print("Error: 'train_cartpole.py' not found or classes are missing.")
    exit()

def get_esn_info(log_dir):
    json_paths = [
        os.path.join(log_dir, "SAC-ESN(Pre-study)_info.json"),
        os.path.join(log_dir, "SAC-ESN(No pre-study)_info.json") 
    ]
    
    for p in json_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except: pass
            
    return {"esn_res_dim": 300, "esn_spectral_radius": 0.9, "esn_leak_rate": 0.3}

def measure_inference_speed(method, log_dir, n_steps=2000, n_trials=5):
    info = get_esn_info(log_dir)
    
    def make_test_env():
        env = ShiftCartPoleRender(is_test_mode=True, wind_bias=0.0, render_mode=None)
        env = TimeLimit(env, max_episode_steps=n_steps + 100)

        if "Oracle" in method:
            env = OracleCartPoleWrapper(env, randomize_training=False)
            
        elif "RMA" in method:
            adaptor_path = os.path.join(log_dir, "SAC-RMA_adaptor.pth")
            if os.path.exists(adaptor_path):
                env = RMAInferenceWrapper(env, adaptor_path=adaptor_path)
            else:
                print(f"Warning: RMA adaptor not found at {adaptor_path}")

        elif "PAD" in method:
            pad_weight_path = os.path.join(log_dir, f"{method}_encoder.pth")
            if os.path.exists(pad_weight_path):
                env = ObservationNoiseWrapper(env, noise_std=0.01)
                env = PADCartPoleWrapper(
                    env, feature_dim=32, is_test_mode=True, 
                    learning_rate=1e-3, load_path=pad_weight_path
                )
            else:
                print(f"Warning: PAD encoder not found at {pad_weight_path}")

        elif "ESN" in method:
            input_dim = env.observation_space.shape[0]
            esn = ESN_FORCE(input_dim=input_dim, 
                            res_dim=info.get("esn_res_dim", 300), 
                            spectral_radius=info.get("esn_spectral_radius", 0.9), 
                            leak_rate=info.get("esn_leak_rate", 0.2),
                            method=method)
            
            train_esn = True 
            
            if "No pre-study" in method:
                esn.reset_weights()
                train_esn = True 
                
            elif "static" in method:
                if os.path.exists("fixed_wout.npy"):
                    esn.Wout = np.load("fixed_wout.npy")
                train_esn = False
                
            elif "Pre-study" in method:
                if os.path.exists("fixed_wout.npy"):
                    esn.Wout = np.load("fixed_wout.npy")
                train_esn = True
            
            env = ESNCartPoleWrapper(env, esn, train_esn=train_esn, washout_steps=0)
        
        return env

    env = DummyVecEnv([make_test_env])
    if "PAD" not in method:
        stats_path = os.path.join(log_dir, f"{method}_vecnormalize.pkl")
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False 
            env.norm_reward = False

    model_path = os.path.join(log_dir, method + ".zip")
    if not os.path.exists(model_path):
        print(f"  [Skipping] Model not found: {model_path}")
        env.close()
        return None

    if "LSTM" in method:
        model = RecurrentPPO.load(model_path)
    elif "SAC" in method:
        model = SAC.load(model_path)
    elif "PPO" in method:
        model = PPO.load(model_path)
    else:
        print(f"Unknown model type for {method}")
        return None

    print(f"  Running warmup for {method}...", end="", flush=True)
    obs = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    for _ in range(100):
        if "LSTM" in method:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    print(" Done.")

    times_per_step = []

    print(f"  Measuring ({n_trials} trials x {n_steps} steps)...")
    for t in range(n_trials):
        obs = env.reset()
        if "LSTM" in method:
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
        start_time = time.perf_counter() 
        
        for _ in range(n_steps):
            if "LSTM" in method:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            
            if dones[0]:
                obs = env.reset()
                if "LSTM" in method:
                    lstm_states = None
                    episode_start = np.ones((1,), dtype=bool)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        avg_time_ms = (elapsed / n_steps) * 1000
        times_per_step.append(avg_time_ms)
        print(f"    Trial {t+1}: {avg_time_ms:.4f} ms/step")

    env.close()
    
    mean_time = np.mean(times_per_step)
    std_time = np.std(times_per_step)
    sps = 1000.0 / mean_time 

    return mean_time, std_time, sps

import matplotlib.pyplot as plt

def plot_speed_comparison(results):
    styles = {
        "SAC":                   {"c": "#1f77b4", "m": "o"}, 
        "SAC-DR":                {"c": "#2ca02c", "m": "H"}, 
        "SAC-RMA":               {"c": "#d62728", "m": "s"}, 
        "SAC-ESN(Pre-study)":    {"c": "#7f7f7f", "m": "*"}, 
        "SAC-PAD":            {"c": "#17becf", "m": "v"}, 
        "SAC-ESN(No pre-study)": {"c": "#e377c2", "m": "^"}, 
    }

    legend_map = {
        "SAC":                   "SAC", 
        "SAC-DR":                "+DR", 
        "SAC-RMA":               "+RMA", 
        "SAC-ESN(Pre-study)":    "+ESN-OA-PT", 
        "SAC-ESN(No pre-study)": "+ESN-OA", 
        "SAC-PAD":            "+PAD",
        "SAC-DR":                "+DR"
    }
    
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]
    
    colors = []
    display_labels = []
    
    for m in methods:
        key = None
        
        if "Oracle" in m:
            key = "SAC-Oracle"
        elif "No pre-study" in m:
            key = "SAC-ESN(No pre-study)"
        elif "Pre-study" in m :
            key = "SAC-ESN(Pre-study)"
        elif "PAD" in m:
            key = "SAC-PAD"
        elif "RMA" in m:
            key = "SAC-RMA"
        elif "DR" in m:
            key = "SAC-DR"
        elif "SAC" in m:
            key = "SAC"
            
        if key and key in styles:
            colors.append(styles[key]["c"])
            display_labels.append(legend_map[key])
        else:
            colors.append("gray")
            display_labels.append(m)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Inference Time (ms)", fontsize=21)
    ax1.grid(axis='y', alpha=0.3)
    
    ax1.set_ylim(0, 0.8)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax1.tick_params(axis='both', which='major', labelsize=21)
    ax1.set_xticklabels(display_labels, fontsize=21) 
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=21, fontweight='bold')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    print("\nSaved graph to Fig_CartPole_Speed.png")

if __name__ == "__main__":
    LOG_DIR = select_log_directory()
    
    # 比較対象メソッド
    target_methods = [
        "SAC",
        "SAC-DR",          
        "SAC-ESN(No pre-study)",
        "SAC-ESN(Pre-study)",
        "SAC-RMA",
        "SAC-PAD"
    ]

    target_methods = [
        "SAC-ESN(No pre-study)",
        "SAC-ESN(Pre-study)"
    ]
    
    results = {}
    csv_data = []
    print("\n=== Starting CartPole Inference Speed Test ===")
    
    for method in target_methods:
        print(f"\nEvaluating: {method}")
        res = measure_inference_speed(method, LOG_DIR, n_steps=3000, n_trials=5)
        
        if res is not None:
            mean_t, std_t, sps = res
            results[method] = {"mean": mean_t, "std": std_t, "sps": sps}
            csv_data.append([method, mean_t, std_t, sps])
            print(f"  Result: {mean_t:.4f} ms/step (+-{std_t:.4f}), {sps:.1f} SPS")
        else:
            print("  Result: Failed")
    if results:
        with open("cartpole_speed_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Method", "Mean_Time_ms", "Std_Dev_ms", "Steps_Per_Second"])
            writer.writerows(csv_data)
        print("\nSaved data to cartpole_speed_results.csv")
        
        plot_speed_comparison(results)
    else:
        print("No results to plot.")