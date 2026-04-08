import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from train_cartpole import ESN_FORCE, ShiftCartPoleRender, TimeLimit, Monitor 

def noisy_rule_policy(obs, noise_prob=0.2):
    theta = obs[2]
    
    if theta > 0:
        action = 1.0
    else:
        action = -1.0
        
    if np.random.rand() < noise_prob:
        action = np.random.uniform(-1.0, 1.0)
        
    return [action] 

def generate_cartpole_fixed_weights():
    TOTAL_STEPS = 30000       
    OUTPUT_FILE = "fixed_esn_weights.npz" 
    WASHOUT_STEPS = 100       
    
    RES_DIM = 300             
    INPUT_DIM = 4
    SPECTRAL_RADIUS = 0.9
    LEAK_RATE = 0.3

    print(f"=== Generating {OUTPUT_FILE} for CartPole (Steps: {TOTAL_STEPS}) ===")

    env = ShiftCartPoleRender(is_test_mode=False)
    
    esn = ESN_FORCE(
        input_dim=INPUT_DIM, 
        res_dim=RES_DIM, 
        spectral_radius=SPECTRAL_RADIUS, 
        leak_rate=LEAK_RATE,
        method="Pre-study_Generation"
    )
    esn.reset_weights() 
    
    obs, _ = env.reset(seed=42)
    esn.forward(obs)

    rmse_history = []
    total_steps_done = 0
    episode_steps = 0

    while total_steps_done < TOTAL_STEPS:
        action = noisy_rule_policy(obs, noise_prob=0.3) 
        next_obs, _reward, terminated, truncated, _ = env.step(action)
        pred = esn.predict()
        
        if episode_steps > WASHOUT_STEPS:
            mse = np.mean((next_obs - pred) ** 2)
            rmse = np.sqrt(mse)
            rmse_history.append(rmse)
            esn.force_update(target=next_obs)

        esn.forward(next_obs)
        obs = next_obs
        total_steps_done += 1
        episode_steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
            esn.reset_state()
            esn.forward(obs)
            episode_steps = 0
            
            if len(rmse_history) > 0:
                recent_rmse = np.mean(rmse_history[-1000:])
                if total_steps_done % 5000 == 0:
                    print(f"Step {total_steps_done}/{TOTAL_STEPS} | Recent RMSE: {recent_rmse:.6f}")
    print(f"\nSaving all weights (Win, W, Wout) to {OUTPUT_FILE}...")
    np.savez(
        OUTPUT_FILE, 
        Win=esn.Win, 
        W=esn.W, 
        Wout=esn.Wout
    )
    print(f"Done! Final Wout shape: {esn.Wout.shape}")
        
    env.close()

if __name__ == "__main__":
    generate_cartpole_fixed_weights()