import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from train_halfcheetah import ESN_FORCE, CrippledHalfCheetah 

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def generate_cheetah_fixed_weights():
    TOTAL_STEPS = 100000      
    OUTPUT_FILE = "cheetah_fixed_esn_weights.npz"
    WASHOUT_STEPS = 100       
    CLIP_OBS = 10.0           

    RES_DIM = 500             
    SPECTRAL_RADIUS = 0.9
    LEAK_RATE = 0.3

    print(f"=== Generating {OUTPUT_FILE} for HalfCheetah (Steps: {TOTAL_STEPS}) ===")

    env = CrippledHalfCheetah(cripple_joint=None) 
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Observation Dim: {input_dim}, Action Dim: {action_dim}, Reservoir Dim: {RES_DIM}")

    esn = ESN_FORCE(
        input_dim=input_dim, 
        res_dim=RES_DIM,
        spectral_radius=SPECTRAL_RADIUS,
        leak_rate=LEAK_RATE,
        method="SAC-ESN(Pre-study)",
        density = 1
    )
    ou_noise = OUNoise(action_dim, sigma=0.3)

    obs, _ = env.reset(seed=42)
    esn.reset_weights()
    esn.forward(obs)
    ou_noise.reset()

    rmse_history = []
    total_steps_done = 0
    episode_steps = 0

    while total_steps_done < TOTAL_STEPS:
        action = ou_noise.noise()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, _reward, terminated, truncated, _ = env.step(action)
        next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=CLIP_OBS, neginf=-CLIP_OBS)
        next_obs = np.clip(next_obs, -CLIP_OBS, CLIP_OBS)
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
            ou_noise.reset()
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
    generate_cheetah_fixed_weights()