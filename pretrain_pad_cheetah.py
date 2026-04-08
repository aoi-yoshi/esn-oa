import gymnasium as gym
import numpy as np
import torch
import os
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv

from train_halfcheetah import (
    CrippledHalfCheetah, 
    PADCheetahWrapper, 
    ObservationNoiseWrapper, 
    DRCheetahWrapper
)

def pretrain_pad_cheetah(steps=200000, save_path="pretrained_models/cheetah_pad_encoder.pth"):
    print(f"=== Pre-training PAD Encoder for HalfCheetah ({steps} steps) ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def make_env():
        env = CrippledHalfCheetah(cripple_joint=None)
        env = DRCheetahWrapper(env)
        
        env = ObservationNoiseWrapper(env, noise_std=0.01)
        env = PADCheetahWrapper(env, feature_dim=32, learning_rate=0.0) 
        return env

    env = DummyVecEnv([make_env])
    obs = env.reset()
    current_env = env.envs[0]
    while not hasattr(current_env, 'pad_net'):
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        else:
            print("Error: PADCheetahWrapper not found!")
            return
            
    pad_net = current_env.pad_net
    device = current_env.device
    optimizer = torch.optim.Adam(pad_net.parameters(), lr=1e-3)

    print("Start collecting data...")
    loss_history = []

    for i in range(steps):
        action = [env.action_space.sample()]
        obs_tensor = current_env.last_obs_tensor.detach().clone()
        action_array = np.array(action)
        action_tensor = torch.FloatTensor(action_array).to(device)
        _ = env.step(action)
        next_obs_tensor = current_env.last_obs_tensor.detach().clone()
        feat_t = pad_net.forward_encoder(obs_tensor)
        feat_next = pad_net.forward_encoder(next_obs_tensor)
        pred_action = pad_net.forward_idm(feat_t, feat_next)
        loss_idm = F.mse_loss(pred_action, action_tensor)
        pred_obs = pad_net.forward_decoder(feat_t)
        loss_rec = F.mse_loss(pred_obs, obs_tensor)
        
        total_loss = loss_idm + loss_rec
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())

        if (i+1) % 5000 == 0:
            avg_loss = np.mean(loss_history[-1000:])
            print(f"Step {i+1}/{steps} | Loss: {avg_loss:.4f} (IDM: {loss_idm.item():.4f} + Rec: {loss_rec.item():.4f})")

    torch.save(pad_net.state_dict(), save_path)
    print(f"Saved pretrained encoder to {save_path}")
    env.close()

if __name__ == "__main__":
    pretrain_pad_cheetah(steps=200000)