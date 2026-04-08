import gymnasium as gym
import numpy as np
import torch
import os
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
# 既存のファイルからインポート
from train_cartpole import ShiftCartPoleRender, PADCartPoleWrapper, ObservationNoiseWrapper

def simple_rule_policy(obs):
    theta = obs[2]
    if theta > 0:
        return 1.0
    else:
        return -1.0

def pretrain_pad_encoder(steps=50000, save_path="pretrained_models/cartpole_pad_encoder.pth"):
    print(f"=== Pre-training PAD Encoder (IDM + Reconstruction) ===")
    
    def make_env():
        env = ShiftCartPoleRender(is_test_mode=False) 
        env = ObservationNoiseWrapper(env, noise_std=0.01)
        env = PADCartPoleWrapper(env, feature_dim=32, learning_rate=0.0) 
        return env

    env = DummyVecEnv([make_env])
    obs = env.reset()
    current_env = env.envs[0]
    while not hasattr(current_env, 'pad_net'):
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        else:
            break
    
    pad_net = current_env.pad_net
    device = current_env.device
    optimizer = torch.optim.Adam(pad_net.parameters(), lr=1e-3)

    for i in range(steps):
        if np.random.rand() < 0.5:
            raw_obs = env.envs[0].unwrapped.state 
            if raw_obs is not None:
                act_scalar = simple_rule_policy(raw_obs)
                action = [np.array([act_scalar], dtype=np.float32)]
            else:
                action = [env.action_space.sample()]
        else:
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
        
        if (i+1) % 5000 == 0:
            print(f"Step {i+1}/{steps} | Loss: {total_loss.item():.4f} (IDM: {loss_idm.item():.4f} + Rec: {loss_rec.item():.4f})")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(pad_net.state_dict(), save_path)
    print(f"Saved pretrained encoder to {save_path}")
    env.close()

if __name__ == "__main__":
    pretrain_pad_encoder()