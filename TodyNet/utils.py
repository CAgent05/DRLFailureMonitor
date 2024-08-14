import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from src.net import GNNStack


def prepare_agent(env_name, input_tag=False):
    if env_name == 'BipedalWalkerHC':
        env = gym.make('BipedalWalker-v3',
                       hardcore=True,)
        model = SAC.load('./gymmodel/BipedalWalkerHC.zip')
        if input_tag:
            num_nodes = 28
        else:
            num_nodes = 24
        
        alg_tag = 'SAC'

    elif env_name == 'Walker2d':
        env = gym.make('Walker2d-v4')
        model = SAC.load('./gymmodel/Walker2d.zip')
        if input_tag:
            num_nodes = 23
        else:
            num_nodes = 17
        
        alg_tag = 'SAC'
    elif env_name == 'InvertedDoublePendulum':
        env = gym.make('InvertedDoublePendulum-v4')
        model = PPO.load('./gymmodel/InvertedDoublePendulum.zip')
        if input_tag:
            num_nodes = 12
        else:
            num_nodes = 11
        
        alg_tag = 'PPO'
    
    elif env_name == 'Hopper':
        env = gym.make('Hopper-v4')
        model = SAC.load('./gymmodel/Hopper.zip')
        if input_tag:
            num_nodes = 14
        else:
            num_nodes = 11
    
        alg_tag = 'SAC'
    
    elif env_name == 'Humanoid':
        env = gym.make('Humanoid-v4')
        model = SAC.load('./gymmodel/Humanoid.zip')
        if input_tag:
            num_nodes = 62
        else:
            num_nodes = 45
        
        alg_tag = 'SAC'
        
    return env, model, num_nodes, alg_tag

def transform_input(obs, action, model, record, alg_tag):
    # SAC version
    state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
    actions = torch.as_tensor(action).to(model.device).unsqueeze(0)
    record.append(torch.cat([state[:,:45], actions], dim=1))

    return record

