import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Data Collection')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=3000)

args = parser.parse_args()

n = args.nsteps

episodes = args.episodes

# Box2d
if args.dataset == 'BipedalWalkerHC':
    env = gym.make('BipedalWalker-v3',
                   hardcore=True,
                   render_mode='rgb_array')
else:
    env = gym.make(args.dataset + '-v4')
# mujoco
# env = gym.make('InvertedDoublePendulum-v4')
# env = gym.make('Walker2d-v4')
# env = gym.make('Hopper-v4')
# env = gym.make('Humanoid-v4')

if args.dataset == 'InvertedDoublePendulum':
    model = PPO.load('./gymmodel/InvertedDoublePendulum.zip')
else:
    model = SAC.load('./gymmodel/' + args.dataset + '.zip')

# label= 1 indicate failure
timeseries = []
labels = []
for i in range(episodes):
    t = time.time()
    done = False
    truncated = False
    total_reward = 0
    obs, _ = env.reset()
    record = []
    cnt = 0
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
        actions = torch.as_tensor(action).to(model.device).unsqueeze(0)
        record.append(torch.cat([state, actions], dim=1))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        cnt += 1
    # print(i+1, cnt, total_reward, reward, time.time()-t)
    if total_reward < 285:
        timeseries.append(torch.stack(record[-n:], dim=1))
        labels.append(1)
    else:
        index = np.random.randint(0, len(record)-n)
        timeseries.append(torch.stack(record[index:index+n], dim=1))
        labels.append(0)
    print('epoch:', i+1, 'steps:', cnt, 'reward:', total_reward, 'label:', labels[-1], 't:', time.time()-t)
    
timeseries = torch.stack(timeseries, dim=0)
labels = torch.tensor(labels, dtype=torch.long)
print('TS_shape:', list(timeseries.shape), 'Fail_num:', labels.sum().item())

X_train, X_valid = timeseries[:2000], timeseries[2000:3000]
y_train, y_valid = labels[:2000], labels[2000:3000]

X_train = X_train.permute(0, 1, 3, 2)
X_valid = X_valid.permute(0, 1, 3, 2)

if args.dataset =='Humanoid':
    X_train = torch.cat((X_train[:, :, :45, :], X_train[:, :, 376:393, :]), dim=2)
    X_valid = torch.cat((X_valid[:, :, :45, :], X_valid[:, :, 376:393, :]), dim=2)

print('X_train_AC shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
print('X_valid_AC_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

save_dir = './data/' + 'Train/' + args.dataset + 'AC' + '_' + str(n) + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(X_train, save_dir + 'X_train.pt')
torch.save(y_train, save_dir + 'y_train.pt')
torch.save(X_valid, save_dir + 'X_valid.pt')
torch.save(y_valid, save_dir + 'y_valid.pt')
    
if n == 20:
    save_dir = './data/' + 'Train/' + args.dataset + '_' + str(n) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.dataset == 'BipedalWalkerHC':
        X_train = X_train[:, :, :24, :]
        X_valid = X_valid[:, :, :24, :]
    elif args.dataset == 'InvertedDoublePendulum':
        X_train = X_train[:, :, :11, :]
        X_valid = X_valid[:, :, :11, :]
    elif args.dataset == 'Walker2d':
        X_train = X_train[:, :, :17, :]
        X_valid = X_valid[:, :, :17, :]
    elif args.dataset == 'Hopper':
        X_train = X_train[:, :, :11, :]
        X_valid = X_valid[:, :, :11, :]
    elif args.dataset == 'Humanoid':
        X_train = X_train[:, :, :45, :]
        X_valid = X_valid[:, :, :45, :]
    
    print('X_train shape:', X_train.shape, 'train_Fail_num:', y_train.sum().item())
    print('X_valid_shape:', X_valid.shape, 'valid_Fail_num:', y_valid.sum().item())

    torch.save(X_train, save_dir + 'X_train.pt')
    torch.save(X_valid, save_dir + 'X_valid.pt')
    torch.save(y_train, save_dir + 'y_train.pt')
    torch.save(y_valid, save_dir + 'y_valid.pt')
# PPO version 
# state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
# actions = torch.as_tensor(action).to(model.device).unsqueeze(0).unsqueeze(0)
# q_values_pi = model.policy.evaluate_actions(state, actions)[0]
# state = state.reshape(1, -1)
# record.append(torch.cat([state, actions, q_values_pi/100], dim=1))


# mujoco
# if cnt == 1000:
#     index = np.random.randint(0, len(record)-n)
#     timeseries.append(torch.stack(record[index:index+n], dim=1))
#     labels.append(0)
# else:
#     timeseries.append(torch.stack(record[-n:], dim=1))        
#     labels.append(1)


# BipedalWalkerHC
# if total_reward < 285:
#     timeseries.append(torch.stack(record[-n:], dim=1))
#     labels.append(1)
# else:
#     index = np.random.randint(0, len(record)-n)
#     timeseries.append(torch.stack(record[index:index+n], dim=1))
#     labels.append(0)