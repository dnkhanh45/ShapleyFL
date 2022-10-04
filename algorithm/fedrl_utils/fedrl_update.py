from pathlib import Path
from ddpg_agent.networks import *
from ddpg_agent.buffer import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import pickle # For reading buffer files

device = torch.device("cuda")

# Agent define
clients_per_round=10
value_lr=0.005
policy_lr=0.005
state_dim=clients_per_round * (4 + clients_per_round)
action_dim=clients_per_round
hidden_dim=256

gamma=0.95
soft_tau=0.01

value_net = ValueNetwork(state_dim, action_dim * 3, hidden_dim).to(device).double()
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device).double()
target_value_net = ValueNetwork(state_dim, action_dim * 3, hidden_dim).to(device).double()
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device).double()


# Buffer training
replay_buffer_size=1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


# Optimizer
value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
value_criterion = nn.MSELoss()


# Load network
def load_net(model_path):
    if Path(f"{model_path}/policy_net.pth").exists():
        policy_net.load_state_dict(torch.load(f"{model_path}/policy_net.pth"))
    if Path(f"{model_path}/value_net.pth").exists():
        value_net.load_state_dict(torch.load(f"{model_path}/value_net.pth"))
    if Path(f"{model_path}/target_policy_net.pth").exists():
        target_policy_net.load_state_dict(torch.load(f"{model_path}/target_policy_net.pth"))
    if Path(f"{model_path}/target_value_net.pth").exists():
        target_value_net.load_state_dict(torch.load(f"{model_path}/target_value_net.pth"))


# Save network
def save_net(model_path):
    if not Path(model_path).exists():
        os.system(f"mkdir -R {model_path}")
    print("Saving models...")
    torch.save(policy_net.state_dict(), f"{model_path}/policy_net.pth")
    torch.save(value_net.state_dict(), f"{model_path}/value_net.pth")
    torch.save(target_policy_net.state_dict(), f"{model_path}/target_policy_net.pth")
    torch.save(target_value_net.state_dict(), f"{model_path}/target_value_net.pth")


# Get batch form list with indexes
def get_batch_list(buffer, batch_idx):
    batch = []
    for index in batch_idx:
        batch.append(buffer[index])
    return batch


# Init priorities
def init_priority(replay_buffer, min_value=-np.inf, max_value=np.inf):
    priority_list = []
    for i in range(len(replay_buffer)):
        # print("Here i", i)
        # print("Here record", replay_buffer.buffer[i])
        state, action, reward, next_state, done = replay_buffer.buffer[i]

        with torch.no_grad():
            state = torch.DoubleTensor(state).squeeze().to(device)
            next_state = torch.DoubleTensor(next_state).squeeze().to(device)
            action = torch.DoubleTensor(action).squeeze().to(device)
            reward = torch.from_numpy(np.asanyarray(reward)).to(device)
            done = torch.from_numpy(np.asanyarray(done)).to(device)

            policy_loss = value_net(state, policy_net(state), 1)
            policy_loss = -policy_loss.mean()
            next_action = target_policy_net(next_state)
            target_value = target_value_net(next_state, next_action.detach(), 1)

            expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
            expected_value = torch.clamp(expected_value, min_value, max_value)

            value = value_net(state, action, 1).squeeze()
            value_loss = value_criterion(value, expected_value)

            temporal_diff = torch.abs(value_loss) + torch.abs(policy_loss)
            priority_list.append(temporal_diff.cpu().numpy())
    
    priority = np.abs(np.asarray(priority_list))
    priority = priority/np.sum(priority)
    return priority


# perform update
def ddpg_update(experience_folder_path, epochs=1, batch_size=16, min_value=-np.inf, max_value=np.inf):

    if not Path(experience_folder_path).exists():
        print("Experience buffer folder not found. Exit")
        return 0

    # Load all experience files
    for files_path in os.listdir(experience_folder_path):
        full_path = f"{experience_folder_path}/{files_path}"
        with open(full_path, "rb") as fp:
            buffer = pickle.load(fp)
            replay_buffer.buffer += buffer

    if len(replay_buffer) == 0:
        print("No experience found. Exit")
        return 0

    # Initiate priority for experiences
    priority = init_priority(replay_buffer)

    # Training with experience buffer
    for _ in tqdm(range(epochs)):
        for _ in range(int(len(replay_buffer)/batch_size)):
            batch_idx = np.random.choice(len(replay_buffer), size=batch_size, p=priority)
            batch = get_batch_list(replay_buffer.buffer, batch_idx)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            # state, action, reward, next_state, done = replay_buffer.sample(batch_size)

            state = torch.DoubleTensor(state).squeeze().to(device)
            next_state = torch.DoubleTensor(next_state).squeeze().to(device)
            action = torch.DoubleTensor(action).squeeze().to(device)
            reward = torch.DoubleTensor(reward).to(device)
            done = torch.DoubleTensor(np.float32(done)).to(device)

            policy_loss = value_net(state, policy_net(state), batch_size)
            policy_loss = -policy_loss.mean()
            next_action = target_policy_net(next_state)
            target_value = target_value_net(next_state, next_action.detach(), batch_size)

            expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
            expected_value = torch.clamp(expected_value, min_value, max_value)

            value = value_net(state, action, batch_size).squeeze()

            value_loss = value_criterion(value, expected_value)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

            for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    return 1


if __name__ == "__main__":
    model_path="models/"
    experience_folder_path="buffers/"
    load_net(model_path)
    updated = ddpg_update(experience_folder_path, batch_size=16, epochs=50)
    if updated:
        save_net(model_path)