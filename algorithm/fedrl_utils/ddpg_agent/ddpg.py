from pathlib import Path
from algorithm.fedrl_utils.ddpg_agent.utils import *
from algorithm.fedrl_utils.ddpg_agent.networks import *
from algorithm.fedrl_utils.ddpg_agent.policy import *
from algorithm.fedrl_utils.ddpg_agent.buffer import *
import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.fedrl_utils.ddpg_agent.policy import NormalizedActions
import pickle


def transform_action(action):
    action = action.view(3,-1)
    return torch.flatten(action[1] + 1/10 * action[2])


class DDPG_Agent(nn.Module):
    def __init__(
        self,
        state_dim=3,
        action_dim=1,
        hidden_dim=256,
        value_lr=1e-3,
        policy_lr=1e-3,
        replay_buffer_size=1000000,
        max_steps=16*50,
        batch_size=4,
        log_dir="./log/epochs",
        gamma = 0.99,
        soft_tau = 2e-2,
    ):
        super(DDPG_Agent, self).__init__()
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        print("\nInit State dim", state_dim)    # K x K
        print("Init Action dim", action_dim)    # K x 3

        self.value_net = ValueNetwork(num_inputs=state_dim + action_dim, hidden_size=hidden_dim).to(self.device).double()
        self.policy_net = PolicyNetwork(num_inputs=state_dim, num_outputs=action_dim, hidden_size=hidden_dim).to(self.device).double()

        self.target_value_net = ValueNetwork(num_inputs=state_dim + action_dim, hidden_size=hidden_dim).to(self.device).double()
        self.target_policy_net = PolicyNetwork(num_inputs=state_dim, num_outputs=action_dim, hidden_size=hidden_dim).to(self.device).double()


        model_path = "../models"
        if Path(f"{model_path}/policy_net.pth").exists():
            print("Loaing policy_net...")
            self.policy_net.load_state_dict(torch.load(f"{model_path}/policy_net.pth"))
        if Path(f"{model_path}/value_net.pth").exists():
            print("Loaing value_net...")
            self.value_net.load_state_dict(torch.load(f"{model_path}/value_net.pth"))
        if Path(f"{model_path}/target_policy_net.pth").exists():
            print("Loaing target_policy_net...")
            self.target_policy_net.load_state_dict(torch.load(f"{model_path}/target_policy_net.pth"))
        if Path(f"{model_path}/target_value_net.pth").exists():
            print("Loaing target_value_net...")
            self.target_value_net.load_state_dict(torch.load(f"{model_path}/target_value_net.pth"))


        # store all the (s, a, s', r) during the transition process
        self.memory = Memory()
        # replay buffer used for main training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()
        self.step = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_dir = log_dir

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)


    def get_action(self, observation, prev_reward=None):
        done = observation['done']
        models = observation['models']

        # reach to maximum step for each episode or get the done for this iteration
        state = get_state(models)
        state = torch.DoubleTensor(state).unsqueeze(0).to(self.device)  # current state
        if prev_reward is not None:
            print("prev_reward", prev_reward)
            self.memory.update(r=prev_reward)

        action = self.policy_net.get_action(state)
        self.memory.act(state, action)

        # if self.step < self.max_steps:
        if self.memory.get_last_record() is None:
            self.step += 1
            return transform_action(action)
        
        if len(self.replay_buffer) >= self.batch_size:
            self.ddpg_update()

        s, a, r, s_next = self.memory.get_last_record()
        self.replay_buffer.push(s, a, r, s_next, done)
        self.step += 1

        return transform_action(action)


    def ddpg_update(self, min_value=-np.inf, max_value=np.inf):
    
        for _ in range(int(len(self.replay_buffer)/self.batch_size)):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            state = torch.DoubleTensor(state).squeeze().to(self.device)
            next_state = torch.DoubleTensor(next_state).squeeze().to(self.device)
            action = torch.DoubleTensor(action).squeeze().to(self.device)
            reward = torch.DoubleTensor(reward).to(self.device)
            done = torch.DoubleTensor(np.float32(done)).to(self.device)

            policy_loss = self.value_net(state, self.policy_net(state), self.batch_size)
            policy_loss = -policy_loss.mean()
            next_action = self.target_policy_net(next_state)
            target_value = self.target_value_net(next_state, next_action.detach(), self.batch_size)

            expected_value = reward + (1.0 - done) * self.gamma * target_value.squeeze()
            expected_value = torch.clamp(expected_value, min_value, max_value)

            value = self.value_net(state, action, self.batch_size).squeeze()

            value_loss = self.value_criterion(value, expected_value)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)


    def dump_buffer(self, buffer_path, run_name):
        if not Path(buffer_path).exists():
            os.system(f"mkdir -p {buffer_path}")
            
        with open(f"{buffer_path}/{run_name}.exp", "ab") as fp:
            pickle.dump(self.replay_buffer.buffer, fp)