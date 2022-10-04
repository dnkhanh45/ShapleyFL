import os
from pathlib import Path
from algorithm.mp_fedbase import MPBasicServer, MPBasicClient
from utils import fmodule

from algorithm.fedrl_utils.ddpg_agent.ddpg import DDPG_Agent
from datetime import datetime

import torch


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        K = self.clients_per_round
        task = self.option['task']
        
        self.ddpg_agent = DDPG_Agent(state_dim= K * K, action_dim= K * 3, hidden_dim=256, gpu_id=self.server_gpu_id, task=task)

        self.prev_reward = None
        self.warmup_length = 50
        self.last_acc = 0
        
        self.load_model(f"algorithm/fedrl_utils/baseline/fedavg/fedavg_{self.warmup_length}_{task}.pth")

        
    def load_model(self, path):
        if Path(path).exists():
            print("====================< Load baseline model for warmup >====================")
            baseline = path.split("/")[-1]
            print(f"Loading baseline {baseline}...", end=" ")
            self.model.load_state_dict(torch.load(path))
            print("done!")


    def unpack(self, packages_received_from_clients):
        
        assert self.clients_per_round == len(packages_received_from_clients), "Wrong at num clients_per_round"

        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return models, train_losses


    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients,pool)
        if not self.selected_clients:
            return

        observation = {
            "done": 0,
            "models": models
        }
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]


        priority = self.ddpg_agent.get_action(observation, prev_reward=self.prev_reward).tolist()
        self.model = self.aggregate(models, p=priority)
        fedrl_test_acc, _ = self.test(model=self.model, device=device0)
        self.prev_reward = 100 * (fedrl_test_acc - self.last_acc)
        self.last_acc = fedrl_test_acc
        
        models.clear()
        return
    
    
    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
    
    
    def run(self):
        super().run()
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        task = self.option['task']
        
        if not Path(f"algorithm/fedrl_utils/models/{task}").exists():
            os.system(f"mkdir -p algorithm/fedrl_utils/models/{task}")
            
        self.ddpg_agent.save_net(f"algorithm/fedrl_utils/models/{task}")
        
        if not Path(f"algorithm/fedrl_utils/buffers/{task}").exists():
            os.system(f"mkdir -p algorithm/fedrl_utils/buffers/{task}")
            
        self.ddpg_agent.dump_buffer(f"algorithm/fedrl_utils/buffers/{task}", f"{now}.exp")
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
