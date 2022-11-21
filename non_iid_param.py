import utils.fflow as flw
import os
import torch
from copy import deepcopy
import json
from tqdm import tqdm

option = flw.read_option()
option['num_gpus'] = len(option['gpu'])
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in option['gpu']])
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = '8888'
os.environ['WORLD_SIZE'] = str(3)
# set random seed
flw.setup_seed(option['seed'])
# initialize server
server = flw.initialize(option)
server_model = server.model
client_model = deepcopy(server.model)
post_server_model = deepcopy(server.model)


total_data = 0
for client in server.clients:
    total_data += len(client.train_data)
    
f = open('eval.json', 'w')
global_store = {}
for round in tqdm(range(option['num_rounds'])):    
    server_model.load_state_dict(torch.load(os.path.join(server.global_save_dir, f'Round{round}.pt')))
    post_server_model.load_state_dict(torch.load(os.path.join(server.global_save_dir, f'Round{round + 1}.pt')))
    round_local_save_dir = os.path.join(server.local_save_dir, f'Round{round + 1}')
    
    server_param_list = list(server_model.state_dict().items())
    post_server_param_list = list(post_server_model.state_dict().items())

    store = dict()
    for client in server.clients:
        propotion = []
        
        client_model.load_state_dict(torch.load(os.path.join(round_local_save_dir, f'{client.name}.pt')))
        client_param_list = list(client_model.state_dict().items())
        # print(client_param_list)
        for i in range(len(client_param_list)):
            prop = torch.sum(client_param_list[i][1] * 1e4 - server_param_list[i][1] * 1e4).item() / torch.sum(post_server_param_list[i][1] * 1e4 - server_param_list[i][1] * 1e4).item()
            propotion.append(prop * len(client.train_data))

        result = [prop / total_data for prop in propotion]
        store[f'{client.name}'] = result
    global_store[f'Round{round}'] = store
    
json.dump(global_store, f, indent=4)

f.close()