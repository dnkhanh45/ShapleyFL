import utils.fflow as flw
import torch
import os
import importlib
import utils.fmodule
from itertools import combinations
from copy import deepcopy
from torch.utils.data import ConcatDataset
from bitsets import bitset

def main():
    option = flw.read_option()
    option['num_gpus'] = len(option['gpu'])
    print(str(option['gpu'][0]))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(option['gpu'][0])

    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
    train_datas, valid_datas, test_data, client_names = task_reader.read_data()

    path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(path), 'Client')
    client = Client(option, name = client_names[0], train_data = train_datas[0], valid_data = valid_datas[0])
    model = utils.fmodule.Model()

    save_chkpts = os.path.join('./central_chkpts', option['task'])
    os.makedirs(save_chkpts, exist_ok=True)
        
    all_clients_indices = tuple(range(len(client_names)))
    BITSET = bitset('clients_indices_bitset', all_clients_indices)
    start = option['start']
    end = option['end']
    if end == -1:
        end = pow(2, len(client_names)) - 1
    print('Start: {} - End: {}'.format(start, end))
    i = 0
    for subset_length in range(1, len(client_names) + 1):
        for subset_clients_indices in combinations(all_clients_indices, subset_length):
            if (i < start) or (i >= end):
                i += 1
                continue
            i += 1
            print('Subset:', list(subset_clients_indices))
            chkpts_filename = '{}.pt'.format(BITSET(subset_clients_indices).bits())
            print('Checkpoints filename: {}'.format(chkpts_filename))
            if os.path.exists(os.path.join(save_chkpts, chkpts_filename)):
                continue
            client.train_data = ConcatDataset([train_datas[index] for index in subset_clients_indices])
            client.valid_data = ConcatDataset([valid_datas[index] for index in subset_clients_indices])
            client.train(model, utils.fmodule.device)
            torch.save(
                model.state_dict(),
                os.path.join(save_chkpts, chkpts_filename)
            )
    return
            
if __name__ == '__main__':
    main()
