import utils.fflow as flw
import torch
import os
import importlib
import utils.fmodule
from itertools import combinations
from copy import deepcopy

option = flw.read_option()
option['num_gpus'] = len(option['gpu'])

bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
train_datas, valid_datas, test_data, client_names = task_reader.read_data()
index = [i for i in range(len(client_names))]
# print(train_datas)
print(index)
for train_data in train_datas:
    print(train_data.X.shape)

join_data = torch.cat((train_datas[0].X, train_datas[1].X), 0)
print(join_data.shape)

for i in range(1, 4):
    comb_list = combinations(index, i)
    print(list(comb_list))

path = '%s.%s' % ('algorithm', option['algorithm'])
Client=getattr(importlib.import_module(path), 'Client')
client = Client(option, name = client_names[0], train_data = train_datas[0], valid_data = valid_datas[0])
client.model = getattr(importlib.import_module(bmk_model_path), 'Model')
print(client.model)

train_dataset = deepcopy(train_datas[0])
valid_dataset = deepcopy(valid_datas[0])

if not os.path.exists('central_chkpts'):
    os.path.mkdir('central_chkpts')
    
count = 1

for i in range(1, len(client_names)+1):
    p = combinations(train_datas, i)
    for j in list(p):
        join_train_data_inputs = torch.cat([train_datas[k].X for k in j], 0)
        join_train_data_labels = torch.cat([train_datas[k].y for k in j], 0)
        join_valid_data_inputs = torch.cat([valid_datas[k].X for k in j], 0)
        join_valid_data_labels = torch.cat([valid_datas[k].y for k in j], 0)
        
        train_dataset.X = join_train_data_inputs
        train_dataset.y = join_train_data_labels
        valid_dataset.X = join_valid_data_inputs
        valid_dataset.y = join_valid_data_labels
        
        client.train_data = train_dataset
        client.valid_data = valid_dataset
        
        client.train()
        
        saved_path = f'central_chkpts/join_data{count}'
        torch.save(client.model.state_dict(), saved_path)
        count += 1
        

