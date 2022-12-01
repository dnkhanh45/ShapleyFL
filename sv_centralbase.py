from algorithm.mp_fedbase import MPBasicServer
import os
import numpy as np
import torch
import networkx as nx
import metis
from bitsets import bitset
from utils import fmodule
import itertools
from tqdm.auto import tqdm
import random
import pickle

CPU = torch.device('cpu')

class CentralizedShapleyValue(MPBasicServer):
    def __init__(self, server):
        super(CentralizedShapleyValue, self).__init__(
            option=server.option,
            model=server.model,
            clients=server.clients,
            test_data=server.test_data
        )
        self.checkpoints_dir = os.path.join('./central_chkpts', self.option['task'])
        self.clients_indices = tuple(range(self.num_clients))
        self.bitset = bitset('bitset', tuple(self.clients_indices))
        self.dict = dict()
        self.save_dir = os.path.join('./SV_result', self.option['task'], 'central')
        os.makedirs(self.save_dir, exist_ok=True)
        self.i = 0


    def utility_function(self, client_indexes_):
        if len(client_indexes_) == 0:
            return 0.0
        bitset_key = self.bitset(client_indexes_).bits()
        if bitset_key in self.dict.keys():
            return self.dict[bitset_key]
        self.i += 1
        self.model.load_state_dict(torch.load(
            os.path.join(self.checkpoints_dir, '{}.pt'.format(bitset_key)),
            map_location=fmodule.device
        ))
        acc, loss = self.test()
        self.dict[bitset_key] = acc
        print(self.i, bitset_key, acc)
        return self.dict[bitset_key]

    
    def shapley_value(self, client_index_, client_indexes_):
        if client_index_ not in client_indexes_:
            return 0.0
        
        result = 0.0
        rest_client_indexes = [index for index in client_indexes_ if index != client_index_]
        num_rest = len(rest_client_indexes)
        for i in range(0, num_rest + 1):
            a_i = 0.0
            count_i = 0
            for subset in itertools.combinations(rest_client_indexes, i):
                a_i += self.utility_function(set(subset).union({client_index_})) - self.utility_function(subset)
                count_i += 1
            a_i = a_i / count_i
            result += a_i
        result = result / len(client_indexes_)
        return result


    def calculate_central_SV(self):
        clients_SV = np.zeros(self.num_clients)
        for client_index in range(self.num_clients):
            clients_SV[client_index] = self.shapley_value(
                client_index_=client_index,
                client_indexes_=self.clients_indices
            )
        with open(os.path.join(self.save_dir, 'central.npy'), 'wb') as f:
            pickle.dump(clients_SV, f)
        return clients_SV
