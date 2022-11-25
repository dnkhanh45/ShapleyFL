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

class ShapleyValueServer(MPBasicServer):
    def __init__(self, server, num_partitions=2):
        super(ShapleyValueServer, self).__init__(
            option=server.option,
            model=server.model,
            clients=server.clients,
            test_data=server.test_data
        )
        self.global_save_dir = server.global_save_dir
        self.local_save_dir = server.local_save_dir
        self.num_partitions = num_partitions

        # Variables used in round
        self.rnd_clients = None
        self.rnd_bitset = None
        self.rnd_clients_bitsetkey = None
        self.rnd_dict = None
        self.rnd_partitions = None
        print(fmodule.device)


    def utility_function(self, client_indexes_, round):
        if len(client_indexes_) == 0:
            return 0.0
        bitset_key = self.rnd_bitset(client_indexes_).bits()
        if bitset_key in self.rnd_dict.keys():
            return self.rnd_dict[bitset_key]
        selected_clients = [self.clients[index] for index in client_indexes_]
        models = [client.model for client in selected_clients]
        # weighted
        p = np.array([client.datavol for client in selected_clients])
        # uniform
        # p = np.ones(len(selected_clients)) / len(selected_clients)
        p = p / p.sum()
        self.model = self.aggregate(models=models, p=p)
        self.model.to(fmodule.device)
        acc, loss = self.test()
        # self.rnd_dict[bitset_key] = acc / pow(round, 2)
        self.rnd_dict[bitset_key] = acc
        return self.rnd_dict[bitset_key]

    
    def shapley_value(self, client_index_, client_indexes_, round):
        if client_index_ not in client_indexes_:
            return 0.0
        
        result = 0.0
        rest_client_indexes = [index for index in client_indexes_ if index != client_index_]
        num_rest = len(rest_client_indexes)
        for i in range(0, num_rest + 1):
            a_i = 0.0
            count_i = 0
            for subset in itertools.combinations(rest_client_indexes, i):
                a_i += self.utility_function(set(subset).union({client_index_}), round) - self.utility_function(subset, round)
                count_i += 1
            a_i = a_i / count_i
            result += a_i
        result = result / len(client_indexes_)
        return result
        

    def init_round(self, round_):
        # List clients participated in current round
        self.rnd_clients = [
            int(file.replace('Client', '').replace('.pt', ''))
            for file in os.listdir(os.path.join(
                self.local_save_dir, 'Round{}'.format(round_)
            ))
        ]
        self.rnd_clients.sort()

        # Load checkpoints
        for index in self.rnd_clients:
            client = self.clients[index]
            if client.model is None:
                client.model = fmodule.Model()
            client.model.load_state_dict(torch.load(
                os.path.join(self.local_save_dir, 'Round{}/{}.pt'.format(round_, client.name)),
                map_location=CPU
            ))

        # Define variables used in round
        self.rnd_bitset = bitset('round_bitset', tuple(self.rnd_clients))
        # self.rnd_clients_bitsetkey = self.rnd_bitset(self.rnd_clients).bits()
        self.rnd_dict = dict()
        self.model.load_state_dict(torch.load(
            os.path.join(self.global_save_dir, 'Round{}.pt'.format(round_)),
            map_location=fmodule.device
        ))
        acc, loss = self.test()
        # self.rnd_dict[self.rnd_clients_bitsetkey] = acc
        print('Round {}: {}'.format(round_, acc), end=' ')
        return


    def calculate_round_exact_SV(self, round):
        round_SV = np.zeros(self.num_clients)
        for client_index in range(self.num_clients):
            round_SV[client_index] = self.shapley_value(
                client_index_=client_index,
                client_indexes_=self.rnd_clients,
                round=round
            )
        return round_SV


    def init_round_MID(self, round_):
        # Build graph
        edges = list()
        for u in self.rnd_clients:
            for v in self.rnd_clients:
                if u >= v:
                    continue
                w = self.utility_function([u], round_) + self.utility_function([v], round_) - self.utility_function([u, v], round_)
                w *= len(self.test_data)
                w = int(np.round(w))
                edges.append((u, v, w))
        rnd_graph = nx.Graph()
        rnd_graph.add_weighted_edges_from(edges)
        rnd_graph.graph['edge_weight_attr'] = 'weight'
        rnd_all_nodes = np.array(rnd_graph.nodes)

        # Partition graph
        self.rnd_partitions = list()
        cutcost, partitions = metis.part_graph(rnd_graph, nparts=self.num_partitions, recursive=False)
        for partition_index in np.unique(partitions):
            nodes_indexes = np.where(partitions == partition_index)[0]
            self.rnd_partitions.append(rnd_all_nodes[nodes_indexes])
        print(cutcost / 960, end=' ')
        return

    
    def calculate_round_const_lambda_SV(self, round):
        round_SV = np.zeros(self.num_clients)
        for m in range(self.num_partitions):
            for client_index in range(self.num_clients):
                if client_index in self.rnd_partitions[m]:
                    round_SV[client_index] += self.shapley_value(
                        client_index_=client_index,
                        client_indexes_=self.rnd_partitions[m],
                        round=round
                    )
        return round_SV


    def sub_utility_function(self, partition_index_, client_indexes_, round):
        partition = self.rnd_partitions[partition_index_]
        intersection = list(set(partition).intersection(set(client_indexes_)))
        return self.utility_function(intersection, round)

    
    def calculate_round_optimal_lambda_SV(self, round, number_of_samples_):
        # Calculate A_matrix and b_vector
        A_matrix = np.zeros((self.num_partitions, self.num_partitions))
        b_vector = np.zeros(self.num_partitions)
        all_rnd_subsets = list(itertools.chain.from_iterable(
            itertools.combinations(self.rnd_clients, _) for _ in range(len(self.rnd_clients) + 1)
        ))
        random.shuffle(all_rnd_subsets)
        for k in range(number_of_samples_):
            subset = all_rnd_subsets[k]
            for i in range(self.num_partitions):
                b_vector[i] += self.sub_utility_function(
                    partition_index_=i,
                    client_indexes_=subset,
                    round = round
                ) * self.utility_function(subset, round)
                for j in range(i, self.num_partitions):
                    A_matrix[i, j] += self.sub_utility_function(
                        partition_index_=i,
                        client_indexes_=subset,
                        round=round
                    ) * self.sub_utility_function(
                        partition_index_=j,
                        client_indexes_=subset,
                        round=round
                    )
                    A_matrix[j, i] = A_matrix[i, j]
        A_matrix = A_matrix / number_of_samples_
        b_vector = b_vector / number_of_samples_

        # Calculate optimal lambda
        optimal_lambda = np.linalg.inv(A_matrix) @ b_vector

        # Calculate round SV
        round_SV = np.zeros(self.num_clients)
        for m in range(self.num_partitions):
            for client_index in range(self.num_clients):
                if client_index in self.rnd_partitions[m]:
                    round_SV[client_index] += self.shapley_value(
                        client_index_=client_index,
                        client_indexes_=self.rnd_partitions[m],
                        round=round
                    ) * optimal_lambda[m]
        return round_SV


    def calculate_FL_SV(self, type_, number_of_samples_=30):
        # Check constraints
        assert type_ in ["exact", "const_lambda", "optimal_lambda"], "Wrong type of calculating Shapley values!"
        if type_ == "exact" and self.clients_per_round > 16:
            raise ValueError("TLE!")

        SV_save_dir = os.path.join('./SV_result', self.option['task'])
        os.makedirs(SV_save_dir, exist_ok=True)
        if type_ == "optimal_lambda":
            saved_filename = '{}-{}.npy'.format(type_, number_of_samples_)
        else:
            saved_filename = '{}.npy'.format(type_)
        # Calculate Shapley values for each client
        clients_SV = list()
        for round in tqdm(range(1, self.num_rounds + 1)):
        # for round in range(1, 2):
            self.init_round(round_=round)
            # Calculate for current round
            if type_ == "exact":
                round_SV = self.calculate_round_exact_SV(round)
                with open(os.path.join(SV_save_dir, 'Round{}{}'.format(round, saved_filename)), 'wb') as f:
                    pickle.dump(round_SV, f)
            elif type_ == "const_lambda":
                self.init_round_MID(round_=round)
                round_SV = self.calculate_round_const_lambda_SV(round)
            elif type_ == "optimal_lambda":
                self.init_round_MID(round_=round)
                round_SV = self.calculate_round_optimal_lambda_SV(round, number_of_samples_)
            clients_SV.append(round_SV.tolist())
            print(round_SV.sum())
        clients_SV = np.array(clients_SV)
        if type_ in ["const_lambda", "optimal_lambda"]:
            with open(os.path.join(SV_save_dir, saved_filename), 'wb') as f:
                pickle.dump(clients_SV, f)
        return clients_SV