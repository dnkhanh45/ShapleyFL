from .fedbase import BasicServer, BasicClient
import torch
import os
import numpy as np
# import networkx as nx
# import metis
from bitsets import bitset
from utils import fmodule
from tqdm.auto import tqdm
import random
import pickle
import itertools
import utils.system_simulator as ss
from utils import fmodule
import collections
import copy
import json
import wandb
import utils.fflow as flw
import time

class Server(BasicServer):
    def __init__(
        self,
        option,
        model,
        clients,
        test_data=None
    ):
        super(Server, self).__init__(option, model, clients, test_data)
        
        self.num_partitions = option['num_partitions']
        self.exact = option['exact']
        self.const_lambda = option['const_lambda']
        self.optimal_lambda = option['optimal_lambda']
        self.optimal_lambda_samples = min(pow(2, self.clients_per_round) - 1, option['optimal_lambda_samples'])
        self.calculate_fl_SV = self.exact or self.const_lambda or self.optimal_lambda
        self.sv_const_logs = []
        self.sv_exact_logs = []
        self.sv_opt_logs = []
        
        if self.exact:
            self.exact_dir = os.path.join('./SV_result', self.option['task'], 'exact-{}'.format(self.clients_per_round))
            os.makedirs(self.exact_dir, exist_ok=True)
            self.acc_dir = os.path.join('./SV_result', self.option['task'], 'accuracy')
            os.makedirs(self.acc_dir, exist_ok=True)
            self.loss_dir = os.path.join('./SV_result', self.option['task'], 'loss')
            os.makedirs(self.loss_dir, exist_ok=True)
        # if self.const_lambda:
        #     self.const_lambda_dir = os.path.join('./SV_result', self.option['task'], 'const_lambda-{}'.format(self.clients_per_round))
        #     os.makedirs(self.const_lambda_dir, exist_ok=True)
        # if self.optimal_lambda:
        #     self.optimal_lambda_dir = os.path.join('./SV_result', self.option['task'], 'optimal_lambda-{}-{}'.format(self.clients_per_round, self.optimal_lambda_samples))
            # os.makedirs(self.optimal_lambda_dir, exist_ok=True)
        
        # Variables used in round
        if self.calculate_fl_SV:
            self.previous_rnd_acc = None
            self.rnd_models_dict = None
            self.rnd_bitset = None
            self.previous_rnd_acc_dict = None
            self.rnd_acc_dict = None
            self.rnd_partitions = None
            self.rnd_loss_dict = None
            self.calculate_SV_time = 0.0
        

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        start = time.time()
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval):
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        end = time.time()
        # save time
        flw.logger.add_time(total=(end - start), calculate_SV=self.calculate_SV_time)
        # save results as .json file
        log_filepath = flw.logger.save_output_as_json()
        wandb.save(log_filepath)
        return
    
    
    def utility_function(self, client_indices_):
        if len(client_indices_) == 0:
            # if self.previous_rnd_acc:
            #     return self.previous_rnd_acc
            return 0.0
        bitset_key = self.rnd_bitset(client_indices_).bits()
        if bitset_key in self.rnd_acc_dict.keys():
            # if self.previous_rnd_acc_dict:
            #     return self.rnd_acc_dict[bitset_key] - self.previous_rnd_acc_dict[bitset_key]
            return self.rnd_acc_dict[bitset_key]
        models = [self.rnd_models_dict[index] for index in client_indices_]
        # New version:
        p = np.array([self.local_data_vols[cid] for cid in client_indices_])
        p = p / p.sum()
        self.model = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        result = self.test()
        acc = result['accuracy']
        self.rnd_acc_dict[bitset_key] = acc
        loss = result['loss']
        self.rnd_loss_dict[bitset_key] = loss
        # if self.previous_rnd_acc_dict:
        #     return self.rnd_acc_dict[bitset_key] - self.previous_rnd_acc_dict[bitset_key]
        return self.rnd_acc_dict[bitset_key]
    
    
    def shapley_value(self, client_index_, client_indices_):
        if client_index_ not in client_indices_:
            return 0.0
        
        result = 0.0
        rest_client_indexes = [index for index in client_indices_ if index != client_index_]
        num_rest = len(rest_client_indexes)
        for i in range(0, num_rest + 1):
            a_i = 0.0
            count_i = 0
            for subset in itertools.combinations(rest_client_indexes, i):
                a_i += self.utility_function(set(subset).union({client_index_})) - self.utility_function(subset)
                count_i += 1
            a_i = a_i / count_i
            result += a_i
        result = result / len(client_indices_)
        return result


    def calculate_round_exact_SV(self):
        round_SV = np.zeros(self.num_clients)
        for client_index in range(self.num_clients):
            round_SV[client_index] = self.shapley_value(
                client_index_=client_index,
                client_indices_=self.received_clients
            )
        return round_SV

    
    def calculate_round_const_lambda_SV(self):
        round_SV = np.zeros(self.num_clients)
        for m in range(self.num_partitions):
            for client_index in range(self.num_clients):
                if client_index in self.rnd_partitions[m]:
                    round_SV[client_index] += self.shapley_value(
                        client_index_=client_index,
                        client_indices_=self.rnd_partitions[m]
                    )
        return round_SV


    def sub_utility_function(self, partition_index_, client_indices_):
        partition = self.rnd_partitions[partition_index_]
        intersection = list(set(partition).intersection(set(client_indices_)))
        return self.utility_function(intersection)

    
    def calculate_round_optimal_lambda_SV(self):
        # Calculate A_matrix and b_vector
        A_matrix = np.zeros((self.num_partitions, self.num_partitions))
        b_vector = np.zeros(self.num_partitions)
        all_rnd_subsets = list(itertools.chain.from_iterable(
            itertools.combinations(self.received_clients, _) for _ in range(1, len(self.received_clients) + 1)
        ))
        random.shuffle(all_rnd_subsets)
        print(len(all_rnd_subsets), self.optimal_lambda_samples)
        for k in range(self.optimal_lambda_samples):
            subset = all_rnd_subsets[k]
            for i in range(self.num_partitions):
                b_vector[i] += self.sub_utility_function(
                    partition_index_=i,
                    client_indices_=subset
                ) * self.utility_function(subset)
                for j in range(i, self.num_partitions):
                    A_matrix[i, j] += self.sub_utility_function(
                        partition_index_=i,
                        client_indices_=subset
                    ) * self.sub_utility_function(
                        partition_index_=j,
                        client_indices_=subset
                    )
                    A_matrix[j, i] = A_matrix[i, j]
        A_matrix = A_matrix / self.optimal_lambda_samples
        b_vector = b_vector / self.optimal_lambda_samples

        # Calculate optimal lambda
        optimal_lambda = np.linalg.inv(A_matrix) @ b_vector

        # Calculate round SV
        round_SV = np.zeros(self.num_clients)
        for m in range(self.num_partitions):
            for client_index in range(self.num_clients):
                if client_index in self.rnd_partitions[m]:
                    round_SV[client_index] += self.shapley_value(
                        client_index_=client_index,
                        client_indices_=self.rnd_partitions[m]
                    ) * optimal_lambda[m]
        return round_SV
        

    def init_round(self):
        # Define variables used in round
        self.rnd_bitset = bitset('round_bitset', tuple(range(self.num_clients)))
        self.previous_rnd_acc_dict = copy.deepcopy(self.rnd_acc_dict)
        self.rnd_acc_dict = dict()
        self.rnd_loss_dict = dict()
        return


    # def init_round_MID(self):
    #     # Build graph
    #     edges = list()
    #     for u in self.received_clients:
    #         for v in self.received_clients:
    #             if u >= v:
    #                 continue
    #             w = self.utility_function([u]) + self.utility_function([v]) - self.utility_function([u, v])
    #             w *= len(self.test_data)
    #             w = abs(int(np.round(w)))
    #             edges.append((u, v, w))
    #     rnd_graph = nx.Graph()
    #     rnd_graph.add_weighted_edges_from(edges)
    #     rnd_graph.graph['edge_weight_attr'] = 'weight'
    #     rnd_all_nodes = np.array(rnd_graph.nodes)

    #     # Partition graph
    #     self.rnd_partitions = list()
    #     cutcost, partitions = metis.part_graph(rnd_graph, nparts=self.num_partitions, recursive=True)
    #     for partition_index in np.unique(partitions):
    #         nodes_indexes = np.where(partitions == partition_index)[0]
    #         self.rnd_partitions.append(rnd_all_nodes[nodes_indexes])
    #     return
    
    @ss.time_step
    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        clients_reply = self.communicate(self.selected_clients)
        models = clients_reply['model']
        names = clients_reply['name']
        print("Round clients:", self.received_clients)
        if self.calculate_fl_SV:
            start = time.time()
            print('Finish training!')
            self.rnd_models_dict = dict()
            for model, name in zip(models, names):
                self.rnd_models_dict[int(name.replace('Client', ''))] = model
                
            print('Start to calculate FL SV round {}'.format(self.current_round))
            self.init_round()
            if self.const_lambda or self.optimal_lambda:
                self.init_round_MID()
            end = time.time()
            self.calculate_SV_time += (end - start)
            print('Finish init round!')
        if self.exact:
            start = time.time()
            print('Exact FL SV', end=': ')
            round_SV = self.calculate_round_exact_SV()
            end = time.time()
            self.calculate_SV_time += (end - start)
            print(round_SV)
            # with open(os.path.join(self.exact_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
            #     pickle.dump(round_SV, f)
            with open(os.path.join(self.acc_dir, 'Round{}.json'.format(self.current_round)), 'w') as f:
                json.dump(self.rnd_acc_dict, f)
            wandb.save(os.path.join(self.acc_dir, 'Round{}.json'.format(self.current_round)))
            with open(os.path.join(self.loss_dir, 'Round{}.json'.format(self.current_round)), 'w') as f:
                json.dump(self.rnd_loss_dict, f)
            wandb.save(os.path.join(self.loss_dir, 'Round{}.json'.format(self.current_round)))
        # if self.const_lambda:
        #     print('Const lambda FL SV', end=': ')
        #     round_SV = self.calculate_round_const_lambda_SV()
        #     print(round_SV)
        #     # with open(os.path.join(self.const_lambda_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
        #     #     pickle.dump(round_SV, f)
        # if self.optimal_lambda:
        #     print('Optimal lambda FL SV', end=': ')
        #     round_SV = self.calculate_round_optimal_lambda_SV()
        #     print(round_SV)
        #     # with open(os.path.join(self.optimal_lambda_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
        #     #     pickle.dump(round_SV, f)
        # aggregate
        self.model = self.aggregate(models)
        self.previous_rnd_acc = self.test()['accuracy']
        return


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        cpkg['name'] = self.name
        return cpkg