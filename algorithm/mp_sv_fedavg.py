from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import os
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import metis
from bitsets import bitset
from utils import fmodule
from tqdm.auto import tqdm
import random
import pickle
import itertools
from copy import deepcopy

CPU = torch.device('cpu')


class Server(MPBasicServer):
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
        self.optimal_lambda_samples = option['optimal_lambda_samples']
        self.calculate_fl_SV = self.exact or self.const_lambda or self.optimal_lambda
        
        if self.exact:
            self.exact_dir = os.path.join('./SV_result', self.option['task'], 'exact')
            os.makedirs(self.exact_dir, exist_ok=True)
        if self.const_lambda:
            self.const_lambda_dir = os.path.join('./SV_result', self.option['task'], 'const_lambda')
            os.makedirs(self.const_lambda_dir, exist_ok=True)
        if self.optimal_lambda:
            self.optimal_lambda_dir = os.path.join('./SV_result', self.option['task'], 'optimal_lambda')
            os.makedirs(self.optimal_lambda_dir, exist_ok=True)
        
        # Variables used in round
        if self.calculate_fl_SV:
            self.rnd_models_dict = None
            self.rnd_bitset = None
            self.rnd_dict = None
            self.rnd_partitions = None
            print('Init FL SV server!!!')
            print('Device:', fmodule.device)
        
    
    def utility_function(self, client_indices_):
        if len(client_indices_) == 0:
            return 0.0
        bitset_key = self.rnd_bitset(client_indices_).bits()
        if bitset_key in self.rnd_dict.keys():
            return self.rnd_dict[bitset_key]
        models = [self.rnd_models_dict[index] for index in client_indices_]
        # weighted
        p = np.array([self.clients[index].datavol for index in client_indices_])
        # uniform
        # p = np.ones(len(selected_clients)) / len(selected_clients)
        p = p / p.sum()
        self.model = self.aggregate(models=models, p=p)
        self.model.to(fmodule.device)
        acc, loss = self.test()
        self.rnd_dict[bitset_key] = acc
        return self.rnd_dict[bitset_key]
    
    
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
                client_indices_=self.selected_clients
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
            itertools.combinations(self.selected_clients, _) for _ in range(len(self.selected_clients) + 1)
        ))
        random.shuffle(all_rnd_subsets)
        number_of_samples = min(len(all_rnd_subsets), self.optimal_lambda_samples)
        for k in range(number_of_samples):
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
        A_matrix = A_matrix / number_of_samples
        b_vector = b_vector / number_of_samples

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
        rnd_clients_bitsetkey = self.rnd_bitset(self.selected_clients).bits()
        self.rnd_dict = dict()
        self.model.to(fmodule.device)
        acc, loss = self.test()
        self.rnd_dict[rnd_clients_bitsetkey] = acc
        return


    def init_round_MID(self):
        # Build graph
        edges = list()
        print(self.selected_clients)
        for u in self.selected_clients:
            for v in self.selected_clients:
                if u >= v:
                    continue
                w = self.utility_function([u]) + self.utility_function([v]) - self.utility_function([u, v])
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
        return
    
    
    def iterate(self, round, pool):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """

        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, train_losses, names = self.communicate(self.selected_clients, pool)
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        
        # weighted
        p = [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients]
        # uniform
        # p = np.ones(len(self.selected_clients)) / len(self.selected_clients)
        self.model = self.aggregate(models, p=p)
        
        # Shapley calculate
        if self.calculate_fl_SV:
            print('Finish training!')
            self.rnd_models_dict = dict()
            for model, name in zip(models, names):
                self.rnd_models_dict[int(name.replace('Client', ''))] = model
            print('Start to calculate FL SV round {}'.format(round))
            self.init_round()
            self.init_round_MID()
            print('Finish init round!')
        if self.exact:
            print('\tExact FL SV', end=': ')
            round_SV = self.calculate_round_exact_SV()
            print(round_SV)
            with open(os.path.join(self.exact_dir, 'Round{}.npy'.format(round)), 'wb') as f:
                pickle.dump(round_SV, f)
        if self.const_lambda:
            print('\Const lambda FL SV', end=': ')
            round_SV = self.calculate_round_const_lambda_SV()
            print(round_SV)
            with open(os.path.join(self.const_lambda_dir, 'Round{}.npy'.format(round)), 'wb') as f:
                pickle.dump(round_SV, f)
        if self.optimal_lambda:
            print('\Optimal lambda FL SV', end=': ')
            round_SV = self.calculate_round_optimal_lambda_SV()
            print(round_SV)
            with open(os.path.join(self.optimal_lambda_dir, 'Round{}.npy'.format(round)), 'wb') as f:
                pickle.dump(round_SV, f)
        return
    
    
    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        names = [cp["name"] for cp in packages_received_from_clients]
        return models, train_losses, names


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def train(self, model, device, log=False):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
            device: the device to be trained on
        :return
        """
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
        return

    
    def custom_train(self, model, device, test_set, log=False):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
            device: the device to be trained on
        :return
        """
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in tqdm(range(self.epochs), desc='Train:'):
        # for iter in range(self.epochs):
        #     train_loss = 0.0
        #     num_batches = 0
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
            #     train_loss += loss
            #     num_batches += 1
            # train_loss = train_loss / num_batches
            # print('{}: {}'.format(iter, train_loss))
        test_loss = 0.0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(test_set, batch_size=64)
        model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data,device)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
            eval_metric = 1.0 * eval_metric / len(test_set)
            test_loss = 1.0 * test_loss / len(test_set)
        return eval_metric


    def test(self, model, dataflag='valid', device='cpu'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model = model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data,device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric = 1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss


    def reply(self, svr_pkg, device):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)
        cpkg = self.pack(model, loss)
        cpkg['name'] = self.name
        return cpkg