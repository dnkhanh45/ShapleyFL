from .fedbase import BasicServer, BasicClient
import torch
import os
import numpy as np
import networkx as nx
import metis
from bitsets import bitset
import random
import pickle
import itertools
import utils.system_simulator as ss
import wandb
import utils.fflow as flw
import torch.multiprocessing as mp
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
        self.optimal_lambda_samples = option['optimal_lambda_samples']
        self.calculate_fl_SV = self.exact or self.const_lambda or self.optimal_lambda
        self.sv_const_logs = []
        self.sv_exact_logs = []
        self.sv_opt_logs = []
        
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
        
    
    def utility_function(self, client_indices_):
        if len(client_indices_) == 0:
            return 0.0
        bitset_key = self.rnd_bitset(client_indices_).bits()
        if bitset_key in self.rnd_dict.keys():
            return self.rnd_dict[bitset_key]
        models = [self.rnd_models_dict[index] for index in client_indices_]
        self.model = self.aggregate(models=models, client_indices=client_indices_)
        acc = self.test()['accuracy']
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
            itertools.combinations(self.received_clients, _) for _ in range(len(self.received_clients) + 1)
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
        rnd_clients_bitsetkey = self.rnd_bitset(self.received_clients).bits()
        self.rnd_dict = dict()
        acc = self.test()['accuracy']
        self.rnd_dict[rnd_clients_bitsetkey] = acc
        return


    def init_round_MID(self):
        # Build graph
        edges = list()
        for u in self.received_clients:
            for v in self.received_clients:
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
        cutcost, partitions = metis.part_graph(rnd_graph, nparts=self.num_partitions, recursive=True)
        for partition_index in np.unique(partitions):
            nodes_indexes = np.where(partitions == partition_index)[0]
            self.rnd_partitions.append(rnd_all_nodes[nodes_indexes])
        return
    
    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        store_path = 'checkpoint'
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        if not os.path.exists(os.path.join(store_path, self.option['task'])):
            os.mkdir(os.path.join(store_path, self.option['task']))
        store_path = f"checkpoint/{self.option['task']}"
        
        if not os.path.exists(os.path.join(store_path, 'local')):
            os.mkdir(os.path.join(store_path, 'local'))
        if not os.path.exists(os.path.join(store_path, 'global')):
            os.mkdir(os.path.join(store_path, 'global'))
        
        if not os.path.exists(os.path.join(store_path, 'global','round0')):
            os.mkdir(os.path.join(store_path, 'global','round0'))
        torch.save(self.model.state_dict(), os.path.join(store_path, 'global', 'round0/global_model.pt'))

        flw.logger.time_start('Total Time Cost')
        
        for round in range(1, self.num_rounds+1):
            global_store_path = os.path.join(store_path, 'global', f'round{round}')
            local_store_path = os.path.join(store_path, 'local', f'round{round}')
            os.makedirs(global_store_path, exist_ok=True)
            os.makedirs(local_store_path, exist_ok=True)
            
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
            self.iterate(round, global_store_path, local_store_path)
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        
        #log wandb
        if self.exact:
            exact_table = wandb.Table(data=self.sv_exact_logs, columns=[str(i + 1) for i in range(self.num_clients + 1)])
        if self.const_lambda:
            const_table = wandb.Table(data=self.sv_const_logs, columns=[str(i + 1) for i in range(self.num_clients + 1)])
        if self.optimal_lambda:
            opt_table = wandb.Table(data=self.sv_opt_logs, columns=[str(i + 1) for i in range(self.num_clients + 1)])
        for i in range(self.num_clients):
            if self.exact:
                wandb.log({f'BarChart-Exact{i}': wandb.plot.bar(exact_table, str(self.num_clients + 1), str(i + 1), title='Exact SV')})
            if self.const_lambda:
                wandb.log({f'BarChart-Const{i}': wandb.plot.bar(const_table, str(self.num_clients + 1), str(i + 1), title='Const SV')})
            if self.optimal_lambda:
                wandb.log({f'BarChart-Optimal{i}': wandb.plot.bar(opt_table, str(self.num_clients + 1), str(i + 1), title='Optimal SV')})
            
        # save results as .json file
        flw.logger.save_output_as_json()
        return
    
    # def aggregate(self, models: list, client_indices=None, *args, **kwargs):
    #     """
    #     Aggregate the locally improved models.
    #     :param
    #         models: a list of local models
    #     :return
    #         the averaged result
    #     pk = nk/n where n=self.data_vol
    #     K = |S_t|
    #     N = |S|
    #     -------------------------------------------------------------------------------------------------------------------------
    #      weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
    #     ==========================================================================================================================
    #     N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k
    #     """
    #     if len(models) == 0: return self.model
    #     if self.aggregation_option == 'weighted_scale':
    #         if client_indices:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in client_indices]
    #         else:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
    #         K = len(models)
    #         N = self.num_clients
    #         return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
    #     elif self.aggregation_option == 'uniform':
    #         return fmodule._model_average(models)
    #     elif self.aggregation_option == 'weighted_com':
    #         if client_indices:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in client_indices]
    #         else:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
    #         w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
    #         return (1.0-sum(p))*self.model + w
    #     else:
    #         if client_indices:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in client_indices]
    #         else:
    #             p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
    #         sump = sum(p)
    #         p = [pk/sump for pk in p]
    #         return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
    
    # @fmodule.with_multi_gpus
    # def test_on_clients(self, dataflag='valid'):
    #     """
    #     Validate accuracies and losses on clients' local datasets
    #     :param
    #         dataflag: choose train data or valid data to evaluate
    #     :return
    #         metrics: a dict contains the lists of each metric_value of the clients
    #     """
    #     all_metrics = collections.defaultdict(list)
        
    #     if self.num_threads > 1:
    #         sample_client = self.clients[0]
    #         with mp.Pool(processes=self.num_threads) as pool:
    #             results =  pool.starmap_async(sample_client.test, [(client.model, dataflag) for client in self.clients])
    #             results = results.get()
    #             for result in results:
    #                 for met_name, met_val in result.items():
    #                     result[met_name].append(met_val)    
    #         pool.close()
    #     else :
    #         for c in self.clients:
    #             client_metrics = c.test(self.model, dataflag)
    #             for met_name, met_val in client_metrics.items():
    #                 all_metrics[met_name].append(met_val)
    #     return all_metrics
    
    # @ss.with_dropout
    # @ss.with_clock
    # def communicate(self, selected_clients, local_store_path, asynchronous=False):
    #     """
    #     The whole simulating communication procedure with the selected clients.
    #     This part supports for simulating the client dropping out.
    #     :param
    #         selected_clients: the clients to communicate with
    #     :return
    #         :the unpacked response from clients that is created ny self.unpack()
    #     """
    #     packages_received_from_clients = []
    #     client_package_buffer = {}
    #     communicate_clients = list(set(selected_clients))
    #     for cid in communicate_clients:client_package_buffer[cid] = None
    #     if self.num_threads <= 1:
    #         # computing iteratively
    #         for client_id in communicate_clients:
    #             response_from_client_id = self.communicate_with(client_id)
    #             torch.save(response_from_client_id['model'].state_dict(), os.path.join(local_store_path, f'client{client_id}_model.pt'))
    #             packages_received_from_clients.append(response_from_client_id)
    #     else:
    #         # computing in parallel with torch.multiprocessing
    #         pool = mp.Pool(self.num_threads)
    #         for client_id in communicate_clients:
    #             self.clients[client_id].update_device(next(utils.fmodule.dev_manager))
                
    #         packages_received_from_clients.append(pool.map_async(self.communicate_with, [int(client_id) for client_id in communicate_clients]))
    #         pool.close()
    #         pool.join()
    #         packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
    #     for i,cid in enumerate(communicate_clients): client_package_buffer[cid] = packages_received_from_clients[i]
    #     packages_received_from_clients = [client_package_buffer[cid] for cid in selected_clients if client_package_buffer[cid]]
    #     self.received_clients = selected_clients
    #     return self.unpack(packages_received_from_clients)

    # @ss.with_latency
    # def communicate_with(self, client_id):
    #     """
    #     Pack the information that is needed for client_id to improve the global model
    #     :param
    #         client_id: the id of the client to communicate with
    #     :return
    #         client_package: the reply from the client and will be 'None' if losing connection
    #     """
    #     # package the necessary information
    #     svr_pkg = self.pack(client_id)
    #     # listen for the client's response
    #     return self.clients[client_id].reply(svr_pkg)
    
    @ss.time_step
    def iterate(self, round, global_store_path, local_store_path):
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
        # aggregate
        self.model = self.aggregate(models)
        # torch.save(self.model.state_dict(), os.path.join(global_store_path, 'global_model.pt'))
        #testing phase
        # valid_accs = np.array(self.test_on_clients('valid')['accuracy'])
        # test_acc = self.test()
        # wandb.log({'Train accuracy': valid_accs.sum() / len(valid_accs), 'Test accuracy': test_acc['accuracy']})
        # calculate Shapley values
        if self.calculate_fl_SV:
            print('Finish training!')
            self.rnd_models_dict = dict()
            for model, name in zip(models, names):
                self.rnd_models_dict[int(name.replace('Client', ''))] = model
                # torch.save(model.state_dict(), os.path.join(local_store_path, 'client{}_model.pt'.format(int(name.replace('Client', '')))))
            print('Start to calculate FL SV round {}'.format(self.current_round))
            self.init_round()
            self.init_round_MID()
            print('Finish init round!')
        # return
        if self.exact:
            print('Exact FL SV', end=': ')
            round_SV = self.calculate_round_exact_SV()
            print(round_SV)
            round_SV = round_SV.tolist()
            round_SV.append(round)
            self.sv_exact_logs.append(round_SV)
            with open(os.path.join(self.exact_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
                pickle.dump(round_SV, f)
        if self.const_lambda:
            print('Const lambda FL SV', end=': ')
            round_SV = self.calculate_round_const_lambda_SV()
            print(round_SV)
            round_SV = round_SV.tolist()
            round_SV.append(round)
            self.sv_const_logs.append(round_SV)
            with open(os.path.join(self.const_lambda_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
                pickle.dump(round_SV, f)
        if self.optimal_lambda:
            print('Optimal lambda FL SV', end=': ')
            round_SV = self.calculate_round_optimal_lambda_SV()
            print(round_SV)
            round_SV = round_SV.tolist()
            round_SV.append(round)
            self.sv_opt_logs.append(round_SV)
            with open(os.path.join(self.optimal_lambda_dir, 'Round{}.npy'.format(self.current_round)), 'wb') as f:
                pickle.dump(round_SV, f)
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