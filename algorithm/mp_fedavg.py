from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import os
import numpy as np
from tqdm.auto import tqdm

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.global_save_dir = os.path.join(option['log_folder'], str(option['task']), 'global')
        os.makedirs(self.global_save_dir, exist_ok=True)
        self.local_save_dir = os.path.join(option['log_folder'], str(option['task']), 'local')
        os.makedirs(self.local_save_dir, exist_ok=True)
    
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
    
    def iterate(self, t, pool):
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
        # Save local checkpoints
        round_local_save_dir = os.path.join(self.local_save_dir, 'Round{}'.format(t))
        os.makedirs(round_local_save_dir, exist_ok=True)
        for model, name in zip(models, names):
            torch.save(model.state_dict(), os.path.join(round_local_save_dir, '{}.pt'.format(name)))
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # weighted
        p = [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients]
        # uniform
        # p = np.ones(len(self.selected_clients)) / len(self.selected_clients)
        # p = p / p.sum()
        # self.model = self.aggregate(models, p = )
        self.model = self.aggregate(models, p=p)
        # Save global checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.global_save_dir, 'Round{}.pt'.format(t)))
        return


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


