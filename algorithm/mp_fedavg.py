from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import os

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.global_save_dir = os.path.join('./chkpts', str(option['task']), 'global')
        os.makedirs(self.global_save_dir, exist_ok=True)
        self.local_save_dir = os.path.join('./chkpts', str(option['task']), 'local')
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
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients])
        # Save global checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.global_save_dir, 'Round{}.pt'.format(t)))
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

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


