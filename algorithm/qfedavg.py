"""This is a non-official implementation of 'Fair Resource Allocation in
Federated Learning' (http://arxiv.org/abs/1905.10497).  And this implementation
refers to the official github repository https://github.com/litian96/fair_flearn """

from .fedbase import BasicServer, BasicClient
import numpy as np
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'q':0.1})

    def iterate(self):
        # sample clients
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models, train_losses = res['model'], res['loss']
        # plug in the weight updates into the gradient
        grads = [(self.model- model) / self.lr for model in models]
        Deltas = [gi*np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # estimation of the local Lipchitz constant
        hs = [self.q * np.float_power(li + 1e-10, (self.q - 1)) * (gi.norm() ** 2) + 1.0 / self.lr * np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # aggregate
        self.model = self.aggregate(Deltas, hs)
        return

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta/demominator for delta in Deltas]
        updates = fmodule._model_sum(scaled_deltas)
        new_model = self.model - updates
        return new_model

class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        train_loss = self.test(model, 'train')['loss']
        self.train(model)
        cpkg = self.pack(model, train_loss)
        return cpkg

    def pack(self, model, loss):
        return {
            "model" : model,
            "loss": loss,
        }
