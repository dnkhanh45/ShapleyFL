import torch
from algorithm.fedbase import BasicServer, BasicClient
from utils import fmodule
import numpy as np


def converge_to_array(dictionary_list):
    max_key = 0
    for dictionary in dictionary_list:
        for key, value in dictionary.items():
            max_key = max(max_key, key)
    
    output = np.zeros([len(dictionary_list), max_key+1])
    for i in range(len(dictionary_list)):
        dictionary = dictionary_list[i]
        for key, value in dictionary.items():
            output[i][key] += value
        
    return output

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.optim_ratio = None
    
    def run(self):
        super().run()
        return
    
    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        insights = [cp["insight"] for cp in packages_received_from_clients]
        return models, insights

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, insights = self.communicate(self.selected_clients)
        if not self.selected_clients: return

        if self.optim_ratio is None:
            self.optim_ratio = self.process_insight(insights)
        self.model = self.aggregate(models, p = self.optim_ratio)
        return
    
    
    def aggregate(self, models, p=...):
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
    

    def process_insight(self, insights):
        sample_distribution_array = converge_to_array(insights)
        print(sample_distribution_array)

        def f(w):
            e = np.exp(w)/ np.sum(np.exp(w))
            return -np.std(np.matmul(sample_distribution_array.T, e))

        npop = 100     # population size
        sigma = 0.1    # noise standard deviation
        alpha = 0.001  # learning rate
        dim = sample_distribution_array.shape[0]

        w = np.random.randn(dim) # initial guess
        for i in range(500):
            N = np.random.randn(npop, dim)
            R = np.zeros(npop)

            for j in range(npop):
                w_try = w + sigma*N[j]
                R[j] = f(w_try)

            A = (R - np.mean(R)) / np.std(R)
            w = w + alpha/(npop*sigma) * np.dot(N.T, A)

        e = np.exp(w) / np.sum(np.exp(w))
        res = np.matmul(sample_distribution_array.T, e)
        print("Final result:", res)
        print("Final std:", np.std(res))
        print("Final impact:", e)
        return e.tolist()


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.insight = None

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)

        if self.insight is None:
            self.insight = self.get_insight()

        cpkg = self.pack(model, self.insight)
        return cpkg


    def pack(self, model, insight):
        return {
            "model": model,
            "insight": insight
        }


    def get_insight(self):
        """
        Returns dictionary of labels and number of samples
        Examples:
            {
                '1': 15,
                '2': 20,
                '6': 12
            }
        """
        features = self.train_data.X
        labels = self.train_data.Y
        
        insight_dict = {}
        for sample, target in zip(features, labels):
            if target.item() not in insight_dict.keys():
                insight_dict[target.item()] = 1
            else:
                insight_dict[target.item()] += 1

        return insight_dict