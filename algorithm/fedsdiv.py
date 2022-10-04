from algorithm.fedbase import BasicServer, BasicClient
from utils import fmodule
import torch


def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    sim = 0
    for layer_a, layer_b in zip(a.parameters(), b.parameters()):
        x, y = torch.flatten(layer_a), torch.flatten(layer_b)
        sim += (x.T @ y) / (torch.norm(x) * torch.norm(y))
    return sim


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: 
            return
        
        impact_factor = self.get_impact_factor(models)
        print("Impact_factor:", impact_factor)
        self.model = self.aggregate(models, p = impact_factor)
        return


    def get_impact_factor(self, model_list):
        models = []
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            models.append(model)
        
        similarity_matrix = torch.zeros([len(models), len(models)])
        for i in range(len(models)):
            for j in range(len(models)):
                similarity_matrix[i][j] = compute_similarity(models[i], models[j])
        
        similarity_matrix = 1/similarity_matrix
        
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)
        similarity_matrix *= (1- torch.eye(similarity_matrix.shape[0]))
                
        impact_factor = 1/(similarity_matrix.shape[0]-1) * torch.sum(similarity_matrix, dim=1).flatten()
        return impact_factor.tolist()
    
    
    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

