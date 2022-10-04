from algorithm.fedbase import BasicServer, BasicClient
from utils import fmodule
import torch
import copy


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
        self.rival_list = None
        self.rival_thr = 0.3
        self.model_list = None
        
    
    def communicate_with(self, client_id):
        svr_pkg = self.pack(client_id)
        if self.clients[client_id].is_drop(): 
            return None
        return self.clients[client_id].reply(svr_pkg)
    
    
    def pack(self, client_id):
        return {
            "model" : copy.deepcopy(self.model),
            "rival_models": self.get_rival_of(client_id)
        }

    
    def iterate(self, t):
        self.selected_clients = self.sample()
        self.model_list, train_losses = self.communicate(self.selected_clients)
        
        self.create_rival_list(self.model_list)
        self.rival_thr = max(self.rival_thr * 1.15, 0.85)
        
        if not self.selected_clients:
            return
        self.model = self.aggregate(self.model_list, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return


    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
    
    
    def create_rival_list(self, model_list):
        """
        Considering model a and model b
        If similarity(a, b) < threshold then a is rival of b
        
        Note: this threshold must be somewhat decayable
        """
        models = []
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            models.append(model)
            
        similarity_matrix = torch.zeros([len(model_list), len(model_list)])
        for i in range(len(models)):
            for j in range(len(models)):
                similarity_matrix[i][j] = compute_similarity(models[i], models[j])
        
        self.rival_list = []
        for i in range(len(models)):
            rival = []
            for j in range(len(models)):
                if similarity_matrix[i][j] <= self.rival_thr:
                    rival.append(j)
            self.rival_list.append(rival)
        return
    
    
    def get_rival_of(self, client_id):
        if self.rival_list:
            i = self.selected_clients.index(client_id)
            return [copy.deepcopy(self.model_list[j]) for j in self.rival_list[i]]
        else:
            return []
    

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


    def unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['rival_models']


    def reply(self, svr_pkg):
        model, rival_list = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model, rival_list)
        cpkg = self.pack(model, loss)
        return cpkg


    def train(self, model, rival_list):
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                
                divergence_loss = 0
                for rival in rival_list:
                    for pm, ps in zip(model.parameters(), rival.parameters()):
                        divergence_loss += torch.sum(torch.pow(pm-ps,2))
                if len(rival_list) > 0:
                    loss = self.calculator.get_loss(model, batch_data) + 0.005 * 1 / len(rival_list) * divergence_loss
                else:
                    loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        return