from algorithm.mp_fedbase import MPBasicServer, MPBasicClient
from random import sample
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn


class MyXYDataset(Dataset):
    def __init__(self, X=[], totensor = True):
        if totensor:
            try:
                self.X = torch.tensor(X)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        print(type(self.test_data.X), len(self.test_data.X), self.test_data.X.shape)
        subset = sample(list(self.test_data.X), 16)
        subset = torch.vstack(subset).unsqueeze(1)
        print(type(subset), subset.shape)
        self.data_loader = DataLoader(MyXYDataset(X=sample(list(self.test_data.X), 16), totensor=False), batch_size=4, shuffle=True)
        
    
    def iterate(self, t, pool):
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients,pool)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: 
            return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        fusion_model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        self.model = self.train(fusion_model, models, device0)


    def train(self, fusion_model, models, device):
        
        fusion_model = fusion_model.to(device)
        fusion_model.train()
        softmax = nn.Softmax(dim=1)

        optimizer = Adam(fusion_model.parameters())
        for iter in range(5):
            for batch_id, batch_data in enumerate(self.data_loader):
                fusion_model.zero_grad()
                batch_data = batch_data.to(device)
                
                logits = torch.vstack([model(batch_data).unsqueeze(0) for model in models])
                fused_logit = softmax(torch.mean(logits, dim=0))
                model_logit = softmax(fusion_model(batch_data))
                
                loss = self.get_loss(pred=model_logit, ground=fused_logit)
                loss.backward()
                optimizer.step()
        
        return fusion_model


    def get_loss(self, pred, ground):
        loss = torch.trace(ground @ (torch.log(ground) -torch.log(pred)).T)
        return loss


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
