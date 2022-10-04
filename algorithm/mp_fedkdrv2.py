from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch.nn as nn
import numpy as np
import torch
import os
import copy

def KL_loss_compute(target, input):
    """
    Compute KL on Similarity Kernel applied matrices
    target: N x d
    input:  N x c
    Author: Anh Duy ft. Nguyen Nang Hung
    """
    dot_inp = input @ input.T
    norm_inp = torch.norm(input, dim=1)
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = cosine_inp - torch.diag(torch.diagonal(cosine_inp))
    cosine_inp = cosine_inp[cosine_inp!=0].view(input.shape[0], -1)
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp / torch.sum(cosine_inp, dim = 1, keepdim=True)
    
    dot_tar = target @ target.T
    norm_tar = torch.norm(target, dim=1)
    norm_mtx_tar = torch.matmul(norm_tar.unsqueeze(1), norm_tar.unsqueeze(0))
    cosine_tar = dot_tar / norm_mtx_tar
    cosine_tar = cosine_tar - torch.diag(torch.diagonal(cosine_tar))
    cosine_tar = cosine_tar[cosine_tar!= 0].view(target.shape[0], -1)
    cosine_tar = 1/2 * (cosine_tar + 1)
    cosine_tar = cosine_tar / torch.sum(cosine_tar, dim = 1, keepdim=True)
    
    losses = cosine_tar * torch.log(cosine_tar / cosine_inp)
    loss = torch.sum(losses)
    
    return loss


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    
    def run(self):
        super().run()
        return
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients,pool)
        if not self.selected_clients: 
            return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        
        
    def train(self, model, device):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
            device: the device to be trained on
        :return
        """
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, src_model, batch_data, device)
                loss.backward()
                optimizer.step()
        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, src_model, data, device):
        tdata = self.data_to_device(data, device)
        output_s, representation_s = model.pred_and_rep(tdata[0])   # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])     # Teacher

        P_t = representation_t
        Q_s = representation_s

        kl_loss = KL_loss_compute(P_t, Q_s)
        loss = self.lossfunc(output_s, tdata[1])
        
        return loss + kl_loss