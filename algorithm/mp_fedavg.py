from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import os

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
    def finish(self, model_path):
        if not Path(model_path).exists():
            os.system(f"mkdir -p {model_path}")
        task = self.option['task']
        torch.save(self.model.state_dict(), f"{model_path}/{self.name}_{self.num_rounds}_{task}.pth")
        pass
    
    def run(self):
        super().run()
        self.finish(f"algorithm/fedrl_utils/baseline/{self.name}")
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


