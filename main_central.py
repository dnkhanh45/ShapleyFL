import utils.fflow as flw
import torch
import wandb
from bitsets import bitset
import itertools
import copy
from torch.utils.data import ConcatDataset

def main():
    # read options
    option = flw.read_option()
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    all_clients = copy.deepcopy(server.clients)
    CLIENTS_BITSET = bitset('clients_bitset', tuple(client.name for client in all_clients))
    used_client = copy.deepcopy(all_clients[0])
    print(server.num_rounds)
    for client in all_clients:
        print(client.name, client.epochs)
    for subset in itertools.chain.from_iterable(itertools.combinations(all_clients, _) for _ in range(1, len(all_clients) + 1)):
        # set random seed
        flw.setup_seed(option['seed'])
        server = flw.initialize(option)
        used_client.train_data = ConcatDataset([client.train_data for client in subset])
        used_client.valid_data = ConcatDataset([client.valid_data for client in subset])
        used_client.datavol = len(used_client.train_data)
        server.clients = [used_client]
        server.num_clients = 1
        server.local_data_vols = [c.datavol for c in server.clients]
        server.total_data_vol = sum(server.local_data_vols)
        for client in subset:
            print(client.name, len(client.train_data), len(client.valid_data))
        print(server.clients[0].name, len(server.clients[0].train_data), len(server.clients[0].valid_data), server.clients[0].datavol)
        print(server.local_data_vols, server.total_data_vol)
        # start federated optimization
        try:
            server.run(suffix_log_filename=CLIENTS_BITSET([client.name for client in subset]).bits())
        except:
            # log the exception that happens during training-time
            flw.logger.exception("Exception Logged")
            raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()