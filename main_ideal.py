import utils.fflow as flw
import torch
import wandb
from bitsets import bitset
import itertools
import copy

def main():
    # read options
    option = flw.read_option()
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    all_clients = copy.deepcopy(server.clients)
    CLIENTS_BITSET = bitset('clients_bitset', tuple(client.name for client in all_clients))
    for subset in itertools.chain.from_iterable(itertools.combinations(all_clients, _) for _ in range(1, len(all_clients) + 1)):
        # set random seed
        flw.setup_seed(option['seed'])
        server = flw.initialize(option)
        server.clients = list(subset)
        server.num_clients = len(subset)
        server.local_data_vols = [c.datavol for c in server.clients]
        server.total_data_vol = sum(server.local_data_vols)
        for client in server.clients:
            print(client.name, len(client.train_data), len(client.valid_data))
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