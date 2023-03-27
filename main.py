import utils.fflow as flw
import torch
import wandb
import time

def main():
    # read options
    option = flw.read_option()
    wandb.init(
        entity="aiotlab",
        project='SV_FL',
        name="FedSV_{}".format(option['task']),
        group=f"{option['task'].split('_')[0]}",
        tags=[option['task'].split('_')[2], option['task'].split('_')[3], option['task'].split('_')[4]],
        config=option
    )
    # set random seed
    print(option)
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    try:
        start = time.time()
        server.run()
        end = time.time()

    except:
        # log the exception that happens during training-time
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()