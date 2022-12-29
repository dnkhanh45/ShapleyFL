import utils.fflow as flw
import torch
import wandb

def main():
    # read options
    option = flw.read_option()
    # wandb.init(
    #     project='ShapleyValue',
    #     name=f"{option['task']}",
    #     group=f"{option['task'].split('_')[0]}",
    #     tags=[option['task'].split('_')[2], option['task'].split('_')[3], option['task'].split('_')[4]],
    #     config=option
    # )
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    try:
        server.run()
    except:
        # log the exception that happens during training-time
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()