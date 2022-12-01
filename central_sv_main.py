import utils.fflow as flw
import os
from sv_centralbase import CentralizedShapleyValue


def main():
    # read options
    option = flw.read_option()
    option['num_gpus'] = len(option['gpu'])
    print(option)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in option['gpu']])
    print('=' * 100)
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    server = CentralizedShapleyValue(server=server)
    # calculate SV
    clients_SV = server.calculate_central_SV()
    print(clients_SV)

if __name__ == '__main__':
    main()
