import utils.fflow as flw
import os
from sv_fedbase import ShapleyValueServer


def main():
    # read options
    option = flw.read_option()
    option['num_gpus'] = len(option['gpu'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in option['gpu']])
    print('=' * 100)
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    server = ShapleyValueServer(server=server, num_partitions=2)
    # calculate SV
    # type_ in ["exact", "const_lambda", "optimal_lambda"]
    server.calculate_FL_SV(type_=option['method'], number_of_samples_=300)
    

if __name__ == '__main__':
    main()
