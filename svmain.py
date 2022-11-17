import shapley.svflow as flw
import numpy as np
import torch
import os
import multiprocessing

def main():
    multiprocessing.set_start_method('spawn')
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
    # for key, value in server.__dict__.items():
    #     print(key)
    #     print('\t', value)
    #     print('-' * 100)
    # print(len(server.test_data))
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()
