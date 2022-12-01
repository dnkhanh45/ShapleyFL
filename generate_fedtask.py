import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='mnist')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0.5)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--num_samples', help='number of samples in clients', type=int, default=4200, required=False)
    parser.add_argument('--zipf_skew', help='the degree of sample size per client', type=float, default=0.7, required=False)
    parser.add_argument('--seed', help='set seed', type=int, default=0)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    print(option['skew'])
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['dataset'], 'core'])), 'TaskGen')
    if option['dataset'] == 'synthetic':
        generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'], num_samples=option['num_samples'], zipf_skew=option['zipf_skew'], seed=option['seed'])
    else:
        generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'])
    generator.run()
