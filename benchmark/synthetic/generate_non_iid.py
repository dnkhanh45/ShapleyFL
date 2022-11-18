import os
import argparse
import json
import numpy as np

def softmax(x):
    scale_x = x - np.max(x)
    result = np.exp(scale_x) / np.sum(np.exp(scale_x))
    return result


def generate(args):
    alpha = args.alpha
    beta = args.beta
    n_clients = args.n_clients
    n_dim = args.n_dim
    n_classes = args.n_classes
    n_train = args.n_train
    n_valid = args.n_valid
    n_test = args.n_test
    zipf_z = args.zipf_z
    seed = args.seed

    np.random.seed(seed)
    p = (1 / np.arange(1, n_clients + 1)) ** zipf_z
    p /= p.sum()
    print('Zipf p:')
    print(p)
    n_clients_train = np.random.multinomial(n=n_train, pvals=p)
    print('Number of training samples:')
    print(n_clients_train)
    n_clients_valid = np.random.multinomial(n=n_valid, pvals=p)
    print('Number of valid samples:')
    print(n_clients_valid)
    n_clients_test = np.random.multinomial(n=n_test, pvals=p)
    print('Number of test samples:')
    print(n_clients_test)

    sigma = np.diag(np.arange(1, n_dim + 1) ** -1.2)
    
    clients = list()
    for k in range(n_clients):
        # At client with index k
        # # Generate model
        u_k = np.random.normal(loc=0.0, scale=alpha)
        b_k = np.random.normal(loc=u_k, scale=1.0, size=n_classes)
        W_k = np.random.normal(loc=u_k, scale=1.0, size=(n_classes, n_dim))
        # # Generate data and label
        B_k = np.random.normal(loc=0.0, scale=beta)
        v_k = np.random.normal(loc=B_k, scale=1.0, size=n_dim)
        # # # Training samples
        x_train_k = np.random.multivariate_normal(mean=v_k, cov=sigma, size=n_clients_train[k])
        y_train_k = np.argmax(softmax(x_train_k.dot(W_k.T) + b_k), axis=1)
        # # # Valid samples
        x_valid_k = np.random.multivariate_normal(mean=v_k, cov=sigma, size=n_clients_valid[k])
        y_valid_k = np.argmax(softmax(x_valid_k.dot(W_k.T) + b_k), axis=1)
        # # # Test samples
        x_test_k = np.random.multivariate_normal(mean=v_k, cov=sigma, size=n_clients_test[k])
        y_test_k = np.argmax(softmax(x_test_k.dot(W_k.T) + b_k), axis=1)

        client_k = {
            'index': k,
            'train_set': {
                'x': x_train_k.tolist(),
                'y': y_train_k.tolist()
            },
            'valid_set': {
                'x': x_valid_k.tolist(),
                'y': y_valid_k.tolist()
            },
            'test_set': {
                'x': x_test_k.tolist(),
                'y': y_test_k.tolist()
            },
            'model': {
                'W': W_k.tolist(),
                'b': b_k.tolist()
            }
        }
        clients.append(client_k)
    
    data = vars(args)
    data['clients'] = clients
    
    filename = 'synthetic({},{}){}.json'.format(alpha, beta, n_clients)
    with open(os.path.join('./data', filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('Saved at file named "{}"'.format(filename))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', help='Alpha', type=float)
    parser.add_argument('--beta', help='Beta', type=float)
    parser.add_argument('--n_clients', help='Number of clients', type=int)
    parser.add_argument('--n_dim', help='Number of dimensions', type=int)
    parser.add_argument('--n_classes', help='Number of classes', type=int)
    parser.add_argument('--n_train', help='Total of training samples', type=int)
    parser.add_argument('--n_valid', help='Total of validation samples', type=int)
    parser.add_argument('--n_test', help='Total of test samples', type=int)
    parser.add_argument('--zipf_z', help='Zipf distribution z parameter', type=float)
    parser.add_argument('--seed', help='Seed number', type=int, default=0)
    args = parser.parse_args()
    print(args)
    generate(args)