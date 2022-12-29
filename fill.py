import numpy as np
import torch
import pickle
import os
import itertools
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fedtask', help='fedtask', type=str)
    parser.add_argument('--exact_dir', help='exact dir', type=str)
    args = parser.parse_args()
    fedtask = args.fedtask
    exact_dir = args.exact_dir
    
    exact = list()
    for i in range(1, 101):
        with open(os.path.join('SV_result', fedtask, exact_dir, 'Round{}.npy'.format(i)), 'rb') as f:
            round_exact = pickle.load(f)
            exact.append(round_exact.tolist())
    exact = np.array(exact)
    
    exact_var = list()
    for i in range(0, 100):
        tmp = list()
        for e in exact[i]:
            if e != 0.0:
                tmp.append(torch.scalar_tensor(e.item(), requires_grad=False))
            else:
                tmp.append(torch.scalar_tensor(0.0, requires_grad=True))
        exact_var.append(tmp)
                
    num_steps = 100
    learning_rate = 0.5
    for step in range(0, num_steps):
        loss = torch.scalar_tensor(0.0, requires_grad=False)
        count = 0
        for i, j in itertools.combinations(range(0, 100), 2):
            n = len(exact_var[i])
            X = torch.scalar_tensor(0.0, requires_grad=False)
            Y = torch.scalar_tensor(0.0, requires_grad=False)
            Z = torch.scalar_tensor(0.0, requires_grad=False)
            for k in range(0, n):
                X += exact_var[i][k] ** 2
                Y += exact_var[j][k] ** 2
                Z += exact_var[i][k] * exact_var[j][k]
            loss += max(1.0 - Z / torch.sqrt(X * Y),
                        torch.scalar_tensor(1.0 - np.exp(- np.abs(i - j) / 1000).item(), requires_grad=False))
            count += 1
        loss = loss / count
        print('Step {}, loss: {}'.format(step + 1, loss.item()))
        loss.backward()
        for var in exact_var:
            for x in var:
                if x.requires_grad:
                    x.data = x.data - learning_rate * x.grad
                    x.grad = None

    new_exact = list()
    for var in exact_var:
        tmp = list()
        for x in var:
            tmp.append(x.data.item())
        new_exact.append(tmp)
    new_exact = np.array(new_exact)

    with open(os.path.join('SV_result', fedtask, exact_dir, 'fill_exact2.npy'), 'wb') as f:
        pickle.dump(new_exact, f)
        