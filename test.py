import numpy as np

def gradient_descent(client_idx, start, gradient, learning_rate, max_iter, tol=0.01):
    steps = [start] # history tracking
    x = start
    total_diff = 0

    for _ in range(max_iter):
        # print(f'Gradient at step {_} of client {client_idx}: {gradient(x)}')
        diff = learning_rate*gradient(x)
        total_diff += diff
        if np.abs(diff)<tol:
            break    
        x = x - diff
        steps.append(x) # history tracing

    return total_diff, x

def func1(x):
    return x**2-4*x+1

def gradient_func1(x):
    return 2*x - 4

if __name__ == '__main__':
    start = [10, -5, -10, 3, -20]
    rounds = 100
    learning_rate = 0.1
    max_iter = [50, 50, 50, 50, 50]
    for round in range(rounds):
        steps1, x1 = gradient_descent(1, start[0], gradient_func1, learning_rate, max_iter[0])
        steps2, x2 = gradient_descent(2, start[1], gradient_func1, learning_rate, max_iter[1])
        steps3, x3 = gradient_descent(3, start[2], gradient_func1, learning_rate, max_iter[2])
        steps4, x4 = gradient_descent(4, start[3], gradient_func1, learning_rate, max_iter[3])
        steps5, x5 = gradient_descent(5, start[4], gradient_func1, learning_rate, max_iter[4])
        print(f'gradient update at round {round}: {steps1} - {steps2} - {steps3} - {steps4} - {steps5}')
        print(f'value before aggregate at round {round}: {x1} - {x2} - {x3} - {x4} - {x5}')
        start[0] = (5 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5) / 75
        start[1] = (5 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5) / 75
        start[2] = (5 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5) / 75
        start[3] = (5 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5) / 75
        start[4] = (5 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5) / 75
    
        print(f'value after aggregate: {start}')
        print('*' * 10)
        