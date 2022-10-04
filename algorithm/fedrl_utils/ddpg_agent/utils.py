
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import os
from sklearn.manifold import MDS

mds = MDS(n_components=10)

def plot(frame_idx, rewards):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    if not os.path.exists('./log/images/'):
        os.makedirs('./log/images/')
    plt.savefig('./log/images/'+date_time)
    plt.show()


def get_state(models):
    
    last_layers = []
    for model in models:
        last_layers.append(torch.flatten(list(model.parameters())[-2]).detach().cpu())
    
    retval = torch.vstack(last_layers)
    retval = retval.view(len(models), -1)
    
    retval = mds.fit_transform(retval)
    
    retval = torch.from_numpy(retval).double()

    return retval.flatten()


def get_reward(losses, M_matrix, beta=0.45):
    # beta = 0.45
    losses = np.asarray(losses)
    # return - beta * np.mean(losses) - (1 - beta) * np.std(losses)
    return - np.mean(losses) - (losses.max() - losses.min()) + 0.05 * np.sum(M_matrix)/2




