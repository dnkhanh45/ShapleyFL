import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
        

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, init_w=1e-1):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, batch_size):
        state = state.view(batch_size, -1)
        action = action.view(batch_size, -1)

        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, init_w=1e-1):
        super(PolicyNetwork, self).__init__()

        assert num_outputs % 3 == 0, "The network is 3-output, hence the number of outputs must be a multiplicity of 3"
        num_outputs = int(num_outputs / 3)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.linear3e = nn.Linear(hidden_size, hidden_size)
        self.linear4e = nn.Linear(hidden_size, num_outputs)

        # Free learning epochs
        freeze_layer(self.linear3e)
        freeze_layer(self.linear4e)

        self.linear3i = nn.Linear(hidden_size, hidden_size)
        self.linear4i = nn.Linear(hidden_size, num_outputs)

        self.linear3n = nn.Linear(hidden_size, num_outputs)

        self.activation = nn.Sigmoid()

        self.linear4i.weight.data.uniform_(-init_w, init_w)
        self.linear4i.bias.data.uniform_(-init_w, init_w)

        self.linear4e.weight.data.uniform_(-init_w, init_w)
        self.linear4e.bias.data.uniform_(-init_w, init_w)

        self.linear3n.weight.data.uniform_(-init_w, init_w)
        self.linear3n.bias.data.uniform_(-init_w, init_w)
        

    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))

        epochs = F.leaky_relu(self.linear3e(x))
        epochs = F.leaky_relu(self.linear4e(epochs))
        epochs = self.activation(epochs)

        impact = F.leaky_relu(self.linear3i(x))
        impact = F.leaky_relu(self.linear4i(impact))
        impact = self.activation(impact)

        noise = F.leaky_relu(self.linear3n(x))
        noise = self.activation(noise)

        return torch.cat([epochs, impact, noise])

    def get_action(self, state):
        action = self.forward(state)
        action = torch.flatten(action)
        return action.detach().cpu()


if __name__ == '__main__':
    model = ValueNetwork(10 * 4, 10 * 3, 256)

    state = torch.ones(size=[4, 40])
    action = torch.ones(size=[12, 10])

    value = model(state, action, 4)
    print(value.shape)
