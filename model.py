import pickle
import sys
import os
import torch
import warnings

warnings.filterwarnings('ignore')
gen = torch.Generator().manual_seed(9968)
if not os.path.exists('data/trainingData.pkl'):
    sys.exit("Training data file does not exist.\n")

with open('data/trainingData.pkl', 'rb') as file:
    data = pickle.load(file)


class linear:
    def __init__(self, fan_in, fan_out, Bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=gen) / fan_in ** 0.5  # kaiming init
        self.bias = torch.zeros(fan_out) if Bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters after back prop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers(training with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.std(0, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        x_norm = (x - x.mean) / torch.sqrt(xvar + self.eps)  # normalizing
        self.out = self.gamma * x_norm + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
