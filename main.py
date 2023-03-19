from model import linear, tanh, BatchNorm1d, gen
import sys
import os
import torch
import pickle
import warnings
import torch.nn.functional as F
from tqdm import tqdm
warnings.filterwarnings('ignore')

torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if not os.path.exists('data/trainingData.pkl') or not os.path.exists('data/encode_decode.pkl'):
    sys.exit("Training data file does not exist. Try running the pre_process.py\n")

with open('data/trainingData.pkl', 'rb') as file:
    data = pickle.load(file)

with open('data/encode_decode.pkl', 'rb') as file:
    encode_decode = pickle.load(file)
    encode = encode_decode['encode']
    decode = encode_decode['decode']

X_train = data['X_train']
y_train = data['y_train']

"""x*w"""

# I am squashing all the input probabilities into 60 dimensions, that is 60 features.
n_emb = 120
n_hidden = 300
block_size = 20
vocab_size = len(encode)
probabilities = torch.randn((vocab_size, n_emb), generator = gen).cuda()

layers = [linear(n_emb * block_size,n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, n_hidden, Bias = False), BatchNorm1d(n_hidden), tanh(),
          linear(        n_hidden, vocab_size, Bias = False), BatchNorm1d(vocab_size)]

with torch.no_grad():
    # making last layer less confident
    layers[-1].gamma *= 0.1
    # applying gain for all the remaining layers, kaiming init, 5/3
    for layer in layers:
        if isinstance(layer, linear):
            layer.weight *= 1.0  #5/3

params = [probabilities] +[p for layer in layers for p in layer.parameters()]
print('number of parameters: ',sum(p.nelement() for p in params)) # number of parameters
for p in params:
    p.requires_grad = True

#training part
steps = 100000
batch_size = 32
lossi = []
ud = []

for i in range(steps):
    # mini batches
    index = torch.randint(0, X_train.shape[0], (batch_size,), generator = gen)

    # forward pass
    emb = probabilities[X_train[index]]
    x = emb.view(emb.shape[0], -1) # concatenating the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y_train[index])

    # backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in params:
        p.grad = None
    loss.backward()

    # updating parameters
    # learning rate decay
    if i < 10000:
        lr = 0.1
    elif i < 50000:
        lr = 0.01
    else:
        lr = 0.001
    for p in params:
        p.data += -lr * p.grad

    # tracking stats
    if i % 10000 ==0:
        print(f'{i:7d}/{steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in params])

for layer in layers:
    layer.training = False

for _ in range(20):

    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        # forward pass the neural net
        emb = probabilities[torch.tensor([context])]  # (1,block_size,n_embd)
        x = emb.view(emb.shape[0], -1)  # concatenate the vectors
        for layer in layers:
            x = layer(x)
        probs = F.softmax(x, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1, generator=gen).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break

    print(' '.join(decode[i] for i in out))  # decode and print the generated word