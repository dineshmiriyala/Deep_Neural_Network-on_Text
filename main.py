from model import linear, tanh, BatchNorm1d, gen, sequential, embedding, flatten
import sys
import os
import torch
import pickle
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if not os.path.exists('data/trainingData_40.pkl') or not os.path.exists('data/encode_decode.pkl'):
    sys.exit("Training data file does not exist. Try running the pre_process.py\n")

with open('data/trainingData_40.pkl', 'rb') as file:
    data = pickle.load(file)

with open('data/encode_decode.pkl', 'rb') as file:
    encode_decode = pickle.load(file)
    encode = encode_decode['encode']
    decode = encode_decode['decode']

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['Xtest']
y_test = data['ytest']

"""x*w"""

# I am squashing all the input probabilities into 240 dimensions, that is 240` features.
n_emb = 240
n_hidden = 300
block_size = 40
vocab_size = len(encode)

model = sequential([
    embedding(vocab_size, n_emb), flatten(), linear(n_emb * block_size, n_hidden, Bias=False), BatchNorm1d(n_hidden), tanh(),
    linear(n_hidden, n_hidden, Bias=False), BatchNorm1d(n_hidden), tanh(),
    linear(n_hidden, n_hidden, Bias=False), BatchNorm1d(n_hidden), tanh(),
    linear(n_hidden, n_hidden, Bias=False), BatchNorm1d(n_hidden), tanh(),
    linear(n_hidden, vocab_size)])

with torch.no_grad():
    # making last layer less confident
    model.layers[-1].weight *= 0.1
    # applying gain for all the remaining layers, kaiming init, 5/3
    for layer in model.layers:
        if isinstance(layer, linear):
            layer.weight *= 1.0  # 5/3

for p in model.parameters():
    p.requires_grad = True

# training part
steps = 10000
batch_size = 32
lossi = []
ud = []

for i in range(steps):
    # mini batches
    index = torch.randint(0, X_train.shape[0], (batch_size,), generator=gen)

    xindex, yindex = X_train[index], y_train[index]
    # model
    x = model(xindex)
    loss = F.cross_entropy(x, y_train[index])
    # backward pass
    for p in model.parameters():
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
    for p in model.parameters():
        p.data += -lr * p.grad

    # tracking stats
    if i % 10000 == 0:
        print(f'{i:7d}/{steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])

print(f'Final loss: {loss.item()}')
# plotting the loss function
plt.plot(torch.tensor(lossi , device = 'cpu').view(-1, 1000).mean(-1))
plt.title(f'Training loss')
plt.ylabel("loss")
plt.xlabel("iteration")
plt.show()
# turning off training mode
for layer in model.layers:
    layer.training = False


# evaluating validation loss
@torch.no_grad()
def split_loss(split):
    loss_test = []
    x, y = {'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)}[split]
    for i in range(x.shape[0] // 50):
        index = torch.randint(0, x.shape[0], (100,))
        logits = model(x[index])
        lossi.append(F.cross_entropy(logits, y[index]))
    plt.plot(torch.tensor(loss_test, device= 'cpu'))
    plt.title(f'{split} loss')
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()

split_loss('val')

#%%
