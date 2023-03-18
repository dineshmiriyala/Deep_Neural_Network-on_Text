# imports
import sys

import torch
import warnings
import regex
import os
import random
import pickle

random.seed(9968)

warnings.filterwarnings('ignore')

if not os.path.exists('reddit_convos.txt') or not os.path.exists('encode_decode.pkl'):
    sys.exit("Data files not found")

lines = open('reddit_convos.txt', 'r').read().splitlines()

with open('encode_decode.pkl', 'rb') as file:
    encode_decode = pickle.load(file)
    words_to_integers = encode_decode['encode']
    integer_to_words = encode_decode['decode']


def text_clean(lines):
    new_string = []
    pattern = regex.compile(r'\b\w*(\w)\1{2,}\w\b')
    for line in lines:
        line = regex.sub(pattern, '', line)
        new_string.append(regex.sub(r"(.)\1+", r"\1", line))
    return new_string


lines = text_clean(lines)
lines = [line for line in lines if len(line.split()) > 0]

"""In this project I am taking 20 words as context. That means the block size would be 20.
I am keeping the data as constant once I run it. So, basically, I am saving the data into 
pkl file which I will be reusing."""

"""I will be using 80% as the training data and 10% each for validation and 
testing data."""

def build_datset(data):
    block_size = 20  # context length
    X,y = [] , []
    for lines in data:
        context = [0] * block_size
        for word in lines.split() + list('.'):
            index = words_to_integers[word]
            X.append(context)
            y.append(index)
            context = context[1:] + [index]
    X = torch.tensor(X)
    y = torch.tensor(y)
    return X,y

random.shuffle(lines)
n1 = int(0.8 * len(lines))
n2 = int(0.9 * len(lines))

if not os.path.exists('trainingData.pkl'):
    open('trainingData.pkl' , 'w')
    Xtrain, ytrain = build_datset(lines[:n1])
    Xval, yval = build_datset(lines[n1:n2])
    Xtest, ytest = build_datset(lines[n2:])
    data = {}
    data['X_train'] = Xtrain
    data['y_train'] = ytrain
    data['X_val'] = Xval
    data['y_val'] = yval
    data['Xtest'] = Xtest
    data['ytest'] = ytest
    with open('trainingData.pkl' , 'wb') as file:
        pickle.dump(data , file)
        print('Dataset files created successfully')