# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:53:52 2021

@author: samib
"""

import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from sklearn.model_selection import train_test_split

train_ds, test_ds = np.load('/content/train.npz'), np.load('/content/test.npz')

train_x = train_ds['arr_0']
train_y = train_ds['arr_1']
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, train_size=0.8,test_size=0.2, random_state=35)
X_test = test_ds['arr_0']
X_test.shape

# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_valid = x_valid.reshape(x_valid.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
x_train /= 255.0
x_valid /= 255.0
X_test /=255.0

print("Shape of training dataset ndarray: \n", x_train.shape)
print("Shape of Validating dataset ndarray: \n", x_valid.shape)
print("Shape of Testing dataset ndarray: \n", X_test.shape)

Test_dataSet = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().reshape(-1,1,28,28))

batch_size = 100
Test_dataLoad = torch.utils.data.DataLoader(Test_dataSet, batch_size = batch_size, shuffle = False)

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().reshape(-1,1,28,28), torch.from_numpy(y_train))
Valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_valid).float().reshape(-1,1,28,28), torch.from_numpy(y_valid))

batch_size = 100

train_load = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
Valid_load = torch.utils.data.DataLoader(Valid_dataset, batch_size = batch_size, shuffle = False)

class CNN(nn.Module):
  def __init__(self):
      super(CNN, self).__init__()
      # same padding Input size = output size
      # Same PAdding = (filtersize-1 / 2) ou (3-1 /2)
      self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
      # the output size of each of the 8 features maps
      # [(input_size - filter_size + 2(padding))/stride + 1] = (28 - 3 + 2)1 + 1 = 28
      self.batchnorm1 = nn.BatchNorm2d(8)
      self.relu = nn.ReLU()
      self.maxpool = nn.MaxPool2d(kernel_size = 2)
      # The output size = 28/2 = 14
      self.cnn2 = nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
      # Output Size of each of the 32 feature maps (14 - 5 + 2(2) / 1 + 1) = 14
      self.batchnorm2 = nn.BatchNorm2d(32)
      # Flatten the 32 feature maps: 7*7*32 = 1568
      self.fc1 = nn.Linear(1568, 600)
      self.dropout = nn.Dropout(p = 0.5)
      self.fc2 = nn.Linear(600, 10)
  # Defining the forward pass    
  def forward(self, x):
      out = self.cnn1(x)
      out = self.batchnorm1(out)
      out = self.relu(out)
      out = self.maxpool(out)
      out = self.cnn2(out)
      out = self.batchnorm2(out)
      out = self.relu(out)
      out = self.maxpool(out)
      # Flatten the 32 feature maps from Max Pool to feed it to the FC1 (100, 1568)
      out = out.view(100, 1568)
      #then we forward through our fully connected layer
      out = self.fc1(out)
      out = self.relu(out)
      out = self.dropout(out)
      out = self.fc2(out)
      return out
  
model = CNN()
CUDA = torch.cuda.is_available()
if CUDA:
    model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

num_epochs = 20 
train_loss = []
train_accuracy = []
Valid_loss = []
Valid_accuracy = []

for epoch in range(num_epochs):
    correct = 0 
    iterations = 0 
    iter_loss = 0.0
    model.train()

    for i,(inputs, labels) in enumerate(train_load):
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        output = model(inputs)
        loss = loss_fn(output, labels)
        iter_loss += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1
    train_loss.append(iter_loss/iterations)
    train_accuracy.append(correct / len(train_dataset))
    print(correct / len(train_dataset))

print(train_loss)

for epoch in range(num_epochs):
    valid_loss = []
    ep_loss = 0
    correct = 0 
    iterations = 0 
    model.eval()

    for i,(inputs, labels) in enumerate(Valid_load):
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        output = model(inputs)
        loss = loss_fn(output, labels)
        iter_loss += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1
    valid_loss.append(ep_loss/iterations)
    Valid_accuracy.append((100 * correct / len(Valid_dataset)))

print(valid_loss)
print(Valid_accuracy)

for epoch in range(num_epochs):
    Test_loss = []
    ep_loss = 0
    correct = 0 
    iterations = 0 
    model.eval()

    for i,(inputs) in enumerate(Test_dataLoad):
       # if CUDA:
        #    inputs = inputs.cuda()
        output = model(inputs)
       # loss = loss_fn(output)
        #iter_loss += loss.item()


       # optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        _, predicted = torch.max(output, 1)
       # correct += (predicted == labels).sum().item()
       # iterations += 1

print("Training Loss: {:.3f}, Testing Loss: {:.3f}, Training Accuracy{:.3f}, Testing Accuracy: {:.3f}".format(train_loss[-1], test_loss[-1], train_accuracy[-1], test_accuracy[-1]))

type(test_loss)
device = torch.device('cuda:0')  
X_test = torch.from_numpy(X_test).float().to(device)
output = model(X_test)
_, pred = torch.max(output)
pd.DataFrame(pred).to_csv('CNN')

import torch
torch.cuda.is_available()

device = torch.device('cuda')

a = torch.rand(3, 3)
a

b = torch.rand(3, 3).to(device)
b

c = torch.rand(3, 3).cuda()
c