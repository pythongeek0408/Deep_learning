# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:04:42 2022

@author: 91895
"""
#Import necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Creating a network
class NN(nn.Module):
    def __init__(self,input_size,num_classes): #input_size = 28*28, num_classes = 10 (0-9)
        super(NN,self).__init__()
        self.layer1 = nn.Linear(input_size,50)
        self.layer2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
    
#test the NN
# x = torch.randn(64,784) -----> 64 pictures with 28*28 pixels
# model = NN(784,10,50)   ------> Nuerons in hidden Layer --->50
# print(model.forward(x).shape) ---> After forward pass, Each picture should have probablities of it 
                                    #being some number from 0 to 9

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
num_classes = 10
#hidden_n = 50
l_rate = 0.001
epochs = 5
batch_size = 64

#Load Data
training_data = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle = True)
test_data = datasets.MNIST(root = 'dataset/',train = False, transform = transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle = True)

#Initialize the network
model = NN(input_size,num_classes).to(device)

#Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=l_rate)

#Training
for i in range(epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        #Transfer all the data to device
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        # print(data.shape) ---> [64,1,28,28]
        data = data.reshape(data.shape[0],-1)
        
        #Forward
        outputs = model.forward(data)
        loss = loss_func(outputs,targets)
        
        #Backprop
        optimizer.zero_grad()
        loss.backward()
        
        #SGD
        optimizer.step()
        
#Check the accuracy
def Check_Accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)
            output = model(x)
            _, predictions = output.max(1)
            
            num_correct+= (predictions == y).sum()
            num_samples+= predictions.size(0)   
            
        print(f'Correct Predictions: {num_correct} out of {num_samples}')
        print(f'Accuracy : {float(num_correct)/float(num_samples)*100:.2f}')
        
    model.train()

Check_Accuracy(train_loader, model)
Check_Accuracy(test_loader, model)


































