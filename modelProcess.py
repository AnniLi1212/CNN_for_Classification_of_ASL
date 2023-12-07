import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import glob
import seaborn as sn
import pandas as pd
import mediapipe as mp

# function to print number of parameters
def count_params(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")
            total += param.numel()
    print(f'total number of trainable params: {total}')
    
def modelTrain(model,learning_rate,num_epochs,loader_train,loader_val,device):
    model = model.to(device)
    # set the loss function
    criterion = nn.CrossEntropyLoss()
    # set the optimizer
    lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    losses_train = []
    accs_train = []
    losses_val = []
    accs_val = []
        
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        
        loss_train = 0.0
        acc_train = 0.0
        loss_val = 0.0
        acc_val = 0.0
            
        # train the model
        model.train()
        
        for inputs, labels in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)
            # reset gradients
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = torch.sum(torch.eq(torch.max(outputs, axis=1).indices, labels))
            
            # backprop
            loss.backward()
            optimizer.step()

            # add up loss and accuracy
            loss_train += loss.item()
            acc_train += acc.item()
            
        # compute average loss and accuracy
        loss_train_avg = loss_train / len(loader_train)
        acc_train_avg = acc_train / len(loader_train.dataset)
        
        losses_train.append(loss_train_avg)
        accs_train.append(acc_train_avg)
        
        train_end_time = time.time()
        train_time = train_end_time - start_time
        
        print(f'epoch [{epoch+1}/{num_epochs}] - train loss: {loss_train_avg:.2f}, train acc: {acc_train_avg:.2f}, time taken: {train_time:.2f}')
        
        # validation
        model.eval()
        
        for inputs, labels in loader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            # disable gradient computation
            with torch.no_grad():
                
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = torch.sum(torch.eq(torch.max(outputs, axis=1).indices, labels))
                
                # add up loss and accuracy
                loss_val += loss.item()
                acc_val += acc.item()
                
        # compute average loss and accuracy
        loss_val_avg = loss_val / len(loader_val)
        acc_val_avg = acc_val / len(loader_val.dataset)
        
        losses_val.append(loss_val_avg)
        accs_val.append(acc_val_avg)
        
        val_end_time = time.time()
        val_time = val_end_time - train_end_time
        
        print(f'epoch [{epoch+1}/{num_epochs}], val loss: {loss_val_avg:.2f}, val acc: {acc_val_avg:.2f}, time taken: {val_time:.2f}')
        
        # save the model with higest val acc as best model 
        if acc_val_avg > best_val_acc:
            best_val_acc = acc_val_avg
            if not os.path.exists('best_models'):
                os.makedirs('best_models')
            torch.save(model.state_dict(), 'best_models/best_model_baseline.pth')
            print(f'model saved with val acc: {best_val_acc}')
            
    print('training done.')
    return losses_train, accs_train, losses_val, accs_val