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

def plotCurve(num_epochs, losses_train, accs_train, losses_val, accs_val):
    plt.figure(figsize=(10, 3))
    
    # make epoch start from 1
    epochs = range(1, num_epochs+1) 
    
    # for loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses_train, label='train loss')
    plt.plot(epochs, losses_val, label='val loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
        
    # for acc curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs_train, label='train accuracy')
    plt.plot(epochs, accs_val, label='val accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()

    plt.show()

def applyModeltoTest(model,loader_test,device,):
    # test
    loss_test = 0.0
    acc_test = 0.0
    # load the best model
    model.load_state_dict(torch.load('best_models/best_model_baseline.pth'))
    model = model.to(device)

    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader_test:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # disable gradient computation
        with torch.no_grad():
            
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = torch.sum(torch.eq(torch.max(outputs, axis=1).indices, labels))
            
            # add up loss and acc
            loss_test += loss.item()
            acc_test += acc.item()

    # get average loss and acc
    loss_test_avg = loss_test / len(loader_test)
    acc_test_avg = acc_test / len(loader_test.dataset)

    print(f'test loss: {loss_test_avg:.2f}, test acc: {acc_test_avg:.2f}')  
    return loss_test_avg, acc_test_avg  
