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
from dataProcess import get_class
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from getBestModel import getBestModel

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

def applyModeltoTest(model,loader_test,device):
    # test
    loss_test = 0.0
    acc_test = 0.0
    # get true label and predicted labels as one-hot
    y_test = []
    y_score = []
    # load the best model
    model = getBestModel(model)
    model = model.to(device)

    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    num_classes = 29
    classes = [get_class(loader_test, i) for i in range(num_classes)]
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = np.zeros((num_classes))
    
    start_time=time.time()
    for inputs, labels in loader_test:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # disable gradient computation
        with torch.no_grad():
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            score=torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            y_score.append(score)
            y_test.append(labels)
            acc = torch.sum(torch.eq(torch.max(outputs, axis=1).indices, labels))
            
            # add up loss and acc
            loss_test += loss.item()
            acc_test += acc.item()
            
            # prediction and create confusion matrix
            for true, pred in zip(labels, preds):
                confusion_matrix[true, pred] += 1
                count[true.long()] += 1
    end_time=time.time()
    # get average loss and acc
    loss_test_avg = loss_test / len(loader_test)
    acc_test_avg = acc_test / len(loader_test.dataset)

    print(f'test loss: {loss_test_avg:.4f}, test acc: {acc_test_avg:.4f}')  
    
    plotConfusionMatrix(confusion_matrix,count,classes)
    printPrecisioAndRecall(confusion_matrix,count,loader_test)
    plotPrecisionAndRecall(y_test,y_score,classes,loader_test)
    
    return loss_test_avg, acc_test_avg,end_time-start_time 
    
def plotConfusionMatrix(confusion_matrix,count,classes):
    plt.figure(figsize=[12,10])
    # create dataframe to hold the matrix
    df = pd.DataFrame(100 * confusion_matrix / count)
    # create heatmap
    ax = sn.heatmap(df, vmin=0, vmax=100, cmap='turbo', annot=True, fmt='.2f', annot_kws={'size':6}, linewidths=0.5, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('True')
    ax.set_ylabel('Prediction')
    ax.set_title('Confusion Matrix for Baseline Model')
    plt.show()

def printPrecisioAndRecall(confusion_matrix,count,loader_test):
    # create dataframe to hold the matrix
    df = pd.DataFrame(100 * confusion_matrix / count)
    # calculate precision and recall for baseline model
    matrix = np.array(df)
    # calculate TP, FP, FN and store in arrays
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    # calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print('Precision and Recall for each class of Baseline Model')
    for i in range(29):
        print(f'class {i+1}: precision = {precision[i]:.2f}, recall = {recall[i]:.2f}')

    lowest_precision_ind = np.argmin(precision)
    lowest_precision_label = get_class(loader_test, lowest_precision_ind)
    lowest_recall_ind = np.argmin(recall)
    lowest_recall_label = get_class(loader_test, lowest_recall_ind)

    print(f'Lowest precision is {precision[lowest_precision_ind]} on class {lowest_precision_ind+1}, label {lowest_precision_label}')
    print(f'Lowest recall is {recall[lowest_recall_ind]} on class {lowest_recall_ind+1}, label {lowest_recall_label}')
    
def plotPrecisionAndRecall(y_test,y_score,classes,loader_test):
    # Concatenate all the collected data
    y_score = torch.cat(y_score).cpu().numpy()
    y_test = torch.cat(y_test).cpu().numpy()
    y_test = [get_class(loader_test, i) for i in y_test]
    y_test = label_binarize(y_test, classes=classes)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(29):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        
    _, ax = plt.subplots(figsize=(10, 10))
    for i in range(29):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i])
        display.plot(ax=ax, name=f'class {i+1}')
    ax.set_xlim([0.0, 1])
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.set_title('Precision-Recall Curve for Baseline Model')
    plt.show()

