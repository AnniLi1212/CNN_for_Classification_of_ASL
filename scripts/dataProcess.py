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
def getDataLoader(folder,batch_size):
    # load the dataset
    dataset = ImageFolder(root=folder)
    print("dataset")
    # number of images in the dataset
    dataset_size = len(dataset)

    # define split sizes; avoid rounding errors
    size_train = int(dataset_size / 10 * 7)
    size_val = int(dataset_size / 10 * 1.5)
    size_test = int(dataset_size / 10 * 1.5)

    # split the dataset
    dataset_train, dataset_val, dataset_test = random_split(dataset, 
                                                            [int(size_train), 
                                                            int(size_val), 
                                                            int(size_test)])
    # double check the size
    print(f'training dataset size: {len(dataset_train)}')
    print(f'validation dataset size: {len(dataset_val)}')
    print(f'test dataset size: {len(dataset_test)}')
    
    # apply transforms to datasets
    dataset_train.dataset = ImageFolder(root=folder, transform=transforms_train)

    dataset_val.dataset = ImageFolder(root=folder, transform=transforms_valtest)
    dataset_test.dataset = ImageFolder(root=folder, transform=transforms_valtest)
    
    # dataLoader for each dataset
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return loader_train,loader_val,loader_test
    
      
# function to normalize images
def normalize_image(tensorimage):
    image_min = tensorimage.min()
    image_max = tensorimage.max()
    tensorimage.clamp_(min=image_min, max=image_max)
    tensorimage.add_(-image_min).div_(image_max - image_min + 1e-5)
    return tensorimage

# function to add gaussian noise
def add_noise(tensorimage, mean, std):
    return tensorimage + torch.randn(tensorimage.size()) * std + mean

# for training transforms
transforms_train = v2.Compose([
    # transform to tensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # normalize
    normalize_image,
    # add noise
    v2.Lambda(lambda x: add_noise(x, mean=0.0, std=0.2)),
    # randomly flip the image horizontally
    v2.RandomHorizontalFlip(),
    # randomly rotate the image up to 15 degrees
    v2.RandomRotation(15),
    # scale and crop the image
    v2.RandomResizedCrop(size=(200, 200), scale=(0.8, 1.0), antialias=True),
])

# for val and test transforms
transforms_valtest = v2.Compose([
    # transform to tensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # normalize
    normalize_image,
    # crop the image
    v2.Resize(size=(200, 200), antialias=True)
])

# function to get class name from numeric label
def get_class(dataloader, label):
    # define the class to label dictionary
    class2label = dataloader.dataset.dataset.class_to_idx
    # revert the dictionary
    label2class = {v: k for k, v in class2label.items()}

    # return the corresponding class name and print error if label is undefined
    return label2class.get(label, 'label not found')

def plot_images(dataloader, num_display):
    fig = plt.figure(figsize=(16, 6))
    
    # fetch a batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    # batch size
    n = len(images)
    
    # define number of rows and cols in the plot
    rows = int(math.ceil(math.sqrt(num_display) / 1.5))
    cols = int(math.ceil(num_display / rows))
    
    # create plots
    for i in range(min(num_display, n)):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        # convert image to np array
        image_np = np.clip(images[i].numpy(), 0, 1)
        # transpose the array and show the image
        plt.imshow(np.transpose(image_np, (1, 2, 0)))
        # get the numeric label
        label = labels[i].item()
        # convert to class name
        classname = get_class(dataloader, label)
        ax.set_title(f'class: {classname}')
        
    plt.show()