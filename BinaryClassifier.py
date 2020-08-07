#Import Pytorch libraries 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets

#Importing other useful libraries and files
import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from config import *



#Hyper-parameters
epochs_num = EPOCHS
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE



# Check for and use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



# Define image transforms. Could modify to add augmentations here

#No augmentation
#transform_set = transforms.Compose([transforms.RandomResizedCrop(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#with augmentation
transform_set = transforms.Compose([
     transforms.Resize(128),
     transforms.CenterCrop((124, 124)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])



# Define train validation and test databases
path_train = TRAIN_PATH                               
path_validation = VALID_PATH                          
path_test = TEST_PATH                                 


train_dataset = datasets.ImageFolder(root = path_train,transform = transform_set)

validation_dataset = datasets.ImageFolder(root = path_validation, transform = transform_set)

test_dataset = datasets.ImageFolder(root = path_test, transform = transform_set)



# Load databases
train_load = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)    

validation_load = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False) 

test_load = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)



# CNN Class. Using a simplified AlexNet Architecture. Original (unsimplified) implementation at 
# https://github.com/pytorch/vision/blob/master/torchvision/models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() 
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



#Creating instance & setting the loss and optimizer functions
model = CNN()
model = model.to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#Training and Validation function
def train():
    total_step_train = len(train_load)
    total_step_val = len(validation_load)
    loss_list = []
    acc_list = []
    loss_val_list = []
    acc_val_list = []
    for epoch in range(epochs_num):
        model.train()
        for i, (images, labels) in enumerate(train_load):
            #Send inputs to our device
            images = images.to(device)
            labels = labels.to(device)

            # Run the forward pass
            outputs = model(images)
            loss = loss_fun(outputs, labels)
            loss_list.append(loss.item())

            # Run the optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 10 == 0:
                print('Training: Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, epochs_num, i + 1, total_step_train, loss.item(),
                            (correct / total) * 100))

        model.eval()
        with torch.no_grad():  #disable autograd as we are in validation
            correct = 0
            total = 0
            for j, (images, labels) in enumerate(validation_load):

                #Send inputs to our device
                images = images.to(device)
                labels = labels.to(device)

                #Loss
                outputs = model(images)
                loss_val = loss_fun(outputs, labels)
                loss_val_list.append(loss_val.item())


                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc_val_list.append(correct / total)

                if (j + 1) % 10 == 0:
                    print('Validation: Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch + 1, epochs_num, j + 1, total_step_val, loss_val.item(),
                                (correct / total) * 100))
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'cnn_bi_classify.ckpt')

    print(*loss_list)
    print(*acc_list)
    print(*loss_val_list)
    print(*acc_val_list)




#Test function. Used to assess the performance of the selected model
def test():
    model.load_state_dict(torch.load(MODEL_STORE_PATH + 'cnn_bi_classify.ckpt'))
    model.eval()
    with torch.no_grad():  #disable autograd as we are in validation
        correct = 0
        total = 0
        
        for k, (images, labels) in enumerate(test_load):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #Print result
        print('Test:  Accuracy: {:.2f}%' .format((correct / total) * 100))



#Run 
train()
test()

