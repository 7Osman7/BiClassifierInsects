#Import Pytorch libraries 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F

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
     transforms.Resize(200),
     transforms.CenterCrop((200, 200)),
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



# CNN Class. Using a simple CNN architecture with 5 conv and pool layers

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=60, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
 
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=120, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=120, out_channels=180, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1)
        
        self.drop = nn.Dropout2d(p=0.40)
        
        self.fc = nn.Linear(in_features=50 * 50 * 180, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
      
        x = F.relu(self.pool(self.conv2(x)))

        x = F.relu(self.drop(self.conv3(x)))

        x = F.relu(self.drop(self.conv4(x)))

        x = F.relu(self.drop(self.conv5(x)))

        x = F.dropout(x, training=self.training)
        x = x.view(-1, 50*50 * 180)
        x = self.fc(x)
        return torch.sigmoid(x)


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
        
        model.Training = True  
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


        model.Training = False  #To prevent dropout
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

                #if (j + 1) % 10 == 0:
                print('Validation: Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, epochs_num, j + 1, total_step_val, loss_val.item(),
                            (correct / total) * 100))
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'cnn_bi_classify3.ckpt')

    print(*loss_list)
    print(*acc_list)
    print(*loss_val_list)
    print(*acc_val_list)




#Test function. Used to assess the performance of the selected model
def test():
    model.load_state_dict(torch.load(MODEL_STORE_PATH + 'cnn_bi_classify3.ckpt'))
    model.eval()
    model.Training = False  #To prevent dropout
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

