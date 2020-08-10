# GluxkindAssignment
Submission for the binary classifier 



# Overview

This repository contains files for a binary classifier that can classify Araneae and Coleoptera. I decided to go with a simplified AlexNet architecture (Alexnet with the last two convolution and pooling layers removed). I chose it as it is a proven and modern architecture that is not too resource intensive (compared to VGG-16 and other more complicated archs). After completing the model, I ran some training trials to tune the hyperparameters based on validation accuracy. The main bottleneck here was time, as I was not usig a GPU (CUDA wouldn't recognize my GPU). Therefore, there is still alot of possible improvements/tuning that can be done to the hyper-parameters, optimizer choice, and so on.



# Scripts

  * config.py: This script is used to centralize all the parameters/variable to be modified by the tester. That includes:
    - Hyper-parameters: Some hyper-parameters for the CNN, including learning rate, epoch numb, batch size, etc..
    - Paths: includes pathes to the datasets. Python/Conda sometimes gives errors with relative pathes, so using direct pathes is best
  
  * BinaryClassifier.py: This script contains the actual CNN architecture along with the training and testing functions.
    - train(): This is used to train the CNN and displays the resulting training & validation losses and accuracy for every 10 batches 
    - test(): This function tests the trained model against the test dataset and outputs the overall accuracy
    



# How to use

To try this CNN, simply clone the repo on your local machine and make sure to change the paths in config.py. The BinaryClassifier script will automatically search for and use a CUDA compatible gpu if available. 

Note: I rearranged the dataset from kaggle into train-test-validation (roughly 60-20-20 split). I uploaded the rearranged the dataset as a zipped file, as github would not allow me to upload folders that big. dataset can be found here: https://www.dropbox.com/sh/m7wih4f46qeerrj/AAAJXL9NgnY8B9f9_fzoKp-Qa?dl=0


# Results

Overall, I ran into a number of problems throughout the different trials, these included:
* Training accuracy not increasing and fluctuating too heavily
* Validation accuracy very high in first epoch, higher than training accuracy 
* Loss not decreasing

In addition to tuning the hyper-parameters to try and address the problems and improve the model, I also decided to try training with and without image augmentations (rotate, flip, crop, etc...). The training process with augmentation yielded higher validation accurracy values.

The final chosen model had the following parameters:
* image resolution: 128x128
* batch size: 80
* epochs: 1000+
* Augmentations: True
* Optimizer: Adam
* Loss Function: Cross Entropy
* CNN Architecture: Simplified AlexNet
* Learning Rate: 0.005


# Note: even with these parameters, the model seems to have a hard time learning. I am currently trying a different architecture


# Future ideas/improvement

Manually inspecting the dataset, it seems that many of the pictures have a lot of background (the actual insect only makes a small percentage of the picture). That means there could be some pre-processing done to reduce the background noise and make it easier for the cnn to identify features in the actual insect. One idea would be to identify all the most prelevant background colors (likely green due to leaves or brown due to dirt) and then somehow filter those out before feeding the dataset into the model.
