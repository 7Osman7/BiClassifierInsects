# GluxkindAssignment
Submission for the binary classifier 



# Overview

This repository contains files for a binary classifier that can classify Araneae and Coleoptera. I decided to go with a simplified AlexNet architecture (Alexnet with the last two convolution and pooling layers removed). I chose it as it is a proven and modern architecture that is not too resource intensive (compared to VGG-16 and other more complicated archs). After completing the model, I ran some training trials to tune the hyperparameters based on validation accuracy. The main bottleneck here was time, as I was not usig a GPU (CUDA wouldn't recognize my GPU).



# Scripts

  * config.py: This script is used to centralize all the parameters/variable to be modified by the tester. That includes:
    - Hyper-parameters: Some hyper-parameters for the CNN, including learning rate, epoch numb, batch size, etc..
    - Paths: includes pathes to the datasets. Python/Conda sometimes gives errors with relative pathes, so using direct pathes is best
  
  * BinaryClassifier.py: This script contains the actual CNN architecture along with the training and testing functions.
    - train(): This is used to train the CNN and displays the resulting training & validation losses and accuracy for every 10 batches 
    - test(): This function tests the trained model against the test dataset and outputs the overall accuracy
    



# How to use

To try this CNN, simply clone the repo on your local machine and make sure to change the paths in config.py. The BinaryClassifier script will automatically search for and use a CUDA compatible gpu if available. 

Note: I rearranged the dataset from kaggle into train-test-validation (roughly 60-20-20 split). I uploaded the rearranged the dataset as a zipped file


# Results

Overall, 
