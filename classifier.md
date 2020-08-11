## Problem statement:
Using a free Dogs vs. Cats Kaggle [dataset](https://www.kaggle.com/c/dogs-vs-cats/data). The dataset contains two folders namely, train and test1. Each folder has various images of cats and dogs. The model should classify each image as cat or dog, since the images in the training folder are given with the true label, it is a supervised learning. 

Model should take number of inputs and give one single output(cat/dog). Using neural network, s simple classifier like logistic regression(binary classification method) can do this task.
## Steps for 


#### Pre-processing the dataset
Images in both the folders are of various sizes, so we should define our image size. 

    rows = 64 
    cols = 64 
    channels = 3 
    
> rows:height_of_the_image, cols:width_of_the_image, channels:RGB-Red,Blue,Green 

Assigning dataset folder locations to the variables

    train_dir = 'FOLDER_LOCATION'
    test_dir = 'FOLDER_LOCATION'
    
Reading all the image loactions and storing them as a list

    import os
    train_images = [train_dir+i for i in os.listdir(train_dir)]
    test_images = [test_dir+i for i in os.listdir(test_dir)]
    
Creating a function which takes imput as file path and gives an output resized image

    import cv2
    def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (rows,cols), interpolation=cv2.INTER_CUBIC)
    
Function which generates the data x(parameters) and y(output) taking list of images as input. The output of each images is 1 or 0, considering 1 as dog and 0 as cat.

    import numpy as np
    def prepare_data(images):
        m = len(images)
        x = np.zeros((m, rows, cols, channels), dtype=np.uint8) #image
        y = np.zeros((1,m)) #output
        for i, image_file in enumerate(images):
            x[i,:] = read_image(image_file)
            if 'dog' in image_file.lower(): 
                y[0,i] = 1
            elif 'cat' in image_file.lower():
                y[0,i] = 0
    return x, y 
    
> all the images in the folder are named as cats and dogs

Preparing data for training and testing data sets

    train_set_x, train_set_y = prepare_data(train_images)
    test_set_x, test_set_y = prepare_data(test_images)
    
Re-shaping images of shape (rows, cols, channels) into single vectors of shape (rows*cols*channels,1). Each column represents a flattened image. 

    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], rows*cols*channels).T
    test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
    
