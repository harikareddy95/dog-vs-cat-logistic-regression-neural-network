## Problem statement:
Using a free Dogs vs. Cats Kaggle [dataset](https://www.kaggle.com/c/dogs-vs-cats/data). The dataset contains two folders namely, train and test1. Each folder has various images of cats and dogs. The model should classify each image as cat or dog, since the images in the training folder are given with the true label, it is a supervised learning. 

Model should take number of inputs and give one single output(cat/dog). Using neural network, s simple classifier like logistic regression(binary classification method) can do this task.
## Steps for 


#### Pre-processing the dataset
Images in both the folders are of various sizes, so we should define our image size. 

    rows = 64 #height of the image
    cols = 64 #width of the image
    channels = 3 #RGB-Red, Blue, Green
    
Assigning dataset folder locations to the variables

    train_dir = 'FOLDER_LOCATION'
    test_dir = 'FOLDER_LOCATION'
    
Reading all the image loactions and storing them as a list

    import os
    train_images = [train_dir+i for i in os.listdir(train_dir)]
    test_images = [test_dir+i for i in os.listdir(test_dir)]
    
Creating a function which takes imput as file path and gives an output resized image

    def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (rows,cols), interpolation=cv2.INTER_CUBIC)
    
Function which generates the data 
