## Problem statement:
Using a free Dogs vs. Cats Kaggle [dataset](https://www.kaggle.com/c/dogs-vs-cats/data). The dataset contains two folders namely, train and test1. Each folder has various images of cats and dogs. The model should classify each image as cat or dog, since the images in the training folder are given with the true label, it is a supervised learning. 

Model should take number of inputs and give one single output(cat/dog). Using neural network, s simple classifier like logistic regression(binary classification method) can do this task.


### Pre-processing the dataset
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

Preparing data for training and testing data sets into numpy arrays

    train_set_x, train_set_y = prepare_data(train_images)
    test_set_x, test_set_y = prepare_data(test_images)
    
Re-shaping images of shape (rows, cols, channels) into single vectors of shape (rows\*cols\*channels,1). Each column represents a flattened image. 

    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], rows*cols*channels).T
    test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
    
Printing out the shapes of the dataset

    print("train_set_x shape: "+str(train_set_x.shape))
    print("train_set_x_flatten shape: "+str(train_set_x_flatten.shape))
    print("train_set_y shape: "+str(train_set_y.shape))
    print("test_set_x shape: "+str(test_set_x.shape))
    print("test_set_x_flatten shape: "+str(test_set_x_flatten.shape))
    print("test_set_y shape: "+str(test_set_y.shape))

Output:

<table style="width:35%">
    <tr>
        <td>train_set_x shape</td>
        <td>(25000, 64, 64, 3)</td>  
    </tr>
    <tr>
        <td>train_set_x_flatten shape</td>
        <td>(12288, 25000)</td>  
    </tr>
    <tr>
        <td>train_set_y shape</td>
        <td>(1, 25000)</td>  
    </tr>
    <tr>
        <td>test_set_x shape</td>
        <td>(12500, 64, 64, 3)</td>  
    </tr>
    <tr>
        <td>test_set_x_flatten shape</td>
        <td>(12288, 12500)</td>  
    </tr>
    <tr>
        <td>test_set_y shape</td>
        <td>(1, 12500)</td>  
    </tr>
</table>
    
Common pre-processing step in machine learning is to center and standardize the dataset, means to substract the mean and divide by standard deviation of the numpy array. For picture dataset, its easy to divide every row of the dataset by 255(maximum value of a pixel channel).

    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    
Image dataset is now ready.

### Algorithm
The architecture of the simple neural network(one neuron) is shown below. Model structure is defined

Main steps for building a neural network are: 
+ Defining model structure
+ Initializing model parameters
+ Learning the parameters (loop)
    + Current loss (forward propagation)
    + Current gradient (backward propagation)
    + Update parameters (Gradient descent)
 + Predict on test data
 + Analyse and conclude
 
<img src="/images/general-architecture.png" width="650"/>

Sigmoid function decides the final output. Applying a sigmoid function, scales the decimal value to 1 or 0 depending on the value, if it is >0.5, output is 1 or else 0.

![sigmoid function](/images/sigmoid-function.png)

    def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

+ Initializing model parameters depending on the dimensions(parameters). w is weights which is a numpy array and b is the bias which is a scalar added.

        def initialize_with_zeros(dim):
            w = np.zeros((dim,1))
            b = 0
            return w,b

+ Learning the parameters with forward and backward propagation. Using weight and bias, output is predicted for a training example(A-predicted output). y being the true label, current loss is calculated using both actual and predicted labels. Total cost is calculated by summing over all training examples. Gradient is calculated in the reverse direction, cost derived by predicted label and then derived by weight and bias results in following equations. Propagate function takes in weight, bias, parameters of training images stored in x and true label vector y and gives gradients(created a dictionary for w and b) and cost are returned.
 
        def propagate(w,b,x,y):
            m = x.shape[1] #number of training examples
            #Forward propagation
            z = np.dot(w.T, x)+b
            A = sigmoid(z)
            cost = (-np.sum(y*np.log(A)+(1-y)*np.log(1-A)))/m
            #Backward propagation
            dw = (np.dot(x,(A-y).T))/m
            db = (np.sum(A-y))/m
            cost = np.squeeze(cost)
            grads = {"dw": dw,
                     "db": db}
            return grads, cost
        
+ Updating the parameters through optimize function. Inorder to update the parameters, first they should be retreived from propagate function. Weight and bias are updated according to the learning rate and retrieved gradient. Then updating them back to the dictionary. every 100 iterations records the cost.

        def optimize(w,b,x,y,num_iterations,learning_rate,print_cost=False):
            costs = []
            for i in range(num_iterations):             
                grads, cost = propagate(w,b,x,y)
                #retrieve derivates
                dw = grads["dw"]
                db = grads["db"]
                #update
                w = w-(learning_rate*dw)
                b = b-(learning_rate*db)
                #Recording costs
                if i%100 == 0:
                    costs.append(cost)
                #Print the cost every 100 training iterations
                if print_cost and i%100 == 0:
                    print("Cost after iteration %i: %f" %(i,cost))
            #Update w and b to dictionary
            params = {"w": w,
                     "b": b}
            #Update derivates to dictionary
            grads = {"dw": dw,
                    "db": db}
            return params, grads, costs
        
+ Finally, predicting on data, this function takes in final weight and bias from dictionary  and x as parameters of training and test data and returns the prediction.

        def predict(w,b,x):
            m = x.shape[1]
            y_prediction = np.zeros((1,m))
            w = w.reshape(x.shape[0],1)
      
            A = sigmoid(np.dot(w.T,x)+b)

            for i in range(A.shape[1]):
                # Convert probabilities A[0,i] to actual predictions p[0,i]
                if A[0,i] > 0.5:
                    y_prediction[[0],[i]] = 1
                else: 
                    y_prediction[[0],[i]] = 0

            return y_prediction
        
+ Putting it all together, into a model function. 

        def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
            w, b = initialize_with_zeros(x_train.shape[0])

            parameters, grads, costs = optimize(w,b,x_train,y_train,num_iterations=2000,learning_rate=0.005,print_cost=True)

            w=parameters["w"]
            b=parameters["b"]

            y_prediction_test = predict(w,b,x_test)
            y_prediction_train = predict(w,b,x_train)

            print("train accuracy: {}%".format(100-np.mean(np.abs(y_prediction_train - y_train))*100))
            print("test accuracy: {}%".format(100-np.mean(np.abs(y_prediction_test - y_test))*100))

            dict = {"costs": costs,
                    "y_prediction_test": y_prediction_test,
                    "y_prediction_train": y_prediction_train,
                    "w": w,
                    "b": b,
                    "learning_rate": learning_rate,
                    "num_iterations": num_iterations
            }
            return dict
        
+ After the final model function, we need to call the model function with necessary parameters, then we will retrieve the cost for every 100 iterations and final accuracy.

        d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3000, learning_rate = 0.003, print_cost = False)
        
 <table>
    <tr>
        <td>train accuracy</td>
        <td>52.292%</td>
    </tr>
    <tr>
        <td>test accuracy</td>
        <td>95.68%</td>
    </tr>
</table>

+Let's test an individual image called "xyz.jpeg"

![xyz.jpeg](/images/xyz.jpeg)

        image = 'xyz.jpeg'
        x=(read_image(image).reshape(1, rows*cols*channels).T)/255
        y=predict(d["w"],d["b"],x)
        print("Its a cat" if y==0 else "Its a dog")
        
 <table>
    <tr>
        <td>Its a cat</td>
    </tr>     
 </table>       
 
+ Analysing with different learning rates, such as 0.001, 0.003,... below graph explains the different costs with different learning rates. 
    + Higher learning rate results in fluctuation of costs.
    + Lower learning rate is not always best, should also check for overfitting of the model, which is usually happens when accuracy of training set is larger than test set.
    + Its optimal to choose learning rate which reduces the cost.
