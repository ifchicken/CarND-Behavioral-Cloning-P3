# **Behavioral Cloning**

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/LeNet5_model.JPG "LeNet5"
[image2]: ./report/NV_CNN.jpg       "CNN"
[image3]: ./report/center_2017_03_17_23_54_37_619.jpg "ex Image"
[image4]: ./report/center_2017_03_17_23_55_10_171.jpg "center Image"
[image5]: ./report/left_2017_03_17_23_55_10_171.jpg "left Image"
[image6]: ./report/right_2017_03_17_23_55_10_171.jpg "right Image"
[image7]: ./report/center_2017_03_17_23_56_54_524.jpg "easy_failed Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* run1.mp4 to show the model driving in autonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track one by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the Nvidia convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

There are two models I tried in this project, the first one is LeNet5 model. However, the result is not good enough to pass the test. I guess it's becasue LeNet5 model is more suitable for the classification than regression.

Here is the model:

![alt text][image1]

Here is the code:

```python
def LeNet5(input_shape):
    '''
    LeNet model with keras (model.py line 103-127)
    '''
    model = Sequential()
    #normalized using Keras lambda layer
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape)) #(160,320,3)
    #crop to filter no-use info
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    #Lenet model
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model
```

The second model I tried to use is Nvidia CNN structure. It includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. [Here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) is the reference website of this model.

Here is the model:

![alt text][image2]

Here is the code:

```python
def NV(input_shape):
    '''
    Nvidia CNN model with keras (model.py line 129-161)
    '''
    model = Sequential()
    #normalized using Keras lambda layer
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape)) #(160,320,3)

    #crop to filter no-use info
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    #CNN model
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    #ouptut is 1 because it's regression, not classifier
    model.add(Dense(1))
    return model
```

#### 2. Attempts to reduce overfitting in the model

I split the whole data to 80% training and 20% validation by using the code:
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

The overfitting is not really severe. Tehrefore, I didn't use any dropout layer in this model. For the future improvement, I will try to use some dropout layer.

the code for dropout layer in keras is like: 
model.add(Dropout(0.5))
 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 175).

model.compile(loss='mse', optimizer='adam')

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a all three camera(center, left and right) to train my car. In this way, the vehicle can learn how to steer if the car drifts off to the left or the right. Another way I know is recording recovery driving from the sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because it also learns from images. However, the result is not quite good andd I guess the reason is bacause this model is only suitable for classification, not regression. Therefore, I try to use Nvidia CNN structure.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I didn't see severe overfitting therefore I didn't use any dropout layer.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I train the vehicle again and slow the speed to get more train data at the fell off place. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		| Layer name	|     Description                                       | 
|:---------------------:|:--------------|:-----------------------------------------------------:| 
| Input         		|           	| 160x320x3 RGB image                                   |
| Normalized       		| Lambda layer  | 160x320x3 RGB image                                   |
| Cropping        		|           	| 160x320x3 RGB image cropping=((50,20), (0,0))         | 
| Convolution 5x5     	| conv1     	| with subsample=(2,2) and RELU, output has 24 layer    |
| Convolution 5x5	    | conv2     	| with subsample=(2,2) and RELU, output has 36 layer	|
| Convolution 5x5	    | conv3     	| with subsample=(2,2) and RELU, output has 48 layer	|
| Convolution 5x5	    | conv4     	| with subsample=(2,2) and RELU, output has 64 layer	|
| Convolution 5x5	    | conv5     	| with subsample=(2,2) and RELU, output has 64 layer	|
| Flatten       		| fc0       	| reshape outputs                                       |
| Fully connected		| fc1       	| outputs 1x100                                         |
| Fully connected		| fc2       	| outputs 1x50                                          |
| Fully connected		| fc3       	| outputs 1x10                                          |
| Output        		| fc4       	| outputs 1x1                                           |
|						|           	|                                                       |



Here is a visualization of the architecture:

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. The first laps with slow speed drive to make the vehicle staying on the center most of time. The second laps with max speed to get data with large steer number around the left/right corner.

Here is an example image of center lane driving:

![alt text][image3]

More, I also used all three camera(center, left and right) and flipped images to train my car. In this way, the vehicle can learn how to steer if the car drifts off to the left or the right. Here are an example of 3 camera images:

![alt text][image4]
![alt text][image5]
![alt text][image6]

After the collection process, I had X number of data points. I then preprocessed this data by normalized and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model, and I used an adam optimizer so that manually training the learning rate wasn't necessary.


I saved the model which can pass 80% tarck, then look at the spot which fell off the track. Then I recorded more laps with slow speed at those fell off region to get more data. I found out the region easily to fail is at the connecting point of 2 different side material. Here is the example image:

![alt text][image7]

Then, I train the model again based on the model with 80% track passed. In that case, it's like fine tune the model with more data at fell off spot. Finally, it can pass all track.


### Future Improvements:
The model can not pass the track 2, and in some training, I did see some overfitting. i'll try to use drop out method to improve the model in the futrure.

