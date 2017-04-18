# **Behavioral Cloning Project**

The goal of this project is to perform end to end learning for steering a car in a driving simulator based on camera images.

To achieve this, the following steps have to be performed:
* Use the driving simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model 
* Test that the model successfully drives around the track without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Recording Data

The training and validation data consists of recorded human driving behavior in the simulator. 
The data contains one lap around the track in each direction while driving in the middle of the lane and several recovery scenarios, i.e. scenarios for steering the car back to the center from the left or right side of the road.
At every frame of the simulation images from three cameras mounted on the vehicle (left, center and right camera) and the steering angle are captured and stored.

## Preprocessing


## Model Architecture 

As the project has many similarities to the famous paper "End to End Learning for Self-Driving Cars" by Bojarski and Del Testa, their model architecture is used here as a basis. An additional fully connected layer has been added to the end of the network, to output only one quantity (steering angle). Also two dropout layers have been added to avoid overfitting and allow for better generalization (model.py lines 21).

The final model structure looks like this:
TODO
TODO -> tanh instead of relu
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 
The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

## Parameter Tuning and Training Strategy

As the model was trained using an Adam optimizer, the learning rate was not tuned manually (model.py line 25). 
The mean squared error over the samples was chosen as loss function for the regression problem of predicting the continuous value of the steering angle.

I experimented with using additional fully connected layers but the validation accuracy did not improve further. 
Instead of using ReLu activation functions in the fully connected layers, I used tanh activations. 
In my tests ReLu activations had much worse convergence properties and often predicted a steering angle near zero even after training for multiple epochs. Tanh seems to better capture the nature of predicting a value of the steering angle between -1 and 1.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
