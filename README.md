# **Behavioral Cloning Project**

The goal of this project is to perform end to end learning for steering a car in a driving simulator based on camera images.
To achieve this, the following steps were performed:
* Use the driving simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model 
* Test that the model successfully drives around the track without leaving the road


[//]: # (Image References)
[images_orig]: ./images_orig.png "Recorded images (center, left and right camera)"
[images_cropped]: ./images_cropped.png "Images cropped to exclude unnecessary data"
[images_mirrored]: ./images_cropped_mirrored.png "Mirrored images"
[loss_function]: ./training_loss.png "MSE loss for training and validation data"


## Recording Data

The training and validation data consists of recorded human driving behavior from the simulator. 
The data contains one lap around the track in each direction while driving in the middle of the road and several recovery scenarios, i.e. scenarios for steering the car back to the center from the left or right side of the road.

At every frame of the simulation images from three cameras mounted on the vehicle (left, center and right camera) as well as the current steering angle are recorded and stored.
The following three images are an example of what each camera sees:
![alt text][images_orig]

All in all 8271 frames have been recorded which leads to 24813 individual camera images. 


## Preprocessing

The images from the left and right camera show the road as if the car was shifted a bit to either side of the road. 
The steering angle corresponding to these images therefore gets adjusted by adding a constant steering angle offset to the actual recorded steering angle (model.py lines 65 and 68). 

As can bee seen in the example images from the last section there is lots of "unnecessary" data (e.g. hills next to the road) that tells us nothing about the course of the road. 
All images therefore get cropped vertically to only contain the more interesting features (model.py line 20):
![alt text][images_cropped]

An easy way to double the amount of data available is simply mirroring the images and switching the sign of their corresponding steering angle (model.py line 23) which in the end leads to 49626 samples (image + steering angle):
![alt text][images_mirrored]

The last preprocessing step consists of normalizing the pixel values of each color channel to the interval [-1, 1] (model.py line 17).  

## Model Architecture 

As the goal of this project (steering a car towards the center of a lane/road) is similar to the problem statement in the reputable paper "End to End Learning for Self-Driving Cars" by Bojarski and Del Testa, their model architecture is used here as a basis. 
An additional fully connected layer has been added to the end of the network, to output only one quantity (steering angle). 
Furthermore two dropout layers have been added to avoid overfitting and allow for better generalization.
Tanh activation functions have been used in all fully connected layers.

The final model structure of the convolutional neural network looks like this (model.py lines 112 - 124):

| Layer         		|     Description	        					| 
|:---------------------:|:------------------------------:| 
| Input         		| 78x320x3 normalized color image   	| 
| Convolution 5x5 | 2x2 stride, output = 37x158x24 	|
| ReLU					       |			                               |
| Convolution 5x5 | 2x2 stride, output = 17x77x36 	|
| ReLU					       |    |
| Convolution 5x5 | 2x2 stride, output = 7x37x48 	|
| ReLU					       | |
| Convolution 3x3 | 1x1 stride, output = 5x35x64 	|
| ReLU					       | |
| Convolution 3x3 | 1x1 stride, output = 3x33x64 	|
| ReLU					       | |
| Flatten        |  	|		       
| Fully connected		| input = 6336, output = 100  	|
| Tanh					       |			  |
| Dropout					       |		    |
| Fully connected		| input = 100, output = 50   	|
| Tanh					       |			       |
| Dropout					       |		   |
| Fully connected		| input = 50, output = 10  	|
| Tanh					       |				   |
| Fully connected		| input = 10, output = 1    	|
| Tanh				     |         									|


## Parameter Tuning and Training Strategy

The mean squared error (mse) over samples was chosen as a loss function for the regression problem of predicting the continuous value of the steering angle.
As the model was trained using an Adam optimizer, the learning rate was not adapted manually (model.py line 127). 

The data has been split into a training (80%) and a validation (20%) data set (model.py line 48). 
After training the network for 10 epochs on the training data the validation accuracy stopped to decrease which indicated that further training was not necessary:
![alt text][loss_function]

I experimented with using additional fully connected layers but the validation accuracy did not improve further. 
Instead of using ReLU activation functions in the fully connected layers, I used tanh activations. 
In my tests ReLU activations had worse convergence properties and often predicted a steering angle near zero even after training for many epochs. Tanh seems to better capture the nature of predicting a value of the steering angle between -1 and 1.

The steering angle offset for images from the left and right camera has been set to 0.25. 
Too small values led to the car drifting off the road in narrow curves while too high values made the car unstable on a straight road.


## Model Evaluation

After tuning the parameters the model is now able to follow the track without leaving the road and also to recover from artificially induced bad situations (car nearly leaving the track) in a robust manner.
As a way to make the movement of the car more natural and fluid I added a low-pass filter to the steering angle values. 
The car now behaves less jittery and mimics human driving behavior in a better way.


