# **Behavioral Cloning Project**

The goal of this project is to perform end to end learning for steering a car in a driving simulator based on camera images.
To achieve this, the following steps were performed:
* Use the driving simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model 
* Test that the model successfully drives around the track without leaving the road


[//]: # (Image References)
[image_orig]: ./recorded_images.png "Recorded images (left, center and right camera)"


## Recording Data

The training and validation data consists of recorded human driving behavior from the simulator. 
The data contains one lap around the track in each direction while driving in the middle of the road and several recovery scenarios, i.e. scenarios for steering the car back to the center from the left or right side of the road.

At every frame of the simulation images from three cameras mounted on the vehicle (left, center and right camera) as well as the current steering angle are recorded and stored.
The following three images are an example of what each camera sees:
![alt text][image_orig]

All in all 8271 frames have been recorded which leads to 24813 individual images. 


## Preprocessing

normailzation
cut out
mirror
left right steering angle adjustment


## Model Architecture 

As the project has many similarities (steering a car towards the center of a lane/road) to the famous paper "End to End Learning for Self-Driving Cars" by Bojarski and Del Testa, their model architecture is used here as a basis. 
An additional fully connected layer has been added to the end of the network, to output only one quantity (steering angle). 
Furthermore two dropout layers have been added to avoid overfitting and allow for better generalization (model.py lines 21).
Tanh activation functions have been used in all fully connected layers.

The final model structure looks like this:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 78x320x3 normalized color image   							                 | 
| Convolution 5x5 | 2x2 stride, output = 28x28x10 	|
| ReLU					       |												                                     |
| Max pooling	2x2 | 2x2 stride, valid padding, output = 14x14x10 				 |
| Convolution 5x5 | 1x1 stride, valid padding, output = 10x10x20 	|
| ReLU					       |												                                     |
| Max pooling	2x2 | 2x2 stride, valid padding, output = 5x5x20 				   |
| Fully connected		| input = 500, output = 120        					|
| ReLU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 120, output = 84        					|
| ReLU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 84, output = 60        					|
| ReLU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 60, output = 43        					|
| Softmax				     |         									|
 
TODO update table

<!--The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 10-16). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.-->

## Parameter Tuning and Training Strategy

As the model was trained using an Adam optimizer, the learning rate was not tuned manually (model.py line 25). 
The mean squared error over the samples was chosen as loss function for the regression problem of predicting the continuous value of the steering angle.

I experimented with using additional fully connected layers but the validation accuracy did not improve further. 
Instead of using ReLu activation functions in the fully connected layers, I used tanh activations. 
In my tests ReLu activations had much worse convergence properties and often predicted a steering angle near zero even after training for multiple epochs. Tanh seems to better capture the nature of predicting a value of the steering angle between -1 and 1.

0.25 TODO

## Model Evaluation

After tuning the parameters the model is now able to follow the track without leaving the road and also to recover from artificially induced bad situations (car nearly leaving the track) in a robust manner.
As a way to make the movement of the car more natural and fluid I added a low-pass filter to the steering angle values. 
The car now behaves less jittery and mimics human driving behavior in a better way.


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
