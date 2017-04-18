import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline

# parameters 
batch_size = 32
steering_correction = 0.25



def normalize_image(img):
    return img/255.0 - 0.5

def crop_image(img):
    return img[62:140, :]

def mirror_image(img):
    return np.fliplr(img)  
	

# create samples from simulation data
data_dirs = os.listdir("./sim_data/")
samples = []

for directory in data_dirs:  # loop over all directories with recorded data

    with open('./sim_data/' + directory + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)        
       
        for line in reader:    
            
            img_path_center = './sim_data/' + directory + '/IMG/'+line[0].split('\\')[-1]
            img_path_left = './sim_data/' + directory + '/IMG/'+line[1].split('\\')[-1]
            img_path_right = './sim_data/' + directory + '/IMG/'+line[2].split('\\')[-1]
            
            steering_angle = float(line[3])    
			
			# A "sample" here contains the paths to the three images (center, left and right camera image) and the recorded steering angle
            samples.append(((img_path_center, img_path_left, img_path_right), steering_angle))

            
# split samples into training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# load images from hard drive and append them to the data lists
def append_sample(sample, images, steering_angles):
    global steering_correction
    
    for ind, img_path in enumerate(sample[0]):
        
        image = cv2.imread(img_path)             
        image = crop_image(image)               
        image = normalize_image(image)
        image_mirrored = mirror_image(image)
        
        steering_angle = sample[1]
        
        if ind == 1: # image is from left camera
            steering_angle += steering_correction                    

        elif ind == 2: # image is from right camera
            steering_angle -= steering_correction  
            
        images.append(image)
        steering_angles.append(steering_angle)
        
        images.append(image_mirrored)
        steering_angles.append(-steering_angle)
    
	
# define a generator to continuously loop over the sample data 
def generator(samples):
    num_samples = len(samples)
    global batch_size
    
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]  
            
            images = []
            steering_angles = []
            
            for sample in batch_samples:
                append_sample(sample, images, steering_angles)                
            
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)


# generators for training the network
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import backend as K

# model is based on the paper "End to End Learning for Self-Driving Cars" by Bojarski and Del Testa 
# two dropout layers and another fully cnnected layer have been added
# input dimensions have been changed 
model = Sequential()
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu', input_shape=(78, 320, 3)))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# use mean squared error as loss function for the steering angle (regression problem) 
model.compile(loss='mse', optimizer='adam')

print('model successfully created!')


num_epochs = 10

# train the model
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch = len(train_samples),                                      
                                     validation_data = validation_generator, 
                                     nb_val_samples = len(validation_samples),                        
                                     nb_epoch = num_epochs, verbose = 1)

model.save('model.h5')

print('model saved!')