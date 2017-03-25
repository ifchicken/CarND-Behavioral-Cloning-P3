import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

"""	
images = []
measurements = []
for line in lines:
	#center, left, right, steer, throttle, break, speed
	#source_path = line[0]
	#filename = source_path.split('/')[-1]
	#current_path = './data/IMG/' + filename
	center_path = './data/IMG/' + line[0].split('/')[-1]
	left_path = './data/IMG/' + line[1].split('/')[-1]
	right_path = './data/IMG/' + line[2].split('/')[-1]
	#image = cv2.imread(center_path)
	img_center = cv2.imread(center_path)
	img_left = cv2.imread(left_path)
	img_right = cv2.imread(right_path)
	#images.append(image)
	#images.extend(img_center, img_left, img_right)
	images.append(img_center)
	images.append(img_left)
	images.append(img_right)
	
	measurement = float(line[3])
	correction = 0.2
	steer_left = measurement + correction
	steer_right = measurement - correction
	measurements.append(measurement)
	#measurements.extend(measurement, steer_left, steer_right)
	measurements.append(steer_left)
	measurements.append(steer_right)
"""



def flip(images, measurements):
	augmented_images, augmented_measurements= [], []
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image, 1))
		augmented_measurements.append(measurement * -1.0)
	return augmented_images, augmented_measurements

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: #loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset + batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				center_name = './data/IMG/' + batch_sample[0].split('/')[-1]
				left_name = './data/IMG/' + batch_sample[1].split('/')[-1]
				right_name = './data/IMG/' + batch_sample[2].split('/')[-1]
				img_center = cv2.imread(center_name)
				img_left = cv2.imread(left_name)
				img_right = cv2.imread(right_name)
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)

				center_angle = float(batch_sample[3])
				correction = 0.2
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				angles.append(center_angle)
				angles.append(left_angle)
				angles.append(right_angle)
				
			images, angles = flip(images, angles)
			
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
			
#X_train = np.array(images)	
#y_train = np.array(measurements)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

def LeNet5(input_shape):
	model = Sequential()
	#normalized using Keras lambda layer
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape)) #(160,320,3)
	#crop to filter no-use info
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	
	#Lenet model
	model.add(Convolution2D(6, 5, 5, activation='relu'))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D())
	
	model.add(Convolution2D(6, 5, 5, activation='relu'))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D())
	
	model.add(Flatten())
	model.add(Dense(120))
	#model.add(Activation('relu'))
	
	model.add(Dense(84))
	#model.add(Activation('relu'))
	#ouptut is 1 because it's regression, not classifier
	model.add(Dense(1))
	return model

def NV(input_shape):
	model = Sequential()
	#normalized using Keras lambda layer
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape)) #(160,320,3)
	
	#crop to filter no-use info
	model.add(Cropping2D(cropping=((50,20), (0,0))))
	
	#CNN model
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D())
	
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D())
	
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
	
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	#model.add(Dropout(0.5))
	
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	#model.add(Activation('relu'))

	model.add(Dense(50))
	model.add(Dropout(0.5))
	#model.add(Activation('relu'))	
	model.add(Dense(10))
	model.add(Dropout(0.5))
	#ouptut is 1 because it's regression, not classifier
	model.add(Dense(1))
	return model	


shape = (160,320,3)

from keras.models import load_model
fine_tune = 0

if fine_tune:
	model = load_model('model.h5')
else:
	#model = LeNet5(shape)
	model = NV(shape)

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

import matplotlib.pyplot as plt

#history_object = 
model.fit_generator(train_generator, 
							samples_per_epoch = len(train_samples),
							validation_data = validation_generator, 
							nb_val_samples = len(validation_samples),
							nb_epoch=5)
							#, verbose=1)

#print(history_object.history.keys())

#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#ply.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save('model.h5')
model.summary()
