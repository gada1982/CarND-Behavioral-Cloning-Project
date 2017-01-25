# Model for driving a car 
# Author: gada1982
# mail: daniel@gattringer.biz

# General imports
import os
import numpy as np
import csv
import cv2
from pathlib import Path

# Imports for keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, ELU, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

# Import json to save the model
import json

def load_data_info(path_of_data):
	if not os.path.exists(path_of_data):
		print("Directory with data not found.")
		sys.exit(-1)

	# Open *.csv with logged driving data
	with open(path_of_data + '/driving_log_small.csv', 'r') as logfile:
		file_read = csv.reader(logfile, delimiter=',')
		drive_data = []
		for i in file_read:
			drive_data.append(i)

	# The log file data is available in the following type: 
	# center_image, left_image,|right_image, steering-data, throttle-data, brake-data, speed-data
	# At the moment only center_image and steering-data are used 
	# TODO: Later on the left and right image will be included -> Expand info
	drive_data = np.array( drive_data )
	drive_data_relevant = np.hstack( (drive_data[1:, 0].reshape((-1,1)), drive_data[1:,3].reshape((-1,1))))

	return drive_data_relevant

def get_normalized_image(image):
    # Change color-space from BGR to RGB
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Normalize from -1 to 1 (zero mean)
    image = image / 127.5 - 1
    return image

def print_image_data(data, number):
	# Print out some information about one image
	image_name = data[number][0]
	image_path = path_of_data + '/' + image_name
	print()
	print('Image:')
	print(image_path)
	image = cv2.imread(image_path)
	print('Shape of the image:')
	print(image.shape)
	print('Steering angle for the image:')
	print(data[number][1])

def get_image_shape(data, number):
	# Print out some information about one image
	image_name = data[number][0]
	image_path = path_of_data + '/' + image_name
	image = cv2.imread(image_path)
	image_shape = image.shape
	return image_shape

def model_test():
	new_size_row, new_size_col, ch = get_image_shape(drive_data_relevant, 0)
	input_shape = (new_size_row, new_size_col, 3)
	filter_size = 3
	pool_size = (2,2)
	model = Sequential()
	model.add(Convolution2D(3,1,1,
                        border_mode='valid',
                        name='conv0', init='he_normal', input_shape=input_shape))
	model.add(Convolution2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv1', init='he_normal'))
	model.add(ELU())
	model.add(Convolution2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv2', init='he_normal'))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Convolution2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv3', init='he_normal'))
	model.add(ELU())

	model.add(Convolution2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv4', init='he_normal'))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Convolution2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv5', init='he_normal'))
	model.add(ELU())
	model.add(Convolution2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv6', init='he_normal'))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(512,name='hidden1', init='he_normal'))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(64,name='hidden2', init='he_normal'))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(16,name='hidden3',init='he_normal'))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(1, name='output', init='he_normal'))
	return model


def model_nvidia_gada():
    # TODO: Check if model is okay -> Paper!!!
    model = Sequential()

    row, col, ch = get_image_shape(drive_data_relevant, 0)
    print('Expected shape: ',row, col, ch)
    model.add(Convolution2D(24, 5, 5, input_shape = (row, col, ch), subsample = (2, 2), border_mode = "valid", activation = 'relu', name = 'conv1'))
    model.add(Convolution2D(36, 5, 5, subsample = (2, 2), border_mode = "valid", activation = 'relu', name = 'conv2'))
    model.add(Convolution2D(48, 5, 5, subsample = (2, 2), border_mode = "valid", activation = 'relu', name = 'conv3'))
    model.add(Convolution2D(64, 3, 3, subsample = (1, 1), border_mode = "valid", activation = 'relu', name = 'conv4'))
    model.add(Convolution2D(64, 3, 3, subsample = (1, 1), border_mode = "valid", activation = 'relu', name = 'conv5'))
    
    model.add(Flatten())
    model.add(Activation('relu'))

    model.add(Dense(1164))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(100, name = 'fc1'))
    model.add(Dense(50, name = 'fc2'))
    model.add(Dense(10, name = 'fc3'))
    model.add(Dense(1, name = 'output'))

    return model

def generate_train_data(data):
	# Generate training data
	while 1:
		x, y = generate_train_data_int(data, printinfo = 0)
		yield x, y

def generate_valid_data(data):
	# Generate validation data
	# TODO: Send all data for validation
	while 1:
		x, y = generate_valid_data_int(data, printinfo = 0)
		yield x, y
	
def generate_train_data_int(data, printinfo):
	# TODO: Preprocessing
	i = np.random.randint(len(data))
	image_name = data[i][0]
	image_path = path_of_data + '/' + image_name
	if printinfo == 1:
		print()
		print('Image to generate:')
		print(image_path)
	x = cv2.imread(image_path)
	x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
	y = data[i][1]
	y = np.array([[y]])
	return x, y

def get_single_validation_data(data,number):
	# TODO: Preprocessing
	i = number
	image_name = data[i][0]
	image_path = path_of_data + '/' + image_name
	print()
	print('Single image to test:')
	print(image_path)
	x = cv2.imread(image_path)
	x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
	y = data[i][1]
	print('Steering angle y: ', y)
	print()
	y = np.array([[y]])
	return x	

def generate_valid_data_int(data, printinfo):
	# TODO: Preprocessing
	i = np.random.randint(len(data))
	image_name = data[i][0]
	image_path = path_of_data + '/' + image_name
	if printinfo == 1:
		print()
		print('Image to generate:')
		print(image_path)
	x = cv2.imread(image_path)
	x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
	y = data[i][1]
	y = np.array([[y]])
	return x, y

def get_preprocessed_image(image_path):
    # TODO
    file = image_path.strip()
    image = cv2.imread(file)
    image = get_normalized_image(image)
    image = np.array(image)
    return image


def test_train_generator():
	x, y = generate_train_data_int(drive_data_relevant, printinfo = 1)
	print('Shape of image:')
	print(x.shape)
	print('Steering angle:')
	print(y)

def test_valid_generator():
	x, y = generate_valid_data_int(drive_data_relevant, printinfo = 1)
	print('Shape of image:')
	print(x.shape)
	print('Steering angle:')
	print(y)


def train_model(model):
	# trains the model
	learning_rate = 0.0001 # Perhaps it has to be changed?
	
	adam_optimizer = Adam(lr = learning_rate)
	
	model.compile(optimizer = adam_optimizer, loss = "mse")

	data_generator_valid = generate_valid_data(drive_data_relevant)
	valid_size = len(drive_data_relevant)

	data_generator_train = generate_train_data(drive_data_relevant)

	model_data = model.fit_generator(data_generator_train,
            samples_per_epoch = 20000, nb_epoch = 1, validation_data = data_generator_valid,
                        nb_val_samples = valid_size, verbose = 1)


	print(model_data)
	X_validation = get_single_validation_data(drive_data_relevant, 0)
	val_preds = model.predict(X_validation)
	print('eins:',min(val_preds), max(val_preds))
	X_validation = get_single_validation_data(drive_data_relevant, 1)
	val_preds = model.predict(X_validation)
	print('zwei:',min(val_preds), max(val_preds))
	X_validation = get_single_validation_data(drive_data_relevant, 2)
	val_preds = model.predict(X_validation)
	print('drei:',min(val_preds), max(val_preds))
	X_validation = get_single_validation_data(drive_data_relevant, 3)
	val_preds = model.predict(X_validation)
	print('vier:',min(val_preds), max(val_preds))

	file_name_model = 'model.json'
	file_name_weights = 'model.h5'

	save_trained_model(file_name_model, file_name_weights)

	valid_loss = model_data.history['val_loss'][0]

	print(valid_loss)
	print()
	print('It worked out.')


def save_trained_model(path_model, path_weights):
    print('Save model at:')
    print('Model: ', path_model)
    print('Weights: ', path_weights)
    
    if Path(path_model).is_file():
        os.remove(path_model)
    json_string = model.to_json()
    with open(path_model,'w' ) as file:
        json.dump(json_string, file)
    if Path(path_weights).is_file():
        os.remove(path_weights)
    model.save_weights(path_weights)

path_of_data = './data_udacity'

drive_data_relevant = load_data_info(path_of_data)

# Only for testing 
debug_test = 1
if debug_test == 1:
	image_index = 2
	print_image_data(drive_data_relevant, image_index)
	print()

if debug_test == 1:
	print()
	print('Test train generator')
	test_train_generator()
	print()

if debug_test == 1:
	print()
	print('Test valid generator')
	test_valid_generator()
	print()

#model = model_test()
model = model_nvidia_gada()
print(model.summary())
print()

train_model(model)

print()
print('done!!!')
