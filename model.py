# Model for driving a car 
# Author: gada1982
# mail: daniel@gattringer.biz

# General imports
import os
import numpy as np
import csv
import cv2
import random
import math
from pathlib import Path

# Imports for keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, ELU, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

# Import json to save the model
import json

# Define global variable
# Check if it possible to get rid of the global variables, because of bad programming style
global_count_left = 0
global_count_center = 0
global_count_right = 0
global_count_valid = 0
new_size_col = 47
new_size_row = 160

def load_data_info(path_of_data):
	if not os.path.exists(path_of_data):
		print("Directory with data not found.")
		sys.exit(-1)

	# Open *.csv with logged driving data
	with open(path_of_data + '/driving_log.csv', 'r') as logfile:
		file_read = csv.reader(logfile, delimiter=',')
		drive_data = []
		for i in file_read:
			drive_data.append(i)

	# The log file data is available in the following type: 
	# center_image, left_image, right_image, steering-data, throttle-data, brake-data, speed-data
	# Onyl center_image, left_image, right_image and steering-data are used 
	drive_data = np.array( drive_data )
	drive_data_relevant = np.hstack(( drive_data[1:, 0].reshape((-1,1)), drive_data[1:, 1].reshape((-1,1)), drive_data[1:, 2].reshape((-1,1)), drive_data[1:,3].reshape((-1,1))))

	return drive_data_relevant

def modify_data_info(drive_data):
	count_left = 0
	count_center = 0
	count_right = 0
	debug_data_generator = 0

	# Count how many images are with steering left, steering right and nearly not steering
	# Steering left: steering data < -0.1
	# Steering right: steering data > 0.1
	# Nearly no steering: -0.1 <= steering data <= 0.1
	for i in range(0, len(drive_data)):
		if float(drive_data[i][3]) < - 0.1:
			# Steering left: steering data < -0.1
			count_left += 1
		elif float(drive_data[i][3]) > 0.1:
			# Steering right: steering data > 0.1
			count_right += 1
		else: 
			# Nearly no steering: -0.1 <= steering data <= 0.1
			count_center += 1

	w = 3
	# leep space for flipped images
	h_left = (count_left + count_right ) * 3
	# No flipping of center images
	h_center = count_center * 3
	# leep space for flipped images
	h_right =  (count_left + count_right ) * 3

	drive_data_left = [[0 for x in range(w)] for y in range(h_left)]
	drive_data_center = [[0 for x in range(w)] for y in range(h_center)]
	drive_data_right = [[0 for x in range(w)] for y in range(h_right)]
	
	steering_offset = 0.2
	i_left = 0
	i_center = 0
	i_right = 0
	for i in range(0, len(drive_data)):
		if float(drive_data[i][3]) < - 0.1:
			# Steering left: steering data < -0.1
			# Include image of center camera
			if i_left < len(drive_data_left):
				# Set image name
				drive_data_left[i_left][0] = drive_data[i][0]
				# Set steering angle
				drive_data_left[i_left][1] = drive_data[i][3]
				# Not flipped image
				drive_data_left[i_left][2] = 0
				i_left += 1
			else:
				print('error')
			# Include image of left camera
			if i_left < len(drive_data_left):
				# Set image name
				drive_data_left[i_left][0] = drive_data[i][1]
				# Set steering angle
				drive_data_left[i_left][1] = str(float(drive_data[i][3]) + steering_offset)
				# Not flipped image
				drive_data_left[i_left][2] = 0
				i_left += 1
			else:
				print('error')
			# Include image of right camera
			if i_left < len(drive_data_left):
				# Set image name
				drive_data_left[i_left][0] = drive_data[i][2]
				# Set steering angle
				drive_data_left[i_left][1] = str(float(drive_data[i][3]) - steering_offset)
				# Not flipped image
				drive_data_left[i_left][2] = 0
				i_left += 1
			else:
				print('error')

			# Include flipped images (center, left, right)
			# Include image of center camera
			if i_right < len(drive_data_right):
				# Set image name of the center image
				drive_data_right[i_right][0] = drive_data[i][0]
				# Set steering angle (flipped)
				drive_data_right[i_right][1] = str(float(drive_data[i][3]) * - 1 )
				# Not flipped image
				drive_data_right[i_right][2] = 1
				i_right += 1
			else:
				print('error')
			# Include image of left camera
			if i_right < len(drive_data_right):
				# Set image name of the center image
				drive_data_right[i_right][0] = drive_data[i][1]
				# Set steering angle (flipped)
				drive_data_right[i_right][1] = str((float(drive_data[i][3]) * - 1 ) - steering_offset)
				# Not flipped image
				drive_data_right[i_right][2] = 1
				i_right += 1
			else:
				print('error')
			# Include image of right camera
			if i_right < len(drive_data_right):
				# Set image name of the center image
				drive_data_right[i_right][0] = drive_data[i][2]
				# Set steering angle (flipped)
				drive_data_right[i_right][1] = str((float(drive_data[i][3]) * - 1 ) + steering_offset)
				# Not flipped image
				drive_data_right[i_right][2] = 1
				i_right += 1
			else:
				print('error')

		elif float(drive_data[i][3]) > 0.1:
			# Steering right: steering data > 0.1
			# Include image of center camera
			if i_right < len(drive_data_right):
				# Set image name
				drive_data_right[i_right][0] = drive_data[i][0]
				# Set steering angle
				drive_data_right[i_right][1] = drive_data[i][3]
				# Not flipped image
				drive_data_right[i_right][2] = 0
				i_right += 1
			else:
				print('error')
			# Include image of left camera
			if i_right < len(drive_data_right):
				# Set image name
				drive_data_right[i_right][0] = drive_data[i][1]
				# Set steering angle
				drive_data_right[i_right][1] = str(float(drive_data[i][3]) + steering_offset)
				# Not flipped image
				drive_data_right[i_right][2] = 0
				i_right += 1
			else:
				print('error')
			# Include image of right camera
			if i_right < len(drive_data_right):
				# Set image name
				drive_data_right[i_right][0] = drive_data[i][2]
				# Set steering angle
				drive_data_right[i_right][1] = str(float(drive_data[i][3]) - steering_offset)
				# Not flipped image
				drive_data_right[i_right][2] = 0
				i_right += 1
			else:
				print('error')

			# Include flipped images (center, left, right)
			# Include image of center camera
			if i_left < len(drive_data_left):
				# Set image name of the center image
				drive_data_left[i_left][0] = drive_data[i][0]
				# Set steering angle (flipped)
				drive_data_left[i_left][1] = str(float(drive_data[i][3]) * - 1 )
				# Not flipped image
				drive_data_left[i_left][2] = 1
				i_left += 1
			else:
				print('error')
			# Include image of left camera
			if i_left < len(drive_data_left):
				# Set image name of the center image
				drive_data_left[i_left][0] = drive_data[i][1]
				# Set steering angle (flipped)
				drive_data_left[i_left][1] = str((float(drive_data[i][3]) * - 1 ) - steering_offset)
				# Not flipped image
				drive_data_left[i_left][2] = 1
				i_left += 1
			else:
				print('error')
			# Include image of right camera
			if i_left < len(drive_data_left):
				# Set image name of the center image
				drive_data_left[i_left][0] = drive_data[i][2]
				# Set steering angle (flipped)
				drive_data_left[i_left][1] = str((float(drive_data[i][3]) * - 1 ) + steering_offset)
				# Not flipped image
				drive_data_left[i_left][2] = 1
				i_left += 1
			else:
				print('error')
		
		else: 
			# Nearly no steering: -0.1 <= steering data <= 0.1
			# Include image of center camera
			if i_center < len(drive_data_center):
				# Set image name
				drive_data_center[i_center][0] = drive_data[i][0]
				# Set steering angle
				drive_data_center[i_center][1] = drive_data[i][3]
				# Not flipped image
				drive_data_center[i_center][2] = 0
				i_center += 1
			else:
				print('error')
			# Include image of left camera
			if i_center < len(drive_data_center):
				# Set image name
				drive_data_center[i_center][0] = drive_data[i][1]
				# Set steering angle
				drive_data_center[i_center][1] = str(float(drive_data[i][3]) + steering_offset)
				# Not flipped image
				drive_data_center[i_center][2] = 0
				i_center += 1
			else:
				print('error')
			# Include image of right camera
			if i_center < len(drive_data_center):
				# Set image name
				drive_data_center[i_center][0] = drive_data[i][2]
				# Set steering angle
				drive_data_center[i_center][1] = str(float(drive_data[i][3]) - steering_offset)
				# Not flipped image
				drive_data_center[i_center][2] = 0
				i_center += 1
			else:
				print('error')

	if debug_data_generator == 1:
		print('Left_Len: ', len(drive_data_left))
		print('Center_Len: ', len(drive_data_center))
		print('Right_Len: ', len(drive_data_right))
		print()
		print()
		print('Left: ')
		print(drive_data_left)
		print('Center: ')
		print(drive_data_center)
		print('Right: ')
		print(drive_data_right)

	# Shuffle data to not have images always in the same order
	# IMPORTANT
	random.shuffle(drive_data_left)
	random.shuffle(drive_data_center)
	random.shuffle(drive_data_right)

	if debug_data_generator == 1:
		print('Left_Len: ', len(drive_data_left))
		print('Center_Len: ', len(drive_data_center))
		print('Right_Len: ', len(drive_data_right))
		print('Valid_Len: ', len(drive_data_valid))

	split_index = int(0.8 * len(drive_data_left))
	new_drive_data_left,  drive_data_valid_tmp1 = np.split(drive_data_left, [split_index])
	split_index = int(0.8 * len(drive_data_center))
	new_drive_data_center,  drive_data_valid_tmp2 = np.split(drive_data_center, [split_index])
	split_index = int(0.8 * len(drive_data_right))
	new_drive_data_right,  drive_data_valid_tmp3 = np.split(drive_data_right, [split_index])

	new_drive_data_valid = np.concatenate((drive_data_valid_tmp1, drive_data_valid_tmp2, drive_data_valid_tmp3), axis=0)
	
	if debug_data_generator == 1:
		print('Länge Left nachher: ',len(new_drive_data_left))
		print('Länge Center nachher: ',len(new_drive_data_center))
		print('Länge Right nachher: ',len(new_drive_data_right))
		print('Länge Valid nachher: ',len(new_drive_data_valid))

	return new_drive_data_left, new_drive_data_center, new_drive_data_right, new_drive_data_valid

def get_normalized_hsv_image(image):
    # Change color-space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Normalize from -1 to 1 (zero mean)
    image = image / 127.5 - 1
    return image

def print_image_data(data, number):
	# Print out some information about one image
	image_name = data[number][0]
	image_name = image_name.strip()
	image_path = path_of_data + '/' + image_name
	print()
	print('Datastring:')
	print(data[number])
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
	image_name = image_name.strip()
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
    
    row = new_size_row
    col = new_size_col
    ch = 3
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

def generate_train_data():
	# Generate training data
	while 1:
		x, y = generate_train_data_int(printinfo = 0)
		yield x, y

def generate_valid_data():
	# Generate validation data
	# TODO: Send all data for validation
	while 1:
		x, y = generate_valid_data_int(printinfo = 0)
		yield x, y
	
def generate_valid_data_int(printinfo):
	# TODO: Preprocessing
	
	# Check if it possible to get rid of the global variables, because of bad programming style
	global global_count_valid
	
	data = drive_data_valid
	count_index = global_count_valid
	global_count_valid += 1
	if global_count_valid == len(drive_data_valid):
		global_count_valid = 0

	image_name = data[count_index][0]
	image_name = image_name.strip()
	image_path = path_of_data + '/' + image_name
	if printinfo == 1:
		print()
		print('Image to generate:')
		print(image_path)
	x = cv2.imread(image_path)
	if data[count_index][2] == 1:
		# Image has to be flipped
		x = cv2.flip(x,1)
	x = preprocessImage(x)
	x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
	y = data[count_index][1]
	y = np.array([[y]])
	return x, y

def generate_train_data_int(printinfo):
	# TODO: Preprocessing
	
	# Check if it possible to get rid of the global variables, because of bad programming style
	global global_count_left
	global global_count_center
	global global_count_right

	# Choose randomly if an image with steering to 
	# the left or right in the center image should be used
	choose_data_pack = np.random.randint(0, 2)
	
	if choose_data_pack == 0:
		# Choose drive_data_left
		data = drive_data_left
		count_index = global_count_left
		global_count_left += 1
		if global_count_left == len(drive_data_left):
			global_count_left = 0
	elif choose_data_pack == 1:
		# Choose drive_data_center
		data = drive_data_center
		count_index = global_count_center
		global_count_center += 1
		if global_count_center == len(drive_data_center):
			global_count_center = 0
	else:
		# Choose drive_data_right
		data = drive_data_right
		count_index = global_count_right
		global_count_right += 1
		if global_count_right == len(drive_data_right):
			global_count_right = 0

	image_name = data[count_index][0]
	image_name = image_name.strip()
	image_path = path_of_data + '/' + image_name
	if printinfo == 1:
		print()
		print('Image to generate:')
		print(image_path)
	x = cv2.imread(image_path)
	if data[count_index][2] == 1:
		# Image has to be flipped
		x = cv2.flip(x,1)
	x = preprocessImage(x)
	x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
	y = data[count_index][1]
	y = np.array([[y]])
	return x, y

def get_single_validation_data(data,number):
	# TODO: Preprocessing
	i = number
	image_name = data[i][0]
	image_name = image_name.strip()
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


def test_train_generator():
	x, y = generate_train_data_int(printinfo = 1)
	print('Shape of image:')
	print(x.shape)
	print('Steering angle:')
	print(y)

def test_valid_generator():
	x, y = generate_valid_data_int(printinfo = 1)
	print('Shape of image:')
	print(x.shape)
	print('Steering angle:')
	print(y)


def train_model(model):
	# trains the model
	learning_rate = 0.0001 # Perhaps it has to be changed?
	
	adam_optimizer = Adam(lr = learning_rate)
	
	model.compile(optimizer = adam_optimizer, loss = "mse")

	data_generator_valid = generate_valid_data()

	data_generator_train = generate_train_data()

	model_data = model.fit_generator(data_generator_train,
            samples_per_epoch = 20000, nb_epoch = 1, validation_data = data_generator_valid,
                        nb_val_samples = len(drive_data_valid), verbose = 1)


	test_it = 0
	if test_it == 1:
		print(model_data)
		X_validation = get_single_validation_data(drive_data_left, 0)
		val_preds = model.predict(X_validation)
		print('eins:',min(val_preds), max(val_preds))
		X_validation = get_single_validation_data(drive_data_left, 1)
		val_preds = model.predict(X_validation)
		print('zwei:',min(val_preds), max(val_preds))
		X_validation = get_single_validation_data(drive_data_left, 2)
		val_preds = model.predict(X_validation)
		print('drei:',min(val_preds), max(val_preds))
		X_validation = get_single_validation_data(drive_data_left, 3)
		val_preds = model.predict(X_validation)
		print('vier:',min(val_preds), max(val_preds))

	file_name_model = 'model.json'
	file_name_weights = 'model.h5'

	save_trained_model(file_name_model, file_name_weights)

	valid_loss = model_data.history['val_loss'][0]

	print(valid_loss)
	print()
	print('It worked out.')

def preprocessImage(image):
	# Preprocessing image files
	shape = image.shape
	print('shape:',shape)
	image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
	image = get_normalized_hsv_image(image)
	image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
	return image 

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

# Configuration area
debug_image_data = 0
debug_train_generator = 1
debug_valid_generator = 0
do_training = 1


path_of_data = './data_udacity'

drive_data_relevant = load_data_info(path_of_data)
drive_data_left, drive_data_center, drive_data_right, drive_data_valid = modify_data_info(drive_data_relevant)

print()
print('Left: ', len(drive_data_left))
print('Center: ', len(drive_data_center))
print('Right: ', len(drive_data_right))
print('Valid: ', len(drive_data_valid))
print()

# Only for testing 
if debug_image_data == 1:
	image_index = 2
	print_image_data(drive_data_center, image_index)
	print()

if debug_train_generator == 1:
	print()
	print('Test train generator')
	test_train_generator()
	print()

if debug_valid_generator == 1:
	print()
	print('Test valid generator')
	test_valid_generator()
	print()

if do_training == 1:
	#model = model_test()
	model = model_nvidia_gada()
	print(model.summary())
	print()

	# Train the model
	train_model(model)

print()
print('done!!!')
