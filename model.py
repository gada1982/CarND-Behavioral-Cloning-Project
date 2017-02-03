# Project for Self-Driving Car Engineer provided from Udacity
# Model for cloning human driving behaviour
# Author: gada1982
# Mail: daniel@gattringer.biz

# General imports
import os
import numpy as np
import csv
import cv2
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

# Imports for keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, Lambda
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

# Import json to save the model
import json

# Define global variable
# TODO: Check if it possible to get rid of the global variables, because of bad programming style
new_size_row = 64
new_size_col = 64
new_size_ch = 3
number_of_epochs = 20
batch_size = 128

path_of_data = './data_udacity/'


'''
This function loads the driving data out of a *.csv-file
INPUT:
        path = path of the data-file
RETURN:
        drive_data_relevant = extracted data
'''
def load_data_info(path):
  if not os.path.exists(path_of_data):
    print("Directory with data not found.")
    sys.exit(-1)

  # Open *.csv with logged driving data
  with open(path + 'driving_log.csv', 'r') as logfile:
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


'''
This function splits the driving data into better useable parts
INPUT:
        drive_data = raw driving data
RETURN:
        center_img = links to images taken from the center camera
        left_img = links to images taken from the left camera
        right_img = links to images taken from the right camera
        steer_angle_center = steering angles connected to center camera
        steer_angle_l_r = steering angles connected to left and right camera
'''
def split_data_info(drive_data):
  center_img = []
  left_img = []
  right_img = []
  steer_angle_center = []
  steer_angle_l_r = []

  # Do split up
  for i in range(0, len(drive_data)):
    center_img.append(drive_data[i][0])
    left_img.append(drive_data[i][1])
    right_img.append(drive_data[i][2])
    steer_angle_center.append(drive_data[i][3])
    steer_angle_l_r.append(drive_data[i][3])

  return center_img, left_img, right_img, steer_angle_center, steer_angle_l_r


'''
This function splits up the images from center camera (training and validation)
Images from the left and right camera are only used for training and not for validation
INPUT:
        center_img = links to images taken from the center camera
        steer_angle_center = steering angles connected to center camera
RETURN:
        center_img = links to images taken from the center camera (for training)
        steer_angle_center = steering angles connected to center camera (for training)
        X_valid = links to images taken from the center camera (for validation)
        y_valid = steering angles connected to center camera (for validation)         
'''
def split_train_valid(center_img, steer_angle_center):
  # TODO Comment, set to 20% ?
  # Shuffle the data before split up
  center_img, steer_angle_center = shuffle(center_img, steer_angle_center)
  # Split training and validation data
  center_img, X_valid, steer_angle_center, y_valid = train_test_split(center_img, steer_angle_center, test_size = 0.10, random_state = 42) 
  print()
  print('Number of images for validation: ', len(X_valid))
  print()

  return center_img, X_valid, steer_angle_center, y_valid


'''
This function prints out some information about the dataset
INPUT:
        data = dataset for which information should be printed
RETURN:
        type = gives information about which of the different input types should be processed
'''
def print_info_data_set(data, type):
  class_info = [0, 0 ,0, 0, 0 ,0, 0, 0 ,0, 0, 0]

  # Total number of entries
  print('Total number: ', len(data))
  
  # Count how often different groups of steering angles appear
  for i in range(0, len(data)):
    if type == 0:
      value = float(data[i][3])
    elif type == 1:
      value = float(data[i])
    else:
      value = float(data[i][1])
    if value < -0.9:
      class_info[0] += 1
    elif value < -0.7:
      class_info[1] += 1
    elif value < -0.5:
      class_info[2] += 1
    elif value < -0.3:
      class_info[3] += 1
    elif value < -0.1:
      class_info[4] += 1
    elif value <= 0.1:
      class_info[5] += 1
    elif value <= 0.3:
      class_info[6] += 1
    elif value <= 0.5:
      class_info[7] += 1
    elif value <= 0.7:
      class_info[8] += 1
    elif value <= 0.9:
      class_info[9] += 1
    else:
      class_info[10] += 1

  # Print class information
  print('Class distribution: ', class_info)

# TODO
def get_fact(steer):
  value = float(steer)
  if value < -0.9:
    fact = 20
  elif value < -0.7:
    fact = 15
  elif value < -0.5:
    fact = 8
  elif value < -0.3:
    fact = 3
  elif value < -0.1:
    fact = 1
  elif value <= 0.1:
    fact = 1
  elif value <= 0.3:
    fact = 1
  elif value <= 0.5:
    fact = 3
  elif value <= 0.7:
    fact = 8
  elif value <= 0.9:
    fact = 15
  else:
    fact = 20

  return fact

'''
This function splits up the driving data in lists for steering left, not steering (straight) and steering right.
This is necessary to get rid of the uneven distribution in the provided data. Mostly straight driving.
Lateron randomly chosen between this three groups
INPUT:
        img_list_center = links to images taken from the center camera
        img_list_left = links to images taken from the left camera
        img_list_right = links to images taken from the right camera
        steering_list_center = steering angles connected to center camera
        steering_list_l_r = steering angles connected to left and right camera
RETURN:
        drive_data_steer_left = data when car is steering to the left
        drive_data_straight = data when car is not steering (straight)
        drive_data_steer_right  = data when car is steering to the right
'''
def split_data_left_straight_right(img_list_center, steering_list_center, img_list_left, img_list_right, steering_list_l_r):
  w = 2
  print_debug = 1
  drive_data_steer_left = [[0 for x in range(w)] for y in range(1)]
  drive_data_straight = [[0 for x in range(w)] for y in range(1)]
  drive_data_steer_right = [[0 for x in range(w)] for y in range(1)]
  drive_data_steer_left_app = [[0 for x in range(w)] for y in range(1)]
  drive_data_straight_app = [[0 for x in range(w)] for y in range(1)]
  drive_data_steer_right_app = [[0 for x in range(w)] for y in range(1)]
  
  # Necessary because of a lack in programming -> Try to get rid of it
  found_left = 0
  found_straight = 0
  found_right = 0
  
  split_left = -0.15 #TODO try different value
  split_right = -1 * split_left

  for i in range(0, len(img_list_center)):
    if float(steering_list_center[i]) < split_left:
      #print('left: ', steering_list_center[i])
      # Steering left: steering data < split_left
      if found_left == 0:
        # Include image of center camera
        # Set image name
        drive_data_steer_left[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_steer_left[0][1] = steering_list_center[i]
        found_left = 1
      else:
        # Set image name
        drive_data_steer_left_app[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_steer_left_app[0][1] = steering_list_center[i]      
        x = 0
        while x < get_fact(steering_list_center[i]):
          drive_data_steer_left = np.concatenate((drive_data_steer_left, drive_data_steer_left_app), axis=0)
          x += 1

    elif float(steering_list_center[i]) > split_right:
      #print('right: ', steering_list_center[i])
      # Steering right: steering data > split_right
      # Include image of center camera
      if found_right == 0:
        # Set image name
        drive_data_steer_right[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_steer_right[0][1] = steering_list_center[i]
        found_right = 1
      else:
        # Set image name
        drive_data_steer_right_app[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_steer_right_app[0][1] = steering_list_center[i]
        x = 0
        while x < get_fact(steering_list_center[i]):
          drive_data_steer_right = np.concatenate((drive_data_steer_right, drive_data_steer_right_app), axis=0)
          x += 1

    else: 
      # Nearly no steering: split_left <= steering data <= split_right
      # Include image of center camera
      #print('center: ', steering_list_center[i])
      if found_straight == 0:
        # Set image name
        drive_data_straight[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_straight[0][1] = steering_list_center[i]
        found_straight = 1
      else:
        # Set image name
        drive_data_straight_app[0][0] = img_list_center[i]
        # Set steering angle
        drive_data_straight_app[0][1] = steering_list_center[i]
        x = 0
        while x < get_fact(steering_list_center[i]):
          drive_data_straight = np.concatenate((drive_data_straight, drive_data_straight_app), axis=0)
          x += 1

  if print_debug == 1:
    print()
    print('Distribution of Data from images from the center camera (for training):')
    d_data = np.concatenate((drive_data_steer_left, drive_data_straight, drive_data_steer_right), axis=0)
    print_info_data_set(d_data, 2)
    print('Original dataset only center images: ')
    print('Steer left: ', len(drive_data_steer_left))
    print('Steer center: ', len(drive_data_straight))
    print('Steer right: ', len(drive_data_steer_right)) 
    print()

  # len(img_list_left) == len(img_list_right)
  # Only len(img_list_center) is smaller, because of split for validation
  for i in range(0, len(img_list_left)):
    #steering_offset = random.uniform(0.10,0.20)
    steering_offset = 0.25
    
    # Include left camera images
    new_steering_angle_left_cam = float(steering_list_l_r[i]) + steering_offset
    
    if new_steering_angle_left_cam < split_left:
      # Set image name
      drive_data_steer_left_app[0][0] = img_list_left[i]
      # Set steering angle
      drive_data_steer_left_app[0][1] = str(new_steering_angle_left_cam)
      x = 0
      while x < get_fact(new_steering_angle_left_cam):
        drive_data_steer_left = np.concatenate((drive_data_steer_left, drive_data_steer_left_app), axis=0)
        x +=1
    elif new_steering_angle_left_cam > split_right:
      # Set image name
      drive_data_steer_right_app[0][0] = img_list_left[i]
      # Set steering angle
      drive_data_steer_right_app[0][1] = str(new_steering_angle_left_cam)
      x = 0
      while x < get_fact(new_steering_angle_left_cam):
        drive_data_steer_right = np.concatenate((drive_data_steer_right, drive_data_steer_right_app), axis=0)
        x += 1
    else: 
      # Set image name
      drive_data_straight_app[0][0] = img_list_left[i]
      # Set steering angle
      drive_data_straight_app[0][1] = str(new_steering_angle_left_cam)
      x = 0
      while x < get_fact(new_steering_angle_left_cam):
        drive_data_straight = np.concatenate((drive_data_straight, drive_data_straight_app), axis=0)
        x += 1

    # Include right camera images
    new_steering_angle_right_cam = float(steering_list_l_r[i]) - steering_offset
    
    if new_steering_angle_right_cam < split_left:
      # Set image name
      drive_data_steer_left_app[0][0] = img_list_right[i]
      # Set steering angle
      drive_data_steer_left_app[0][1] = str(new_steering_angle_right_cam)
      x = 0
      while x < get_fact(new_steering_angle_right_cam):
        drive_data_steer_left = np.concatenate((drive_data_steer_left, drive_data_steer_left_app), axis=0)
        x += 1
    elif new_steering_angle_right_cam > split_right:
      # Set image name
      drive_data_steer_right_app[0][0] = img_list_right[i]
      # Set steering angle
      drive_data_steer_right_app[0][1] = str(new_steering_angle_right_cam)
      x = 0
      while x < get_fact(new_steering_angle_right_cam):
        drive_data_steer_right = np.concatenate((drive_data_steer_right, drive_data_steer_right_app), axis=0)
        x += 1
    else: 
      # Set image name
      drive_data_straight_app[0][0] = img_list_right[i]
      # Set steering angle
      drive_data_straight_app[0][1] = str(new_steering_angle_right_cam)
      x = 0
      while x < get_fact(new_steering_angle_right_cam):
        drive_data_straight = np.concatenate((drive_data_straight, drive_data_straight_app), axis=0)
        x += 1

  if print_debug == 1:
    print('Distribution of Data from images from left/right camera:')
    d_data = np.concatenate((drive_data_steer_left, drive_data_straight, drive_data_steer_right), axis=0)
    print_info_data_set(d_data, 2)
    print('Dataset with left, center, right camera images: ')
    print('Steer left: ', len(drive_data_steer_left))
    print('Steer center: ', len(drive_data_straight))
    print('Steer right: ', len(drive_data_steer_right)) 
    print()
      
  return drive_data_steer_left, drive_data_straight, drive_data_steer_right

'''
This function changes the brightness of an image randomly.
INPUT:
        image = image, which brightness should be changed
RETURN:
        final_image = image, which brightness has been changed
'''
def change_brightness(image):
    # Convert from RGB to HSV to change brightness
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    #TODO test random_brightness = random.uniform(0.25,1.25)
    random_brightness = random.uniform(0.25,1.25)
    image_hsv[:,:,2] = image_hsv[:,:,2] * random_brightness
    #Convert back to RGB colorspace
    final_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return final_image 

'''
This function changes the color space from an image (RGB to HSV)
INPUT:
        image = image, which color space should be changed
RETURN:
        image_hsv = image, which color space has been changed
'''
def change_to_hsv(image):
    # Convert from RGB to HSV to change brightness
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image_hsv 

'''
This function flips an image vertically and adjusts it steering angle (inverse)
INPUT:
        image = image, which should be flipped
        steer = steering data to inverse
RETURN:
        flipped_image = image, which has been flipped
        flipped_steer = inversed steering data 
'''
def flip_image(image, steer):
  flipped_image = cv2.flip(image, 1)
  flipped_steer = -1 * steer
  return flipped_image, flipped_steer


'''
This function does affine transformation an image. Shrinks horicontally and adjusts it steering angle
INPUT:
        image = image, which should be transformed
        steer = steering data adjusted
RETURN:
        image_trans = image, which has been transformed
        steer_trans = adjusted steering data 
'''
def affi_trans_image(image,steer):
  
  row, col, ch = image.shape
  
  # Apply horizontal translation with steering compensation
  trans_hor = 50 * np.random.uniform() - 25
  steer_trans = float(steer) + trans_hor / 125
  
  # Set up transformation matrix and apply
  trans_matrix = np.float32([[1,0,trans_hor],[0,1,0]])
  image_trans = cv2.warpAffine(image,trans_matrix,(col,row))
  
  return image_trans, steer_trans


'''
This function cut and resizes an image
INPUT:
        image = image, which sould be cut and resized
RETURN:
        image_resized = image, which has been cut and resized
'''
def cut_and_resize_image(image):
  # Cut image on top (20px) to cut the sky and bottom (20px) to cut the hood of the car
  # Cut image on (40px) left and right
  image_cut = image[20:140,40:280]
  # Resize the image to the defined input size for the neuronal network
  image_resized = cv2.resize(image_cut, (new_size_row,new_size_col))
  return image_resized


'''
This function generates batches of training data
INPUT:
        drive_data_steer_left = data with steering left
        drive_data_straight = data with nearly no steering (straigt)
        drive_data_steer_right = data with steering right
        batch_size = defines how many tuples should be given
RETURN:
        batch_train = yields the batches with images
        batch_steer = yields the batches with steering data
'''
def data_generator_train(drive_data_steer_left, drive_data_straight, drive_data_steer_right, batch_size):
    batch_train = np.zeros((batch_size, new_size_row, new_size_col, new_size_ch), dtype = np.float32)
    batch_steer = np.zeros((batch_size,), dtype = np.float32)
    while 1:
        for i in range(batch_size):
          # Choose randomly if an image with steering to 
          # the left or right in the center image should be used
          choose_data_pack = np.random.randint(0, 2)

          if choose_data_pack == 0:
            # Choose drive_data_steer_left
            data = drive_data_steer_left
          elif choose_data_pack == 1:
            # Choose drive_data_straight
            data = drive_data_straight
          else:
            # Choose drive_data_steer_right
            data = drive_data_steer_right

          # Randomly select an image to of the choosen data_list
          i_rand = int(np.random.choice(len(data),1))
          
          image = mpimg.imread(path_of_data + data[i_rand][0].strip())
          mod_image = change_brightness(image)
          
          do_affin = 1

          if do_affin == 1:
            # Don't apply horizontal shrinking for images with no steering
            if abs(float(data[i_rand][1])) > 0.1: 
              mod_image, mod_steer = affi_trans_image(mod_image, data[i_rand][1])
            else:
              mod_steer = float(data[i_rand][1])
          else:
              mod_steer = float(data[i_rand][1])

          batch_train[i] = cut_and_resize_image(mod_image)
          #batch_train[i] = change_to_hsv(mod_image)
          #batch_steer[i] = mod_steer * (1 + np.random.uniform(-0.10,0.10)) # TODO try other value
          batch_steer[i] = mod_steer
          # Randomly flip the image vertically -> steer data has to be inverted
          flip_images = random.randint(0,1)
          if flip_images == 1:
            batch_train[i], batch_steer[i] = flip_image(batch_train[i], batch_steer[i])

        yield batch_train, batch_steer


'''
This function generates batches of validation data
INPUT:
        data = data to choose batches of
        steer = steering data connected with data
        batch_size = defines how many tuples should be given
RETURN:
        batch_valid = yields the batches with images
        batch_steer = yields the batches with steering data
'''
def data_generator_valid(data, steer, batch_size):
    batch_valid = np.zeros((batch_size, new_size_row, new_size_col, new_size_ch), dtype = np.float32)
    batch_steer = np.zeros((batch_size,), dtype = np.float32)
    while 1:
      for i in range(batch_size):
        i_rand = int(np.random.choice(len(data),1))
        image = mpimg.imread(path_of_data + data[i_rand].strip())
        mod_image = cut_and_resize_image(image)
        #batch_valid[i] = change_to_hsv(mod_image)
        batch_steer[i] = steer[i_rand]
      yield batch_valid, batch_steer



'''
  TODO OLD get rid
'''
def model_nvidia_gada_2():  
  input_shape = (new_size_row, new_size_col, new_size_ch)

  model = Sequential()
  
  # Layer 1: Normalization of the input in range -1 to 1
  model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = input_shape))
  
  # Layer 2: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(24, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 3: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(36, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 4: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(48, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 5: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(64, 3, 3, border_mode = 'same', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 6: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(64, 3, 3, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 7: Flatten for the following fully connceted layers
  model.add(Flatten())
  
  # Layer 8: Fully connected layer with size 80 and l2-weights-regularization
  #           Dropout (0.5) to prevent overfitting
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  
  # Layer 9: Fully connected layer with size 40 and l2-weights-regularization
  #           Dropout (0.5) to prevent overfitting
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  
  # Layer 10: Fully connected layer with size 16 and l2-weights-regularization
  #           Dropout (0.5) to prevent overfitting
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  
  # Layer 11: Fully connected layer with size 10 and l2-weights-regularization
  #           Dropout (0.5) to prevent overfitting
  model.add(Dense(10, W_regularizer = l2(0.001)))
  
  # Layer 12: Fully connected layer with size 1 (Output-Layer) and l2-weights-regularization
  model.add(Dense(1, W_regularizer = l2(0.001)))
  return model

'''
  This function defines the architecture of the artificial neuronal network
'''
def model_nvidia_gada():  
  input_shape = (new_size_row, new_size_col, new_size_ch)

  model = Sequential()
  
  # Layer 1: Normalization of the input in range -1 to 1
  model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = input_shape))
  
  # Layer 2: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(24, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 3: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(36, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 4: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(48, 5, 5, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 5: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(64, 3, 3, border_mode = 'same', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 6: Convolution Layer with relu-activation and l2-weights-regularization
  model.add(Convolution2D(64, 3, 3, border_mode = 'valid', subsample = (2,2), activation = 'relu', W_regularizer = l2(0.001)))
  
  # Layer 7: Flatten for the following fully connceted layers
  model.add(Flatten())
  
  # Layer 8: Fully connected layer with size 100 and l2-weights-regularization
  #           Dropout (0.3) to prevent overfitting
  model.add(Dense(100, W_regularizer = l2(0.001)))
  model.add(Dropout(0.3))
  
  # Layer 9: Fully connected layer with size 50 and l2-weights-regularization
  #           Dropout (0.3) to prevent overfitting
  model.add(Dense(50, W_regularizer = l2(0.001)))
  model.add(Dropout(0.3))
  
  # Layer 10: Fully connected layer with size 10 and l2-weights-regularization
  #           Dropout (0.3) to prevent overfitting
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dropout(0.3))
  
  # Layer 12: Fully connected layer with size 1 (Output-Layer) and l2-weights-regularization
  model.add(Dense(1, W_regularizer = l2(0.001)))
  return model


'''
  This function trains the model of the artificial neuronal network
  
  INPUT:
        model = structure of the model to train
'''
def train_model(model, drive_data_steer_left, drive_data_straight, drive_data_steer_right, X_valid, y_valid):
  # trains the model
  learning_rate = 0.0001

  adam_optimizer = Adam(lr = learning_rate)

  model.compile(optimizer = adam_optimizer, loss = 'mse')

  generator_train = data_generator_train(drive_data_steer_left, drive_data_straight, drive_data_steer_right, batch_size)
  generator_valid = data_generator_valid(X_valid, y_valid, batch_size)
  count_all_images = len(drive_data_steer_left) +  len(drive_data_straight) + len(drive_data_steer_right)

  num_per_epoch = ((int(count_all_images / batch_size)) + 1) * batch_size
  
  model_data = model.fit_generator(generator_train, samples_per_epoch = num_per_epoch, nb_epoch = number_of_epochs, validation_data = generator_valid, nb_val_samples = len(X_valid))

  print('Training finished!')

  file_name_model = 'model.json'
  file_name_weights = 'model.h5'

  save_trained_model(file_name_model, file_name_weights, model)

  valid_loss = model_data.history['val_loss'][0]

  print(valid_loss)


'''
  This function saves the entire model of the artificial neuronal network
  
  INPUT:
        path_model = path and name of the model -> saved as *.json
        path_weights = path and name of the weights -> saved as *.h5
'''
def save_trained_model(path_model, path_weights, model):
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

    print('Model architecture and weights saved!')

'''
  Main function
'''
def main():
  # Configuration area
  do_training = 1
  print_debug = 1

  # New Data Collection
  drive_data = load_data_info(path_of_data)
  
  if print_debug == 1:
    print_info_data_set(drive_data, 0)
  
  center_img, left_img, right_img, steer_angle_center, steer_angle_l_r = split_data_info(drive_data)
  center_img, X_valid, steer_angle_center, y_valid = split_train_valid(center_img, steer_angle_center)

  drive_data_steer_left, drive_data_straight, drive_data_steer_right = split_data_left_straight_right(center_img, steer_angle_center, left_img, right_img, steer_angle_l_r)

  if do_training == 1:
    # model = model_nvidia_gada_2()
    model = model_nvidia_gada()
    print(model.summary())
    print()

    # Train the model
    train_model(model, drive_data_steer_left, drive_data_straight, drive_data_steer_right, X_valid, y_valid)

  print()
  print('Everything done!!!')

if __name__ == "__main__":
    main()
