# CarND-Behavioral-Cloning-Project
This project is done as a part of the Nanodegree - *Self-Driving Car Engineer* provided from Udacity. The outcome of the projct is, that a system with an artificial neuronal network should be able to learn driving a track successfully by only getting human driving behaviour as input. Images from cameras at the front (left/center/right) and steering angles are the input values. Collection of training data and testing is done within a simulator, which is provided by Udacity.

## Requirements
- Python 3.5
- Environment [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) provided by Udacity
- Car Simulator provided by Udacity
  - [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
  - [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
  - [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
  - [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
- [Sample data] (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided from Udacity
  - if you don't want to collect own training data with the simulator

## Files
- model.py - The script used to create and train the model.
- drive.py - The script to drive the car. 
- model.json - The model architecture.
- model.h5 - The model weights.
- README.md - explains the structure of your network and training approach. 

## Usage of the simulator
(Text is mostly taken from Udacity-Source)
Once you’ve downloaded the simulator (see chapter Requirements), extract it and run it.

When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like. We suggest running at the smallest size and the fastest graphical quality. We also suggest closing most other applications (especially graphically intensive applications) on your computer, so that your machine can devote its resource to running the simulator.

The next screen gives you two options: Training Mode and Autonomous Mode.

### Training mode
You’ll enter the simulator and be able to drive the car with your arrow keys, just like it’s a video game.

### Autonomous mode
Autonomous mode will be used in a later step once the neural network is trained.

### Collection data
If you want to collect own training data -> use training mode of the simulator.

- Enter Training Mode in the simulator.
- Start driving the car to get a feel for the controls.
- When you are ready, hit the record button in the top right to start recording.
- Continue driving for a few laps or till you feel like you have enough data.
- Hit the record button in the top right again to stop recording.

How the data is stored will be explained later-on.

## Project Requirements
The simulator provides two tracks (in both modes). The simplier one (on the left) is called Track 1, the heavier one (in the mountains on the right) is called Track 2.

Within the project Deep Learning is used. Training only happens with data from Track 1. 

In the final solution the car has to successfully drive autonomously and without getting off the road on Track 1. Track 2 is only for self evaluation.

## Approach while Development
A good starting point for the project has been a [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA. Finally the CNN, which has been used in this project is strongly influenced by the model mentioned in this paper.

At the beginning I tried to collect own training data with the simulator. Using a keyboard produces mostly data with steering angle zero and the driving was not really smooth. Because of this I decided to start with the sample data (see chapter Requirements).

### Data is the key
While training an artificial neuronal network *Garbage in - Garbage out* really matters. Understanding the data and getting the balanced dataset are the key figures.

The simulator delivers images of  left / center / right camera mounted at the front-window of the car.
In the dataset the following data is stored (additional to the images).
center_image, left_image, right_image, steering-data, throttle-data, brake-data, speed-data
This data is stored in an .csv-file.
Only center_image, left_image, right_image and steering-data are used.

#### Training, Validation and Testing data
The input data (sample data or own collected data) is shuffled and split up in training data and validation data. 15% of the data is used for validation. The validation data is NOT used for training at all.
No own testing data is split up because during development I found out that only testing in the simulator can give a reliable feedback if the final model works in a proper way.

#### Distribution of the training data

#### Modification of the training data
Because of bad distribution within the training data the following tasks are made.
- Useage of images from left-camera and right camera
  - Steering angle correction of +/- 0.25
  - This amount was found out experimentally, in a real world example this can be calculated out of the geometry
- Flip images (vertical axis)
  - left-camera-image -> right-camera-image (with inverse steering angle)
  - right-camera-image -> left-camera-image (with inverse steering angle)
- Randomly adjustment of brightness
  - to avoid the model from getting biasd to lighter or darker conditions
- Change from RGB to HSV Color-Space
  
To avoid getting biasd to drive straight the data has been split up in three groups.
- Steer left ( x < -0.10 )
- No steering ( -0.10 <= x <= 0.1 )
- Steer right ( x < 0.10 )

Out of the groups the data is randomly selected while generating the training batch (by using fit_generator)

## Normalization
Normalize the values (between -1 to 1). Normalisation of images doesn't change the content of the images, but it makes it much easier for the optimization to proceed numerical and the variables should always have zero mean, if possible. This is done within the model.

## Preprocess images
The model should only learn "reading" the track and shouldn't be influence by the hood of the car or the sky. Because of this, the images (160x320) are taken cut by TODO XXpx on top and XXpx on the bottom. This gives an image size of XXx320.
After the lot of test I found out that the image size can by shrinked without big disadvantages. So the images are shrinked from XXx320 to 64x64 before feeding into the model. Furthermore images are changed from RGB to HSV Color-Space because this gave better results.
IMPORTANT: Preprocessing has to be done with training and testing data. This preprocessing has to be included into drive.py too.

## Generation of training and validation data
The model for the artificial neuronal network is trained with [Keras](https://keras.io/) and a Tensorflow backend. In this project lots of data is needed. Because of this, it is not useful to keep the hole data in memory, as this was done in the last project. To get training and validation data fit_generator is used.
Additionally random data augmentation is included in the data-generators.

Two different generators are used. One for training data and one for validation data.
- Training generator
TODO

- Validation generator
TODO

# Model / Architecture
As mentioned above the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA has strongly influenced the model which was finally used for this project.

- Layer 1: Normalization of the input 
  - Normalize the images within the range -1 to 1
    - This could be done outside the model too (while preprocessing), but including this into the model is much smarter.
- Layer 2: Convolution Layer
  - Filter Size: 5x5
- Layer 3: Convolution Layer 
  - Filter Size: 5x5
- Layer 4: Convolution Layer 
  - Filter Size: 5x5
- Layer 5: Convolution Layer 
  - Filter Size: 3x3
- Layer 6: Convolution Layer 
  - Filter Size: 3x3
- Layer 7: Flatten
- Layer 8: Fully Connected Layer
  - Size: 80
  - L2 Regularization: 0.001
  - Dropout: 0.5
- Layer 9: Fully Connected Layer
  - Size: 40
  - L2 Regularization: 0.001
  - Dropout: 0.5
- Layer 10: Fully Connected Layer
  - Size: 16
  - L2 Regularization: 0.001
- Layer 11: Fully Connected Layer
  - Size: 10
  - L2 Regularization: 0.001
- Layer 12: Fully Connected Layer - Output Layer
  - Size: 1
  - L2 Regularization: 0.001

To avoid overfitting, 50% Dropout is used for the first two (TODO) fully connected layers. L2 weight regularization is in every layer for getting a better driving, which is less snappy. 

For this model / project an Adam optimizer seems to be the best solution. To avoid jumping around a rather small learning rate (0.0001) has been used.

TODO include images from paper.



  

 
