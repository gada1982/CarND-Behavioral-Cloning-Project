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
A good starting point for the project has been a [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA. Finally the CNN, which has been used in this project is strongly influenced by the the model mentioned in this paper.

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
Normalize the values (between -1 to 1). Normalisation of images doesn't change the content of the images, but it makes it much easier for the optimization to proceed numerical and the variables should always have zero mean, if possible.





  

 
