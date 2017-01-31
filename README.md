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

  

 
