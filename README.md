# Introduction
This project is done as a part of the Nanodegree - *Self-Driving Car Engineer* provided from Udacity. The outcome of the projct is, that a system with an artificial neuronal network is able to learn driving a track successfully by only getting human driving behaviour as input. It has to be able to gerneralize to other tracks too. Images from cameras at the front (left/center/right) and steering angles are taken as input values. Collection of training data and testing is done within a simulator, which is provided by Udacity.

# Outline
1. Requirements
2. Files
3. Data Collection
4. Data Processing
5. Model Training
6. Model Testing
7. Conclusion

# 1. Requirements
- Python 3.5
- Environment [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) provided by Udacity
- Car Simulator provided by Udacity
  - [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
  - [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
  - [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
  - [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
- [Sample data] (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided from Udacity
  - if you don't want to collect own training data with the simulator

# 2. Files
- model.py - The script used to create and train the model
- drive.py - The script to drive the car for testing 
- model.json - The model architecture
- model.h5 - The model weights
- README.md - Explains the structure of the software and the approach to solve the problem 

# 3. Data Collection

### 3.1 Using the Simulator
The simulator can be used for data collection and testing the model. The **Training Mode** is used for human driving and collecting and storing data. The **Autonomous Mode** is used for testing the trained model. Both modes are available for both tracks. Track 1 is rather flat with soft curves. Track 2 is in the mountains, steeper and with more sharp turns.

![Simulator](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/simulator.png)

The simulator delivers and stores images of  left / center / right camera mounted behind the front-window of the car. In the dataset (.csv) the following data is stored. 
* Links to stored images of the three cameras
* Steering-data
* Throttle-data
* Brake-data
* Speed-data

At the moment only images and steering-data are used for training the model.

### 3.1 Using Data from Udacity
While collecting good training data is rather difficult, especially when using keyboard input to steer the car, Udacity provided a basis dataset (see chapter Requirements).

This data is not distributed very evenly over the possible steering angles (-1 to 1). Most of the data was for straight driving. When training artificial neuronal networks a balanced dataset is really important. If not, the final model gets biased to this uneven distribution.

**Original dataset (from Udacity) - only images from track 1:**
![unprocessed data](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/info%20data%20not%20processed.png)

Finally it wasn't necessary to use own collected data to train the model. The sample data with lots of data processing did the job.

# 4. Data Processing
The input data (sample data or own collected data) is shuffled and split up in training data and validation data. 10% of the data from the center camera is used for validation. While testing with the simulator only the images of the center camera are used. Because of this, only the images of this camera are used for validation. The validation data (804 images) is **never** used for training. No own testing data is split up, because during development I found out that **only** testing in the simulator can give a reliable feedback if the final model works properly.

### Data Preprocessing
Because of the uneven distribution within the training data and to make the model able to generalize to other conditions and tracks, the following steps have been done while preprocessing:
- Images from left / center / right camera have been used:
  - This data is mostly used for recovery (back to the middle of the road)
  - Steering angle correction of +/- 0.25
  - When taken an image from the left camera and interpreted as one from the center camera the car seem to be to far on the right side of the road -> steer a little bit to the left.
  - When taken an image from the right camera and interpreted as one from the center camera the car seem to be to far on the left side of the road -> steer a little bit to the right.
![Images from left / center / right camera](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/3%20cameras.png) 
- Add data for stronger curves:
  - Most of the steering data in the sample data is straight or for light turns. With this data the artificial neuronal network would be training to go straight or take light curves. When a single image is putten to the training data, the steering angle is checked and the sharper the turn is, the more often the same image is included to the training data. 
- Split the data into steering left / nearly no steearing (straight) / steering right:
  - Steering left: (x < -0.15)
  - No steering - straight: (-0.15 <= x <= 0.15)
  - Steering right: (x > 0.15)
  - During data generation it is randomly choosen, data out of which group is taken

**Processed dataset - only images from track 1:**
![processed data](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/info%20data%20processed.png)

**Final dataset:**
- Total number of images: 38520
- Number of images with steering left: 14736
- Number of images with nearly no steering: 7853
- Number of images with steering right: 15931

### Data Generation
To get training and validation data *fit_generator* is used. Additionally random data augmentation is included into the data-generators. The following steps have been done while data generation:
- Flip images randomly (vertical axis) - to have the same number of data for left and right turns:
  - Inverse steering angle
  - When flipping a left turn gets a right turn and the other way around
![Flipped images](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/Flipped_image.png)
- Randomly adjustment of brightness:
  - To avoid the model from getting biasd to lighter or darker conditions, to make it able to operate properly at lighter and darker conditions
![Image with adjusted brightness](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/Brightness_image.png)
- Apply affine transformations:
  - Squeeze the images in the horizontal direction (with a randomly choosen amount)
  - This generates sharper turns
  - Adjust the steering angles. More squeezing, more steering data adjustment
![Image with affine transformation](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/Squeezed%20image.png)
- Cut and resize the images:
  - from 160x320 to 64x64
![Cut and resized image](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/cut%20and%20resize.png)
- Choose randomly between the groups (steer left / straight / steer right)

The model should only learn "reading" the track and shouldn't be influenced by the hood of the car or the sky. Because of this, the images (160x320) are cut 20px on top and bottom, and 40px on the left and right side. This gives an image size of 120x240.
This cut images get shrinked to 64x64 without big disadvantages. 

Testing data (in the simulator) has to be adjusted to the same size (64x64). This preprocessing has been included into drive.py.

# 5. Model Training
The [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA has strongly influenced the model, which was finally used for this project. 

The following picture is taken out of this paper.

![Architecture NVIDIA](https://github.com/gada1982/CarND-Behavioral-Cloning-Project/blob/master/info_for_readme/architecture_nvidia.png)

**The final architecture:**
- Layer 1: Normalization of the input 
  - Input size: 64x64x3
  - Normalize the images within the range -1 to 1
- Layer 2: Convolution Layer
  - Number filters: 24
  - Filter Size: 5x5
  - Stride: 2x2
  - Mode: valid
  - L2 Regularization: 0.001
  - Activation: ReLu
- Layer 3: Convolution Layer 
  - Number filters: 36
  - Filter Size: 5x5
  - Stride: 2x2
  - Mode: valid
  - L2 Regularization: 0.001
  - Activation: ReLu
- Layer 4: Convolution Layer 
  - Number filters: 48
  - Filter Size: 5x5
  - Stride: 2x2
  - Mode: valid
  - L2 Regularization: 0.001
  - Activation: ReLu
- Layer 5: Convolution Layer 
  - Number filters: 64
  - Filter Size: 3x3
  - Stride: 2x2
  - Mode: same
  - L2 Regularization: 0.001
  - Activation: ReLu
- Layer 6: Convolution Layer 
  - Number filters: 64
  - Filter Size: 3x3
  - Stride: 2x2
  - Mode: valid
  - L2 Regularization: 0.001
  - Activation: ReLu
- Layer 7: Flatten
- Layer 8: Fully Connected Layer
  - Size: 100
  - L2 Regularization: 0.001
  - Dropout: 0.3
- Layer 9: Fully Connected Layer
  - Size: 50
  - L2 Regularization: 0.001
  - Dropout: 0.3
- Layer 10: Fully Connected Layer
  - Size: 10
  - L2 Regularization: 0.001
  - Dropout: 0.3
- Layer 12: Fully Connected Layer - Output Layer
  - Size: 1
  - L2 Regularization: 0.001

To avoid overfitting, 30% dropout is used for all fully connected layers, except the last one, which is the output layer. L2 weight regularization was applied to every layer. This is helpful for getting a smoother driving, which is less snappy. 

Normalisation of images doesn't change the content of the images, but it makes it much easier for the optimization to proceed numerical and the variables should always have zero mean, if possible. This is done within the model.

For this model / project an Adam optimizer seems to be the best solution. To avoid jumping around a rather small learning rate (0.0001) has been used.

The model for the artificial neuronal network is trained with [Keras](https://keras.io/) and a Tensorflow backend. In this project lots of data is needed. Because of this, it is not useful to keep the hole data in memory. As mentioned above *fit_generator* was used instead.

The metrics for measurement of the training progress *Mean Squared Error (mse)* has been used.

**For training use:**
> python model.py


The model was trained for 20 epochs with 38520 samples per epoch with batch-size 128.

The final loss (mse) was: 0.035

# 6. Model Testing

**For testing use:**
> python drive.py model.json

For testing track 1 (within the simulator) has been used. This was successful, because the car was able to pass the track in a smooth way, without hitting any track limits. This test was required to pass the project.

[Video - Track 1](https://youtu.be/rETuxAycmF0)

To show that the model generalizes to different tracks or conditions the second track (within the simulator) has been used. The model has **never** been trained on track 2. Anyway the car passes a significant part of the track sucessfully. But it was not able to pass a really sharp right turn on the track. This will be part of future work, because it was not necessary to pass this track for finalizing the project.

[Video - Track 2](https://youtu.be/wno412WE0h0)

# 7. Conclusion
The model, which was an adaption of the NVDIA-model, could clone the human driving behaviour and was able to generalize quite well to a track it has never seen while training. At the moment the model can only control the steering angle of the car. Trottle and brake data could be used for further work.

While training an artificial neuronal network **Garbage in - Garbage out** really matters. Understanding the data and getting a balanced dataset are the key figures.
