# Keras  Implementation of Capsule Networks and CNN on MNIST Digits Dataset

[You can find the notebook file here for more explanation and steps of running the code](https://github.com/Desire100/Capsule-Network-and-CNN-Keras-Implementation-on-MNIST-Dataset/blob/master/Keras%20Implementation%20of%20CNN%20and%20Capsule%20Network%20on%20MNIST%20.ipynb)

In this demo we will build both capsule and convolutional neural network model for recognizing handwritten digits.
### Chapters

####   1. Intro to CNN and Capsule Network
####   2. Steps of building capsule network
####    3. Importing the dependencies
####   4. Loading and Reshaping the mnist dataset
####   5. Building CNN and CapsNet models
####   5.1 Training, Saving and Testing a CNN model
####   5.2 Training, Saving and Testing a Capsnet Model

## 1. Intro to  CNN and Capsule Network

##### What is convolution?
In purely mathematical terms, convolution is a function derived from two given functions by integration which expresses how the shape of one is modified by the other.

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. Unlike CNN, A capsule is a group of neurons which uses vectors to represent an object or object part. Length of a vector represents presence of an object and orientation of vector represents its pose(size, position, orientation, etc). Group of these capsules forms a capsule layer and then these layers lead to form a capsule network.

[References](https://arxiv.org/pdf/1710.09829.pdf)


##  2. Steps of Building Capsule Networks

#### Step One : Initial Convolutional Layer  
 This layer uses convolution to get low level features from image and pass them to the next layer of the network (a primary capsule layer).
#### Step Two: Primary Capsule Layer
A primary capsule layer reshapes output from  the previous layer (convolution layer) into capsules containing vectors of equal dimension. Length of each of these vector represents the probability of presence of an object.
#### Step Three: Digit Capsule Layer
#### Step Four:  Decorder Network
A decoder network reconstructs the original image using an output of digit capsule layer. 
