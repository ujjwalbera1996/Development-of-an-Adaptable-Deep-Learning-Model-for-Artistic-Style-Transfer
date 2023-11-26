# Development-of-an-Adaptable-Deep-Learning-Model-for-Artistic-Style-Transfer
# Overview
This repository contains code for an adaptable deep learning model capable of transferring artistic styles from one image to another. The model learns the stylistic features of a given artwork and applies those features to a new image, creating a stylized output that resembles the artistic style of the original artwork.
# Importing Libraries
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint
%matplotlib inline
# Dataset Collection

# Data Preprocessing
Load and preprocess the dataset.
Resize, normalize, and prepare the images for training.
# Model Architecture
Using a pre-trained models like VGG as the basis for style transfer.
Define the layers and architecture suitable for extracting and transferring artistic styles.
# Computing cost and gram matrix
It is coming as 7.056877 and matrix as GA = 
tf.Tensor(
[[ 63.188793  -26.721273   -7.7320204]
 [-26.721273   12.76758    -2.5158243]
 [ -7.7320204  -2.5158243  23.752384 ]], shape=(3, 3), dtype=float32)
 Then compute layer style cost and it it coming as 14.017806
 After that I compute style cost and total cost
 
Implement training loops, loss functions for content preservation and style emulation, and optimization techniques.
Train the model using the prepared dataset.
