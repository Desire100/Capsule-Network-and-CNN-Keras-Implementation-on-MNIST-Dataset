import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Reshape, Lambda, Layer, Flatten
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from keras import initializers
from keras.utils import to_categorical
from keras.layers.core import Activation