import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
import numpy as np

model = tf.keras.Sequential()
model.add(layers.DepthwiseConv2D((5,5), strides=(2,2), activation='relu', input_shape=(66,200,3)))
model.add(layers.Conv2D(24, (1,1), strides=(1,1), activation='relu'))

model.add(layers.DepthwiseConv2D((5,5), strides=(2,2), activation='relu'))
model.add(layers.Conv2D(36, (1,1), strides=(1,1), activation='relu'))

model.add(layers.DepthwiseConv2D((5,5), strides=(2,2), activation='relu'))
model.add(layers.Conv2D(48, (1,1), strides=(1,1), activation='relu'))

model.add(layers.DepthwiseConv2D((3,3), activation='relu'))
model.add(layers.Conv2D(64, (1,1), activation='relu'))

model.add(layers.DepthwiseConv2D((3,3), activation='relu'))
model.add(layers.Conv2D(64, (1,1), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))