import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import params

def myround(x, base=4):
    return base * round(x/base)

# Construct model based on layer strings
def createModel(conv_str="1111", fc_str="111", depth_multiplier=1.0, h_len=66, w_len=66, d_len=1):
	# Define layers in terms of an array, use strings to index which layers get added
	conv2d_arr = []
	conv2d_arr.append(layers.Conv2D(myround(36 * depth_multiplier), (5,5), strides=(2,2), activation='relu'))
	conv2d_arr.append(layers.Conv2D(myround(48 * depth_multiplier), (5,5), strides=(2,2), activation='relu'))
	conv2d_arr.append(layers.Conv2D(myround(64 * depth_multiplier), (3,3), activation='relu'))
	conv2d_arr.append(layers.Conv2D(myround(64 * depth_multiplier), (3,3), activation='relu'))

	fc_arr = []
	fc_arr.append(layers.Dense(round(100 * depth_multiplier), activation='relu'))
	fc_arr.append(layers.Dense(round(50 * depth_multiplier), activation='relu'))
	fc_arr.append(layers.Dense(round(10 * depth_multiplier), activation='relu'))

	# Create model and add input Conv2D layer
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(myround(24 * depth_multiplier), (5,5), strides=(2,2), activation='relu', input_shape=(h_len,w_len,d_len), name="input"))

	# Add Conv2D layers
	for i in range(len(conv_str)):
		if int(conv_str[i]) == 1:
			model.add(conv2d_arr[i])

	# Add Flatten layer
	model.add(layers.Flatten())

	# Add FC layers
	for i in range(len(fc_str)):
		if int(fc_str[i]) == 1:
			model.add(fc_arr[i])

	# Add final output layer
	model.add(layers.Dense(1, name="output"))
	return model
