import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import params

# Construct model based on layer strings
def createModel(dw_conv_str="1111" , fc_str="111", depth_multiplier=1.0, h_len=66, w_len=66, d_len=1):

	dw_conv_arr = []
	dw_conv_arr.append(layers.DepthwiseConv2D((5,5), strides=(2, 2)))
	dw_conv_arr.append(layers.Conv2D(round(36 * depth_multiplier), (1, 1), strides=(1, 1)))

	dw_conv_arr.append(layers.DepthwiseConv2D((5,5), strides=(2, 2)))
	dw_conv_arr.append(layers.Conv2D(round(48 * depth_multiplier), (1, 1), strides=(1, 1)))

	dw_conv_arr.append(layers.DepthwiseConv2D((3,3), strides=(1, 1)))
	dw_conv_arr.append(layers.Conv2D(round(64 * depth_multiplier), (1, 1), strides=(1, 1)))

	dw_conv_arr.append(layers.DepthwiseConv2D((3,3), strides=(1, 1)))
	dw_conv_arr.append(layers.Conv2D(round(64 * depth_multiplier), (1, 1), strides=(1, 1)))

	fc_arr = []
	fc_arr.append(layers.Dense(round(100 * depth_multiplier), activation='relu'))
	fc_arr.append(layers.Dense(round(50 * depth_multiplier), activation='relu'))
	fc_arr.append(layers.Dense(round(10 * depth_multiplier), activation='relu'))

	# Create model and add input Conv2D layer
	model = tf.keras.Sequential()
	model.add(layers.Input(shape=(h_len,w_len,d_len), name='input'))
	model.add(layers.DepthwiseConv2D((5,5), strides=(2, 2), name='dwi'))
	model.add(layers.Conv2D(round(24 * depth_multiplier), (1, 1), strides=(1, 1), name='pwi'))
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())

	for i in range(0, len(dw_conv_str)):
		if int(dw_conv_str[i]) == 1:
			model.add(dw_conv_arr[i*2])
			model.add(dw_conv_arr[i*2+1])
			model.add(layers.BatchNormalization())
			model.add(layers.ReLU())

	# Add Flatten layer
	model.add(layers.Flatten())

	# Add FC layers
	for i in range(len(fc_str)):
		if int(fc_str[i]) == 1:
			model.add(fc_arr[i])

	# Add final output layer
	model.add(layers.Dense(1, name="output"))
	return model