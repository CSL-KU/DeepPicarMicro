import numpy as np
import math
import glob
import csv
import cv2
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import params
import large_nas as m
import gc

def deg2rad(deg):
    return deg * math.pi / 180.0
def rad2deg(rad):
    return 180.0 * rad / math.pi

def preprocess(img, resize_vals, input_channels):
    # Convert to grayscale and readd channel dimension
    if input_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_vals)
        img = np.reshape(img, (resize_vals[1], resize_vals[0], 1))
    # For RGB, just need to resize image
    else:
        img = cv2.resize(img, resize_vals)
    img = img / 255.
    return img

imgs = []
vals = []
resize_vals = params.inputres
input_channels = params.inputchannels
# Load train/test data
total_frames=params.totalframes
frame_num=0
for folder in glob.glob("../Dataset/{}/*".format(params.dataset)):
	print(folder)
	vid_file = glob.glob("{}/*.avi".format(folder))[0]
	vid = cv2.VideoCapture(vid_file)
	ret,img = vid.read()
	while(ret and frame_num < total_frames):
		# Can only preprocess images when model params are known
		imgs.append(preprocess(img, resize_vals, input_channels))
		ret,img = vid.read()
		frame_num += 1
	vid.release()

	csv_file = glob.glob("{}/*.csv".format(folder))[0]
	temp = np.asarray(read_csv(csv_file)["wheel"].values)[:total_frames]
	vals.extend(temp)

if params.dataset == "nvidia":
    vals = list(map(deg2rad, vals))
imgs = np.asarray(imgs, dtype=np.float16)
vals = np.asarray(vals, dtype=np.float16)
print(len(imgs), len(vals))
assert len(imgs) == len(vals)

imgs_train, imgs_test, vals_train, vals_test = train_test_split(imgs, vals, test_size=0.35, random_state=0)

print(len(imgs_train), len(vals_train))
print(len(imgs_test), len(vals_test))
#quit()

# Quantize and train the model
quantize_model = tfmot.quantization.keras.quantize_model

# Create new train and test datasets for the current architecture
#	Train multiple models to try and pass the val_loss check
val_losses = []
for i in range(5):
    model = m.createModel("1111", "111", 1.0, 66, 200, 3)
    q_aware_model = quantize_model(model)
    q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss=tf.keras.losses.MeanSquaredError())
    history = q_aware_model.fit(imgs_train, vals_train,
                        batch_size=params.batch_size,  epochs=params.epochs, # steps_per_epoch=24,
                        validation_data=(imgs_test, vals_test))

    val_losses.append(history.history['val_loss'][-1])

    del(model)
    del(q_aware_model)
    del(history)
    gc.collect()

print(val_losses)
print("{:.4f}".format(np.average(val_losses)))
