import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import cv2
import glob
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.get_logger().setLevel('ERROR')
import tensorflow_model_optimization as tfmot
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import subprocess
import params
import gc
import math

modelname = params.modelname
m   = __import__(modelname)

def deg2rad(deg):
    return deg * math.pi / 180.0
def rad2deg(rad):
    return 180.0 * rad / math.pi
def get_action(angle_rad):
    degree = rad2deg(angle_rad)
    if degree < 15 and degree > -15:
        return "center"
    elif degree >= 15:
        return "right" 
    elif degree <-15:
        return "left"

def preprocess(img, resize_vals, input_channels):
	img = cv2.resize(img, (320, 240))
	img = cv2.flip(img, 1)
	img = img[0:210, 60:190]
	# Convert to grayscale and readd channel dimension
	if input_channels == 1:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (resize_vals[1], resize_vals[0]))
		img = np.reshape(img, (resize_vals[0], resize_vals[1], 1))
	# For RGB, just need to resize image
	else:
		img = cv2.resize(img, (resize_vals[1], resize_vals[0]))
	img = img / 255.
	return img

# Currently based on Raspberry Pi Pico
def predict_time(connections):
	m = 0.000236
	b = 22.189388
	inf = (m * connections) + b
	return inf

# Train/test data lists
imgs = []
vals = []

# Load train/test data
total_frames=params.totalframes
frame_num = 0
for folder in glob.glob("../../Dataset/{}/*".format(params.dataset)):
    print(folder)
    vid_file = glob.glob("{}/*.avi".format(folder))[0]
    vid = cv2.VideoCapture(vid_file)
    ret,img = vid.read()
    while(ret and frame_num < total_frames):
        # Can only preprocess images when model params are known
        imgs.append(img)
        ret,img = vid.read()
        frame_num += 1
    vid.release()

    csv_file = glob.glob("{}/*.csv".format(folder))[0]
    temp = np.asarray(read_csv(csv_file)["wheel"].values)[:total_frames]
    vals.extend(temp)

if params.dataset == "nvidia":
    vals = list(map(deg2rad, vals))
vals = np.asarray(vals, dtype=np.float16)
assert len(imgs) == len(vals)

# Open file for writing relevant statistics for all trained models
stats_file = open("{}/trained_models.csv".format(params.dataset), 'w', newline='')
writer = csv.writer(stats_file)
writer.writerow(["Pass", "conv_str", "fc_str", "width_mult", "h_len", "w_len", "d_len",
	"Weights", "Connections", "Loss", "Val loss", "Accuracy(%)", "SRAM usage(KB)", "Flash size(KB)",
	"Predicted inf time(ms)", "Inference time(ms)"])

# Open the file that contains the parameters for the models that passed
#	the filtering process
pass_loc = glob.glob("{}/pass*.csv".format(params.dataset))[0]
pass_file = open(pass_loc, 'r')
reader = csv.reader(pass_file)
for i,row in enumerate(reader):
	# Skip first row with column headers
	if i == 0:
		continue

	# Get model parameters for current architecture
	[conv_str, fc_str, width_mult, h_len, w_len, d_len, Connections, Weights] = row
	model_file = "{}/models/{}-{}_{}x{}x{}_{}/".format(params.dataset, conv_str, fc_str, w_len, h_len, d_len, width_mult)
	if not os.path.exists(model_file):
		os.makedirs(model_file)

	imgs_temp = np.array([preprocess(img, (int(h_len), int(w_len)), int(d_len)) for img in imgs], dtype=np.float16)
	imgs_train, imgs_test, vals_train, vals_test = train_test_split(imgs_temp, vals, test_size=0.35, random_state=0)
	class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(vals_train), y=vals_train)
	class_weights = {i:class_weights[i] for i in range(len(class_weights))}

	# Quantize and train the model
	quantize_model = tfmot.quantization.keras.quantize_model

	# Create new train and test datasets for the current architecture
	#	Train multiple models to try and pass the val_loss check
	for j in range(5):
		model = m.createModel(conv_str, fc_str, float(width_mult), int(h_len), int(w_len), int(d_len))
		q_aware_model = quantize_model(model)
		q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
		                loss=tf.keras.losses.MeanSquaredError())
		history = q_aware_model.fit(imgs_train, vals_train,
		                    batch_size=params.batch_size,  epochs=params.epochs, # steps_per_epoch=24,
		                    validation_data=(imgs_test, vals_test), class_weight=class_weights)

		# Exit the training loop if the val_loss check is met or val_loss is too high
		if history.history['val_loss'][-1] <= params.val_loss or history.history['val_loss'][-1] > params.val_high or j==4:
			break

		del(model)
		del(q_aware_model)
		del(history)
		gc.collect()

	# Save the model as H5 and TFLite files
	q_aware_model.save(model_file+"model.h5")
	converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	quantized_tflite_model = converter.convert()
	with open(model_file+"model.tflite", 'wb') as f:
		f.write(quantized_tflite_model)

	# Plot training and validation losses
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
            
	# Load the TFLite model
	interpreter = tf.lite.Interpreter(model_path=model_file+"model.tflite")
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

	# Test the TFLite model
	length = len(imgs_test)
	interpreter.resize_tensor_input(input_index, [length, h_len, w_len, d_len])
	interpreter.allocate_tensors()
	interpreter.set_tensor(input_index, np.array(imgs_test, dtype=np.float32))
	interpreter.invoke()
	predicted_angles = interpreter.get_tensor(output_index)
	predicted = np.array(list(map(get_action, predicted_angles)))
	ground = np.array(list(map(get_action, vals_test)))
	accuracy = np.mean(predicted==ground)*100

	# Print out and write model statistics to csv file
	print("Model", i)
	val_loss = "{:.4f}".format(history.history["val_loss"][-1])
	loss = "{:.4f}".format(history.history["loss"][-1])
	print("Accuracy is {:.2f}%".format(accuracy))
	val = subprocess.Popen(["./find-arena-size", model_file+"model.tflite"], stdout=subprocess.PIPE)
	sram_size = "{:.2f}".format(float(val.communicate()[0].decode("UTF-8")))
	flash_size = "{:.2f}".format((os.path.getsize(model_file+"model.tflite"))/1024.)
	predict_inf = "{:.2f}".format(predict_time(int(Connections)))
	did_model_pass = 1 if float(val_loss) <= params.val_loss else 0
	print("SRAM size = {} kb".format(sram_size))
	print("Flash size = {} kb".format(flash_size))
	print("Predicted inference time = {} ms".format(predict_inf))
	print("MODEL PASS  {}".format(did_model_pass))
	writer.writerow([did_model_pass, conv_str, fc_str, width_mult, h_len, w_len, d_len,
		Connections, Weights, loss, val_loss, accuracy, sram_size, flash_size, predict_inf])

	# If the model passes the val loss check, create an empty file in the model directory
	#	Used by create-cc.sh to determine which models to add to the .cpp/.h files
	if float(val_loss) <= params.val_loss:
		empty = open(model_file+"pass.txt", 'w')
		empty.close()

	del(model)
	del(q_aware_model)
	del(history)
	del(imgs_temp)
	del(imgs_train)
	del(imgs_test)
	del(vals_train)
	del(vals_test)
	gc.collect()

stats_file.close()
