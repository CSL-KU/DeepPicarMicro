import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TF model architecture
import params
modelname = params.modelname
m   = __import__(modelname)
model = m.model

# Define output model file names
#     First value is model size (e.g. "small")
model_file = "models/udacity/{}-{}x{}x{}/".format(modelname[6:], params.inputres[0], params.inputres[1], params.inputchannels)

# Print model weights and calculate model connections
from connections import calculate_connections
print("-----------------------------------------------------------------------")
print("MODEL WEIGHTS")
model.summary()
#quit()
calculate_connections(model)

# Imports
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from math import sqrt
import os
import random

if not os.path.exists(model_file):
    os.makedirs(model_file)

def deg2rad(deg):
    return deg * math.pi / 180.0

f = lambda x, y: x+y

# Train/test data lists
imgs = []
vals = []

resize_vals=params.inputres # Input dimensions
input_channels = params.inputchannels # Input channels, should be 1 (grayscale) or 3 (RGB)

# Load all train/test data into their respective lists
for i in [0,1,2,3,4,5]:
    imgs_cur = []
    vals_cur = []
    
    imgs_rev = []
    vals_rev = []
    
    vid = cv2.VideoCapture("data/epoch{}/out-video.avi".format(i))
    ret,img = vid.read()
    
    while(ret):        
        img = img[70:-25]
        
        # Convert to grayscale and readd channel dimension
        if input_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (params.inputres[0], params.inputres[1]))
            img = np.reshape(img, (resize_vals[1], resize_vals[0], 1))
        # For RGB, just need to resize image
        else:
            img = cv2.resize(img, (params.inputres[0], params.inputres[1]))
        img = img / 255.
        
        imgs_cur.append(img)
        ret,img = vid.read()
        
    temp = np.asarray(read_csv("data/epoch{}/driving_log.csv".format(i), header=None).iloc[:,3].values)
    vals_cur.extend(f(temp,0))
    
    
    print(len(imgs_cur), len(vals_cur))
    
    imgs.extend(imgs_cur)
    vals.extend(vals_cur)
    print(len(imgs), len(vals))    

imgs = np.asarray(imgs)
vals = np.asarray(vals)

assert len(imgs) == len(vals)

# Quantize and train the model
print("Train/Test")
imgs_train, imgs_test, vals_train, vals_test = train_test_split(imgs, vals, test_size=0.25, random_state=1)

print("Model Compile")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

print("Model Fit")
history = model.fit(imgs_train, vals_train, 
                    batch_size=128,  epochs=15, 
                    validation_data=(imgs_test, vals_test)) 
 
# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_file+"full-loss.png")

# Save both the Keras and TFLite models      
print("Model Save")                  
model.save(model_file+"full-model.h5")


print("TFLite Model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

with open(model_file+"full-model.tflite", 'wb') as f:
    f.write(quantized_tflite_model)

interpreter = tf.lite.Interpreter(model_path=model_file+"full-model.tflite")

interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

length = len(imgs_test)
predicted_angles = []
for i in tqdm(range(length)):
  img = imgs_test[i]
  img = np.expand_dims(img, axis=0).astype(np.float32)

  interpreter.set_tensor(input_index, img)
  interpreter.invoke()

  predicted_angle = interpreter.get_tensor(output_index)[0][0]
  predicted_angles.append(predicted_angle)

rmse = sqrt(mean_squared_error(vals_test, predicted_angles))

print("RMSE is {:.2f}".format(rmse))

