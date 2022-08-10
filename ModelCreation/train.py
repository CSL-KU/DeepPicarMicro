import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TF model architecture
import params
modelname = params.modelname
m   = __import__(modelname)
model = m.model

# Define output model file names
#     First value is model size (e.g. "small")
model_file = "models/full/{}-{}x{}x{}/".format(modelname, params.inputres[0], params.inputres[1], params.inputchannels)

# Print model weights and calculate model connections
from connections import calculate_connections
print("-----------------------------------------------------------------------")
print("MODEL WEIGHTS")
model.summary()
calculate_connections(model)

# Imports
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import glob

if not os.path.exists(model_file):
    os.makedirs(model_file)

# Train/test data lists
imgs = []
vals = []
resize_vals=params.inputres # Input dimensions
input_channels = params.inputchannels # Input channels, should be 1 (grayscale) or 3 (RGB)

# Load all train/test data into their respective lists
for i in range(0,10):
    vid_file = glob.glob("Dataset/{}/epoch{}/*.avi".format(params.dataset, i))[0]
    vid = cv2.VideoCapture(vid_file)
    ret,img = vid.read()
    while(ret):
        # Image translation to match Micro's camera
        img = cv2.resize(img, (320,240))
        img = cv2.flip(img, 1)
        img = img[0:210,60:190]
    
        # Convert to grayscale and readd channel dimension
        if input_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, resize_vals)
            img = np.reshape(img, (resize_vals[1], resize_vals[0], 1))
        # For RGB, just need to resize image
        else:
            img = cv2.resize(img, resize_vals)
        img = img / 255.
        imgs.append(img)
        ret,img = vid.read()
    csv_file = glob.glob("Dataset/{}/epoch{}/*.csv".format(params.dataset, i))[0]
    temp = np.asarray(read_csv(csv_file)["wheel"].values)
    vals.extend(temp)
    print(len(imgs), len(vals))    

# Convert lists to numpy arrays and ensure they are of equal length    
imgs = np.asarray(imgs)
vals = np.asarray(vals)
assert len(imgs) == len(vals)

# Train the model
print("Train/Test")
imgs_train, imgs_test, vals_train, vals_test = train_test_split(imgs, vals, test_size=0.25, stratify=vals, random_state=1)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(vals_train), y=vals_train)
class_weights = {i:class_weights[i] for i in range(len(class_weights))}

print("Model Compile")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='mse')

print("Model Fit")
history = model.fit(imgs_train, vals_train, 
                    batch_size=128,  epochs=15, # steps_per_epoch=24, 
                    validation_data=(imgs_test, vals_test),
                    class_weight=class_weights)
 
# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_file+"loss.png")

# Save both the Keras and TFLite models      
print("Model Save")                  
model.save(model_file+"full-model.h5")

print("TFLite Model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_file+"full-model.tflite", 'wb') as f:
    f.write(tflite_model)

# Helper functions for evaluating model accuracy
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

interpreter = tf.lite.Interpreter(model_path=model_file+"full-model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

length = len(imgs_test)
accuracy = 0 
for i in tqdm(range(length)):
  img = imgs_test[i]
  img = np.expand_dims(img, axis=0).astype(np.float32)

  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predicted_angle = interpreter.get_tensor(output_index)[0][0]
  groundtrue_angle = vals_test[i]

  if get_action(predicted_angle) == get_action(groundtrue_angle):
    accuracy = accuracy + 1

print(f"\nAccuracy is {(accuracy/length)*100:.2f}%")
