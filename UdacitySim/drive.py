import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import socketio
import eventlet
import eventlet.wsgi
import params
from flask import Flask, render_template
from io import BytesIO
import base64
from PIL import Image
from PIL import ImageOps
import time

modelname = params.modelname
resize_vals=params.inputres # Input dimensions
input_channels = params.inputchannels # Input channels, should be 1 (grayscale) or 3 (RGB)

sio = socketio.Server()
app = Flask(__name__)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        img_str = data["image"]
        img = Image.open(BytesIO(base64.b64decode(img_str)))
        
        # frames incoming from the simulator are in RGB format
        img = cv2.cvtColor(np.asarray(img), code=cv2.COLOR_RGB2BGR)
        
        img = img[70:-25]
        if input_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (params.inputres[0], params.inputres[1]))
            img = np.reshape(img, (resize_vals[1], resize_vals[0], 1))
        # For RGB, just need to resize image
        else:
            img = cv2.resize(img, (params.inputres[0], params.inputres[1]))
        img = img / 255.
        
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        start = time.time()
        #steering_angle = model.predict(img)[0][0]
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        steering_angle = interpreter.get_tensor(output_index)[0][0]
        dur = (time.time() - start) * 1000
        
        throttle = 1.0
        print(steering_angle, throttle, dur)
        send_control(steering_angle, throttle)
    else:
        print("Nothing")

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)
  
if __name__ == '__main__':  
    #model = keras.models.load_model("models/udacity/{}-{}x{}x{}/full-model.h5".format(modelname[6:], params.inputres[0], params.inputres[1], params.inputchannels))
    interpreter = tf.lite.Interpreter("models/udacity/{}-{}x{}x{}/full-model.tflite".format(modelname[6:], params.inputres[0], params.inputres[1], params.inputchannels))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)