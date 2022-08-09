import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import large_nas_dw as m
import params
from util import networkValues
import tensorflow as tf
import tensorflow_model_optimization as tfmot
tf.get_logger().setLevel('ERROR')

target_kb = int(sys.argv[1]) # For the Pico, ~600,000 seems to be the max
target_weights = int(target_kb / 1.1)    # Scale weights to roughly match model filesize in kb (1.1 is relatively pessimistic)

output_folder = "nas_dw"

width_mult = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
h_len = 66
w_len = [66, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
d_len = 1

csv_file = open("models/{}_info.csv".format(output_folder), 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["parameter", "Weights", "Connections", "Inf Time"])

###############
# RESOLUTION
###############
print("RESOLUTION")
for w in w_len:
    model = m.createModel(w_len=w, d_len=d_len)
    [weights, connections] = networkValues(model)
    
    if(weights < target_weights):
        print("{} --> {}, {}".format(w, weights, connections))
        writer.writerow([w, weights, connections])
        
        quantize_model = tfmot.quantization.keras.quantize_model
        q_model = quantize_model(model)
        converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        with open("models/{}/model-resolution-{}.tflite".format(output_folder, str(w).zfill(3)), 'wb') as f:
            f.write(quantized_tflite_model)
        del(q_model)

###############
# WIDTH
###############
print("WIDTH")
for mult in width_mult:
    model = m.createModel(depth_multiplier=mult, d_len=d_len)
    [weights, connections] = networkValues(model)
    
    if(weights < target_weights):
        print("{} --> {}, {}".format(mult, weights, connections))
        writer.writerow([mult, weights, connections])
        
        quantize_model = tfmot.quantization.keras.quantize_model
        q_model = quantize_model(model)
        converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        with open("models/{}/model-width-{}.tflite".format(output_folder, mult), 'wb') as f:
            f.write(quantized_tflite_model)
        del(q_model)
csv_file.close()