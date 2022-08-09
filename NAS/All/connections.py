# Needed for layer definitions
import tensorflow as tf

# Calculate per-layer and total model connections
#   Input: TF model
#   Output: Table with per-layer and total model connections
def calculate_connections(model):
    # Table headers
    print("-----------------------------------------------------------------------")
    print("MODEL CONNECTIONS")
    print("=======================================================================")
    print("Layer (type)\t\tConnection Formula\tConnections")
    print("=======================================================================")

    layer_num = 1   # Current layer
    connections = 0  # Number of connections in current layer
    for layer in model.layers:
        layer_connections = 0
        
        # Only calculate connections for Conv2D and Dense (FC) layers
        if(type(layer) == tf.keras.layers.Conv2D):
            input_depth = layer.input.shape[3]
            output_shape = layer.output.shape
            kernel_size = layer.kernel_size
            layer_connections = input_depth * output_shape[1] * output_shape[2] * output_shape[3] * kernel_size[0] * kernel_size[1]
            print("{} (Conv2D)\t\t{}*{}*{}*{}*{}*{}\t\t{}".format(layer_num, input_depth, output_shape[1], output_shape[2], output_shape[3], kernel_size[0], kernel_size[1], layer_connections))
            print("-----------------------------------------------------------------------")
        elif(type(layer) == tf.keras.layers.Dense):
            input_neurons = layer.input.shape[1]
            output_neurons = layer.output.shape[1]
            layer_connections = input_neurons * output_neurons
            extra_tab = ""
            if(len("{}*{}".format(input_neurons, output_neurons)) < 8):
                extra_tab="\t"
            print("{} (Conv2D)\t\t{}*{}{}\t\t{}".format(layer_num, input_neurons, output_neurons, extra_tab, layer_connections)) 
            print("-----------------------------------------------------------------------")
        else:
            continue    # Skip to next layer
           
        # Add current layer connections to total model connections, iterate layer number
        connections += layer_connections
        layer_num += 1
    
    # Display the total number of layer connections
    print("Total connections: {}".format(connections))
    print("=======================================================================")