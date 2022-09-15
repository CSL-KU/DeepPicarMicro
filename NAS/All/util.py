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
        elif(type(layer) == tf.keras.layers.DepthwiseConv2D):
            input_depth = layer.input.shape[3]
            output_shape = layer.output.shape
            kernel_size = layer.kernel_size
            layer_connections = input_depth * output_shape[1] * output_shape[2] * kernel_size[0] * kernel_size[1]
            print("{} (DWConv2D)\t\t{}*{}*{}*{}*{}\t\t{}".format(layer_num, input_depth, output_shape[1], output_shape[2], kernel_size[0], kernel_size[1], layer_connections))
            print("-----------------------------------------------------------------------")
        else:
            continue    # Skip to next layer
           
        # Add current layer connections to total model connections, iterate layer number
        connections += layer_connections
        layer_num += 1
    
    # Display the total number of layer connections
    print("Total connections: {}".format(connections))
    print("=======================================================================")
    

# Calculate total model connections
#   Input: TF model
#   Output: Total model weights and connections
def networkValues(model):
    weights = 0 # Total number of weights
    connections = 0  # Total number of connections
    
    for layer in model.layers:
        layer_connections = 0 # Number of connections in current layer
        
        # Only calculate weights and connections for Conv2D and Dense (FC) layers
        if(type(layer) == tf.keras.layers.Conv2D):
            input_depth = layer.input.shape[3]
            output_shape = layer.output.shape
            kernel_size = layer.kernel_size
            
            temp = input_depth * kernel_size[1] * kernel_size[1] * output_shape[3]
            layer_weights = temp + output_shape[3]
            layer_connections = temp * output_shape[1] * output_shape[2]
        elif(type(layer) == tf.keras.layers.Dense):
            input_neurons = layer.input.shape[1]
            output_neurons = layer.output.shape[1]
            layer_connections = input_neurons * output_neurons
            layer_weights = layer_connections + output_neurons
        elif(type(layer) == tf.keras.layers.DepthwiseConv2D):
            input_depth = layer.input.shape[3]
            output_shape = layer.output.shape
            kernel_size = layer.kernel_size
            
            temp = input_depth * kernel_size[1] * kernel_size[1] #* output_shape[3]
            layer_weights = temp + output_shape[3]
            layer_connections = temp * output_shape[1] * output_shape[2]
        elif(type(layer) == tf.keras.layers.BatchNormalization):
            layer_weights = layer.input.shape[3] * 4
            layer_connections = 0
        else:
            continue    # Skip to next layer
           
        # Add current layer connections to total model connections, iterate layer number
        weights += layer_weights
        connections += layer_connections
    
    # Return total number of network connections
    return [weights,connections]
    
