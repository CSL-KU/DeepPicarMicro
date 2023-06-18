# DeepPicarMicro

## Demo

[![DeepPicarMicro demo video](https://img.youtube.com/vi/tpxJCQZ17Os/0.jpg)](https://www.youtube.com/watch?v=tpxJCQZ17Os)

## Model Training

### Dependencies

In order to train the DeepPicar models the following Python modules are required:

	$ pip install tensorflow numpy pandas scikit-learn opencv-python tensorflow-model-optimization tqdm
	
The model architecture can be set by changing the modelname field in any params.py:

	modelname = "pilotnet"
	
For example, to train the PilotNet model with Depthwise Separable layers, change "pilotnet" to "pilotnet_depthwise". 

### Dataset

We assume that all datasets are located in a directory named "Dataset/" with the following structure:

	```
	Dataset
	|____<dataset #1>
	|	|____<epoch #1>
	|	|	|	out_video.avi
	|	|	|	out_key.avi
	|	|____<epoch #2>
	|	|	...
	|____<dataset #2>
	|	...
	```
	
The dataset used for training can then be changed in any params.py::

	dataset="<dataset #1>"	# Replace with actual dataset name

The datasets used to train the models in the paper can be found at:
- Real-world dataset: https://drive.google.com/file/d/1Fjwy-dLDp5sNilPTUMXUkmnUF1SrXDOl/view
- Udacity simulator dataset: https://drive.google.com/file/d/1WW0r-Zx0_sPULpsV_44i-M1zZhQwX6ao/view

### Keras/TFLite/TFLiteMicro Model Creation

To create both Keras and TFLite models for the chosen architecture, run the following:

	$ python quant-train.py
	
To create a TFLiteMicro compatible model representation, run the following command inside the desired models directory:

	$ xxd -i quant-model.tflite <model-name>.cc
	
Where <model-name> is the name of the new file.

## Neural Architecture Search (NAS)

To perform a NAS for a given model backbone/dataset combination, the following commands can be run:

	$ python filter_models.py <max_MACs>
	$ python train_pass.py
	$ ./create-cc.sh <dataset_name>
	
Note: The NAS process can be resource intensive in terms of system and GPU memory. In the paper, we used a PC with 64GB of system memory and a GPU with 10GB of memory.

## Pico

Detailed instructions can be found in the [Pico directory](https://github.com/CSL-KU/DeepPicarMicro/tree/main/Pico).

## Udacity Simulator

Detailed instructions can be found in the [UdacitySim directory](https://github.com/CSL-KU/DeepPicarMicro/tree/main/UdacitySim).
	
