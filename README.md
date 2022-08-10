# DeepPicarMicro

## Model Training

### Dependencies

In order to train the DeepPicar models the following Python modules are required:

	$ pip install tensorflow numpy pandas scikit-learn opencv-python tensorflow-model-optimization
	
The model architecture also needs to be manually changed in the following lines of the quant-train.py file:

	1 from model_small import model 
	...
	56 q_aware_model.save('models/quant_model_small.h5')
	...
	66 with open('models/quant-model-small.tflite', 'wb') as f:
	
For example, to train the large model you would want to change model_small and model-small in the above lines to model_large and model-large, respectively. 

### Dataset

The dataset used to train the models can be found at https://drive.google.com/file/d/1Fjwy-dLDp5sNilPTUMXUkmnUF1SrXDOl/view, or custom data can be used instead. In either case, the data should be placed in a new folder ModelCreation/Dataset.

### Keras/TFLite/TFLiteMicro Model Creation

To create both Keras and TFLite models for the chosen architecture, run the following:

	$ python quant-train.py
	
To create a TFLiteMicro compatible model representation, run the following command inside the ModelCreation/models/ directory:

	$ xxd -i quant-model-<size>.tflite > <model-name>.cc
	
Where <size> is the model architecture trained and <model-name> is the name of the new file.

## Neural Architecture Search (NAS)

For the NAS, we assume that the datasets are added to a different folder in the NAS directory called "NAS/Dataset/". We use the same file name conventions as in the ModelCreation folder, but with the following directory structure:

	```
	NAS
	|___Dataset
	|	|____<dataset #1>
	|	|	|____<epoch #1>
	|	|	|	|	out_video.avi
	|	|	|	|	out_key.avi
	|	|	|____<epoch #2>
	|	|	|	...
	|	|____<dataset #2>
	|	|	...
	```
	
This is to allow for multiple datasets to be present, but manually select which dataset to use for the NAS. This selection can be done by setting the "dataset" parameters in NAS/All/params.py to the name of the dataset. For example:

	dataset="<dataset #1>"	# Replace with actual dataset name

To perform a NAS for a given model backbone/dataset combination, the following commands can be run:

	$ python filter_models.py <max_connections>
	$ python train_pass.py
	$ ./create-cc.sh <dataset_name>

## Pico

Detailed instructions can be found in the [Pico directory](https://github.com/CSL-KU/DeepPicarMicro/tree/main/Pico).

## Udacity Simulator

Detailed instructions can be found in the [UdacitySim directory](https://github.com/CSL-KU/DeepPicarMicro/tree/main/UdacitySim).
	
