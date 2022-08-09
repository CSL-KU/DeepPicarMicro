## Udacity Simulator

This folder contains scripts for working with the Udacity self-driving car simulator. 

### Dataset Creation

Data samples can be manually collected in the simulator, and we assume that the samples will written to a folder called "data". The camera images can then be converted to one or more videos by running:

	$ python create_video.py
	
This will create an AVI file in each subdirectory for the "data" folder. 

The dataset used to train the models in the paper can be found at https://drive.google.com/file/d/1Da-RPCvNleqegYoPW-8H1ZRSdwe90d5-/view?usp=sharing.

### Model Training

Individual models can be trained by running:

	$ python train.py
	
This is similar to the training process in the ModelCreation directory. 

Note: To perform a NAS on the Udacity simulator dataset, it should be copied to the Dataset directory. From there, the same NAS steps can be followed.