## Udacity Simulator

This folder contains scripts for working with the Udacity self-driving car simulator. 

### Dataset Creation

Data samples can be manually collected in the simulator, and we assume that the samples will written to a folder called "data". The camera images can then be converted to one or more videos by running:

	$ python create_video.py
	
This will create an AVI file in each subdirectory for the "data" folder. 

Once created, the "data" folder can be copied to the Dataset/ directory and renamed.

### Model Training

Individual models can be trained by running:

	$ python train.py
	
This is similar to the training process in the ModelCreation directory. 