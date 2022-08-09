## PilotNet Model Creation

This folder contains scripts for training PilotNet models. This includes both the original 32-bit model (train.py) and the 8-bit quantized model (quant-train.py).

Note: It is assumed that the dataset used for training is placed in a folder called "Dataset/" with the following file naming conventions:
- Videos containing input images => out-video-<epoch#>.avi
- CSV files containing input steering angle => out-key-<epoch#>.avi
Where <epoch#> is the integer value of the epoch associated with those respective files.

To train the 32-bit version of PilotNet, run the following command:

	$ python train.py
	
Likewise, to train the 8-bit quantized version, run this command instead:

	$ python quant-train.py