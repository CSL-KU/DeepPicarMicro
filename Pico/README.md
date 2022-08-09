## Pico

We assume that a Raspberry Pi 3/4 system is used for building the Pico code. As such, the Pico SDK can be installed by running the following commands:

	$ wget https://raw.githubusercontent.com/raspberrypi/pico-setup/master/pico_setup.sh
	$ chmod +x pico_setup.sh
	$ ./pico_setup.sh
	
For other build systems, please refer to [Getting started with Raspberry Pi Pico](https://datasheets.raspberrypi.org/pico/getting-started-with-pico.pdf).

Once the SDK is installed, run the following commands from this directory:

	$ git clone https://github.com/raspberrypi/pico-tflmicro
	$ patch pico-tflmicro/CMakeLists.txt < tflmicro-cmake.patch
	$ mkdir build; cd build
	$ cmake ..
	$ make -j4
	
The above patch isn't strictly necessary, but will reduce build times by skipping building the examples and tests in the pico-tflmicro repository.
	
This should result in several uf2 files being created. The main one is deeppicar.uf2 which drives the DeepPicarMicro using camera inputs and CNN outputs. This can then be loaded to a Pico board using the picotool utility (on a Pi 4 this is installed as part of the pico_setup.sh script):

	$ sudo picotool load deeppicar.uf2
	
There are two other programs currently present: (1) camera.uf2, which can be used to test and debug camera outputs, and (2) dnn.uf2, which runs the CNN on fake data (i.e. the camera isn't used).

For the camera program, it can either take a single image and output it to the user or continually grab images and print the capture time for each image. Which mode it performs in can be controlled by changing the following variable in camera.cpp (0 for single image, 1 for continuous capture):

	26   int continuous = 0;
	
Finally, to see the program output, reset the Pico and open a new screen terminal on the corresponding /dev/ location (e.g. /dev/ttyACM0):

	$ tio /dev/ttyACM0		# change /dev/ location as necessary