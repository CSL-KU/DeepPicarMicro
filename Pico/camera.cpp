// General C++/Pico includes
#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"

// DeepPicarMicro includes
#include "model_settings.h" // Misc parameters (image dimensions, etc.)
#include "image_provider.h" // Camera

// TFLMicro includes
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

int main(int argc, char* argv[]) {
  stdio_init_all(); // Initialize all stdio on the Pico
  
  sleep_ms(1000); // Give time for USB print output to work
  
  tflite::ErrorReporter* error_reporter = nullptr;
  
  float image[kNumCols * kNumRows];
  
  // NOTES
  // 0 - single image capture, print pixel values
  // 1 - continuous image capture, prints capture time for each image
  int continuous = 0;
  
  int imageNum = 0; // Keep track of number of images
  
  // Run CNN inferences until the car is stopped
  while(true)
  {
	  // Get image
	  absolute_time_t startTime = get_absolute_time();
	  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
								image)) {
		TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
	  }
	  absolute_time_t capTime = get_absolute_time();
	  
	  // Process image, need to divide all pixels by 255
	  for(int i = 0; i < kNumCols * kNumRows; i++)
	  {
		  image[i] = image[i] / 255.0;
		  if(continuous == 0)
			printf("%f, ", image[i]);
	  }
	  
	  if(continuous == 0)
		break;
	  
	  absolute_time_t endTime = get_absolute_time();
	  absolute_time_t totalDur = (endTime - startTime) / 1000;
	  
	  printf("Image %i --> cap: %llu ms\n", imageNum, totalDur);
	  imageNum++;
  }
  
}