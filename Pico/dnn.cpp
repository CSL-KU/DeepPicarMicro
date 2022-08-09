// C++/Pico general includes
#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"

// DeepPicarMicro includes 
#include "pico_depthwise.h" // Reduced CNN model
#include "model_settings.h" // Parameters (image dimensions, etc.)

// TFLMicro includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 190000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Convert value from radians to degrees
float rad2deg(float rad)
{
	float deg = (180.0 * rad) / M_PI;
	return deg;
}

// Launch a corunner on core 1
// Currently modeled after BwWrite
void corunner()
{    
    int arrSize = 100000000;
    int values[arrSize];

    while (1)
    {
        for(int i=0; i < arrSize; i+=16)
        {
            values[i] = i;
        }
    }
}

int main(int argc, char* argv[]) {
  stdio_init_all();
  
  sleep_ms(1000); // Give time for USB print output to work
  
  // Create TFLMicro model
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(pico_nas_68x68x1_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }
  printf("Model copied\n");

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  printf("Interpreter built\n");

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return -2;
  }
  printf("Tensors allocated\n");

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
  
  // NOTE: uncomment to launch co-runner on core 1
  //multicore_launch_core1(corunner);
  
  // Run inferences until the program is stopped
  while(true)
  {
	  // Set fake input to model
	  for(int i = 0; i < 68*68; i++)
	  {
		  input->data.f[i] = 0;
	  }
	  
	  // Feed image to model, run inferencing
	  absolute_time_t startTime = get_absolute_time();
	  TfLiteStatus invoke_status = interpreter->Invoke();
	  if (invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
		return -3;
	  }
	  absolute_time_t infTime = get_absolute_time();
	  
	  // Get output and perform control actuation
	  TfLiteTensor* output = interpreter->output(0);
	  float angle = output->data.f[0];
	  float degrees = rad2deg(angle);
	  
	  // Print out timing information for current frame
	  absolute_time_t endTime = get_absolute_time();
	  absolute_time_t totalDur = (endTime - startTime) / 1000;
	  
	  printf("Inference %i --> %.2f rad, %.2f deg | inf: %llu ms\n", inference_count, angle, degrees, totalDur);
	  inference_count++; 
  }
  
}