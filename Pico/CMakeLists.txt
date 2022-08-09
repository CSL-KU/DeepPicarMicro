cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(deeppicarmicro C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

aux_source_directory(. DIR_SRCS)
add_subdirectory(src)
add_subdirectory(pico-tflmicro)

add_library(links INTERFACE)
target_link_libraries(links INTERFACE
  pico-tflmicro
  arducam
  pico_stdlib
  pico_multicore
  hardware_pwm
  hardware_i2c
  hardware_spi
)

# DEEPPICAR EXECUTABLE
# main executable, used for autonomous driving
add_executable(deeppicar
	main.cpp
	image_provider.cpp
	pico_depthwise.cpp
	model_settings.h
	image_provider.h
	pico_depthwise.h
)
target_link_libraries(deeppicar links)
pico_enable_stdio_usb(deeppicar 1)
pico_enable_stdio_uart(deeppicar 0)
pico_add_extra_outputs(deeppicar)

# CAMERA EXECUTABLE
# used to test/debug the camera module
add_executable(camera
	camera.cpp
	image_provider.cpp
	model_settings.h
	image_provider.h
)
target_link_libraries(camera links)
pico_enable_stdio_usb(camera 1)
pico_enable_stdio_uart(camera 0)
pico_add_extra_outputs(camera)

# DNN EXECUTABLE
# used to evaluate DNN inferencing time on fake data
add_executable(dnn
	dnn.cpp
	pico_depthwise.cpp
	model_settings.h
	pico_depthwise.h
)
target_link_libraries(dnn links)
pico_enable_stdio_usb(dnn 1)
pico_enable_stdio_uart(dnn 0)
pico_add_extra_outputs(dnn)