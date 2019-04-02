#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=/home/tianran/tensorflow/nets

# for float tflite, both works, while 'bazel run' need to be run in tensorflow dir
tflite_convert \
    --graph_def_file=$OUTPUT_DIR/p_net_v36.pb \
    --output_file=$OUTPUT_DIR/p_net_v36.tflite \
    --input_shapes=1,24,24,3 \
    --input_arrays='input_image' \
    --output_arrays='P_Net/conv5-1/conv2d/BiasAdd','P_Net/conv5-2/conv2d/BiasAdd'  \
    --inference_type=FLOAT \
    --allow_custom_ops \


# for float tflite, both works, while 'bazel run' need to be run in tensorflow dir
tflite_convert \
    --graph_def_file=$OUTPUT_DIR/r_net_v36.pb \
    --output_file=$OUTPUT_DIR/r_net_v36.tflite \
    --input_shapes=1,48,48,3 \
    --input_arrays='input_image' \
    --output_arrays='R_Net/conv5-1/conv2d/BiasAdd','R_Net/conv5-2/conv2d/BiasAdd','R_Net/conv5-3/conv2d/BiasAdd'  \
    --inference_type=FLOAT \
    --allow_custom_ops \