import numpy as np
import tensorflow as tf
import time

MODEL_PATH = 'r_net_v36.tflite'
# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)
# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
start = time.time()
interpreter.invoke()
print("invoke time: ", time.time()-start)
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
