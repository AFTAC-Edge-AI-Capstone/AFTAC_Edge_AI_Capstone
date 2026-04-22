import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="models/maintenance_model_int8.tflite")

# Total size of tensor buffers (approx model in-memory footprint)
total_bytes = 0

for tensor in interpreter.get_tensor_details():
    shape = tensor["shape"]
    dtype = tensor["dtype"]

    # compute number of elements
    num_elements = np.prod(shape)

    # bytes per element
    bytes_per_element = np.dtype(dtype).itemsize

    total_bytes += num_elements * bytes_per_element

print(f"Approx in-memory tensor size: {total_bytes / 1024:.2f} KB")