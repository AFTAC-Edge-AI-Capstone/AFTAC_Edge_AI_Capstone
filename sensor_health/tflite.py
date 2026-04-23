import tensorflow as tf
import numpy as np

def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    return interpreter

def quantize_input(x, input_details):
    scale, zero_point = input_details[0]["quantization"]
    
    x = x.astype(np.float32)
    x_q = x / scale + zero_point
    return x_q.astype(np.int8)

def dequantize_output(y, output_details):
    scale, zero_point = output_details[0]["quantization"]
    
    y = y.astype(np.float32)
    return (y - zero_point) * scale

def tflite_predict(interpreter, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    preds = []

    for i in range(len(X)):
        x = X[i:i+1]

        # Quantize input
        x_q = quantize_input(x, input_details)

        interpreter.set_tensor(input_details[0]["index"], x_q)
        interpreter.invoke()

        y_q = interpreter.get_tensor(output_details[0]["index"])
        y = dequantize_output(y_q, output_details)

        preds.append(y.squeeze())

    return np.array(preds)