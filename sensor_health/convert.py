import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from utils import load_data
from matplotlib import pyplot as plt
from config import DATASETS, MAX_RUL, WINDOW

train_X, val_X, test_X, train_y, val_y, test_y, num_features = load_data(DATASETS, WINDOW, MAX_RUL)

def representative_dataset():
    for i in range(100):
        yield [train_X[i:i+1].astype("float32")]

# -----------------------------
# Load the model and converter
# -----------------------------

keras_model = tf.keras.models.load_model('models/maintenance_model_optimized')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# -------------------------
# Quantize to int8
# -------------------------

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

converter.representative_dataset = representative_dataset

tflite_int8_model = converter.convert()

# -------------------------------
# Load and set up for prediction
# -------------------------------

with open("models/maintenance_model_int8.tflite", "wb") as f:
    f.write(tflite_int8_model)

interpreter = tf.lite.Interpreter(model_path="models/maintenance_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def quantize_input(x, input_details):
    scale, zero_point = input_details[0]["quantization"]
    
    x = x.astype(np.float32)
    x_q = x / scale + zero_point
    return x_q.astype(np.int8)

def dequantize_output(y, output_details):
    scale, zero_point = output_details[0]["quantization"]
    
    y = y.astype(np.float32)
    return (y - zero_point) * scale

def tflite_predict(X):
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

# --------------------------------
# Predict and analyze predictions
# --------------------------------

true_y = test_y * MAX_RUL
keras_y = keras_model.predict(test_X).flatten() * MAX_RUL
tflite_y = tflite_predict(test_X) * MAX_RUL

def nasa_score(true_y, pred_y):
    errors = pred_y - true_y
    score = 0.0
    
    for e in errors:
        if e < 0:
            score += np.exp(-e / 13.0) - 1
        else:
            score += np.exp(e / 10.0) - 1
            
    return score

print(f'Score: {nasa_score(true_y, keras_y):.3f}')

index_map = sorted(range(len(true_y)), key=lambda i: true_y[i])
true_y = [true_y[index_map[i]] for i in range(len(true_y))]
keras_y = [keras_y[index_map[i]] for i in range(len(true_y))]
tflite_y = [tflite_y[index_map[i]] for i in range(len(true_y))]

plt.figure(figsize=(12, 5))
plt.plot(true_y, label='True RUL', color='blue', marker='o', markersize=3)
plt.plot(keras_y, label='Keras RUL', color='red', linestyle='--')
plt.plot(tflite_y, label='TFLite RUL', color='green', linestyle='--')
plt.xlabel('Engine Unit Number')
plt.ylabel('Remaining Cycles')
plt.title('Actual vs Predicted RUL')
plt.legend()
plt.show()