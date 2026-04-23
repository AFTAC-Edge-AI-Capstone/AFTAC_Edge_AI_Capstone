import tensorflow as tf
import tensorflow_model_optimization as tfmot
from utils import load_data, evaluate_model

# -------
# Config
# -------
WINDOW = 30
MAX_RUL = 130
DATASETS = [1]

# ----------
# Load data
# ----------
train_X, val_X, test_X, train_y, val_y, test_y, num_features = load_data(DATASETS, WINDOW, MAX_RUL, "data")

# -------------------
# Load trained model
# -------------------
model = tf.keras.models.load_model('maintenance_model.keras')

# --------
# PRUNING
# --------
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,
        final_sparsity=0.5,
        begin_step=0,
        end_step=len(train_X) // 64 * 10  # ~10 epochs worth
    )
}

model_pruned = prune_low_magnitude(model, **pruning_params)

model_pruned.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
]

model_pruned.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=10,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Strip pruning wrappers for export
model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)

# ----
# QAT
# ----

def apply_qat(model):
    def quantize_layer(layer):
        if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.BatchNormalization, tf.keras.layers.MaxPooling1D)):
            return layer
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=quantize_layer
    )

    return tfmot.quantization.keras.quantize_apply(annotated_model)

model_qat = apply_qat(model_pruned)

model_qat.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)
model_qat.summary()

model_qat.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=10,
    batch_size=64,
    verbose=1
)
model_qat.save('models/maintenance_model_optimized', save_format="tf")

evaluate_model(MAX_RUL, model_qat, test_X, test_y)
