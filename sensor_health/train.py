import pandas as pd
from utils import load_data, evaluate_model
from config import DATASETS, MAX_RUL, WINDOW

pd.set_option('display.max_columns', None)

# -------
# Config
# -------

EPOCHS = 35

train_X, val_X, test_X, train_y, val_y, test_y, num_features = load_data(DATASETS, WINDOW, MAX_RUL, "data")

# Import here, since tensorflow takes a while to import.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D

# -----------------
# Define the model
# -----------------

model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(WINDOW, num_features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Conv1D(filters=32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Flatten(),

    Dense(32, activation='relu'),
    Dropout(0.5),

    Dense(1)
])

model.compile(optimizer='adam', loss='mae', metrics=['mse'])
model.summary()

# -------------------------
# Actually train the model
# -------------------------

model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=EPOCHS,
    batch_size=64,
    verbose=1,
    shuffle=True
)

model.save('models/maintenance_model.keras')

evaluate_model(MAX_RUL, model, test_X, test_y)
