import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def load_data(DATASETS, WINDOW, MAX_RUL):
    # ---------------------
    # Load data from files
    # ---------------------

    train = pd.DataFrame(columns=['unit', 'time'])
    test = pd.DataFrame(columns=['unit', 'time'])
    RUL = pd.DataFrame()

    train_units = 0
    test_units = 0

    for i in DATASETS:
        df = pd.read_csv(f'data/train_FD00{i}.txt', sep=r'\s+', header=None)
        df.rename(columns={0: 'unit', 1: 'time'}, inplace=True)
        df['unit'] += train_units
        train_units += len(df['unit'].unique())
        train = pd.concat([train, df], ignore_index=True)

        df = pd.read_csv(f'data/test_FD00{i}.txt', sep=r'\s+', header=None)
        df.rename(columns={0: 'unit', 1: 'time'}, inplace=True)
        df['unit'] += test_units
        test_units += len(df['unit'].unique())
        test = pd.concat([test, df], ignore_index=True)

        df = pd.read_csv(f'data/RUL_FD00{i}.txt', sep=r'\s+', header=None)
        RUL = pd.concat([RUL, df], ignore_index=True)

    train.rename(columns={0: 'unit', 1: 'time'}, inplace=True)
    test.rename(columns={0: 'unit', 1: 'time'}, inplace=True)

    data_max_window = min(
        train.groupby('unit')['time'].max().min(),
        test.groupby('unit')['time'].max().min()
    )
    if data_max_window < WINDOW:
        print(f'Window too small for data (actual {data_max_window}, requested {WINDOW})')

    # ---------------------------------------
    # Build windows for training and testing
    # ---------------------------------------

    def _build_sequences(df, test=False):
        X, y, units = [], [], []

        for unit in df['unit'].unique():
            unit_df = df[df['unit'] == unit]

            base_features = unit_df.drop(['unit', 'time'], axis=1)
            delta = base_features.diff().fillna(0)
            features = pd.concat([base_features, delta], axis=1).values

            if test:
                rul = np.clip(RUL[0][unit-1], 0, MAX_RUL) / MAX_RUL
                X.append(features[-WINDOW:])
                y.append(rul)
            else:
                rul = np.arange(len(features) - 1, -1, -1)

                rul = np.clip(rul, 0, MAX_RUL) / MAX_RUL

                max_start = len(features) - WINDOW
                for i in range(max_start):
                    X.append(features[i:i+WINDOW])
                    y.append(rul[i+WINDOW])
                    units.append(unit)

        return np.array(X), np.array(y), np.array(units)

    train_X, train_y, train_units = _build_sequences(train)
    test_X, test_y, _ = _build_sequences(test, test=True)

    # -------------------------------------------------
    # Split training into training and validation data
    # -------------------------------------------------

    unique_units = np.unique(train_units)
    train_units_split, val_units_split = train_test_split(unique_units, test_size=0.2, random_state=0)

    train_mask = np.isin(train_units, train_units_split)
    val_mask = np.isin(train_units, val_units_split)

    val_X = train_X[val_mask]
    val_y = train_y[val_mask]

    train_X = train_X[train_mask]
    train_y = train_y[train_mask]

    # ---------------
    # Scale features
    # ---------------

    num_features = train_X.shape[2]

    scaler = StandardScaler()

    train_X_reshaped = train_X.reshape(-1, num_features)
    val_X_reshaped = val_X.reshape(-1, num_features)
    test_X_reshaped = test_X.reshape(-1, num_features)

    train_X = scaler.fit_transform(train_X_reshaped).reshape(train_X.shape)
    val_X = scaler.transform(val_X_reshaped).reshape(val_X.shape)
    test_X = scaler.transform(test_X_reshaped).reshape(test_X.shape)

    return train_X, val_X, test_X, train_y, val_y, test_y, num_features

def evaluate_model(MAX_RUL, model, test_X, test_y):
    pred_y = model.predict(test_X).flatten() * MAX_RUL
    true_y = test_y * MAX_RUL

    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    mae = mean_absolute_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R^2:  {r2:.3f}")

    def plot_line(y_true, y_pred):
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label='True RUL', color='blue', marker='o', markersize=3)
        plt.plot(y_pred, label='Predicted RUL', color='red', linestyle='--')
        plt.xlabel('Engine Unit Number')
        plt.ylabel('Remaining Cycles')
        plt.title('Actual vs Predicted RUL')
        plt.legend()
        plt.show()

    index_map = sorted(range(len(true_y)), key=lambda i: true_y[i])
    plot_line(true_y, pred_y)
    plot_line(
        list([true_y[index_map[i]] for i in range(len(true_y))]),
        list([pred_y[index_map[i]] for i in range(len(pred_y))])
    )

    plt.figure()
    plt.scatter(true_y, pred_y, alpha=0.5)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL")
    plt.plot([0, MAX_RUL], [0, MAX_RUL])
    plt.show()
