"""
LSTM without weather fusion — ablation study.

This model uses only lagged sequences (54 hours lookback) without
contemporaneous weather inputs. It demonstrates the importance of
the weather fusion branch by comparing performance.

Architecture:
    - Two stacked LSTM layers (128 → 64 units)
    - Single dense layer (48 → 1)
    - No weather branch, no skip connection

This serves as a lower-bound ablation to quantify the contribution
of current weather covariates to forecast accuracy.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


N_PAST = 54       # 3 × 18 hours lookback (longer window to compensate for no weather)
N_FUTURE = 1
BATCH_SIZE = 128
MAX_EPOCHS = 100

DROP_COLS = ["gsp_id", "Longitude", "Latitude", "Elevation", "Season", "is_daylight"]


def train_lstm_no_weather(data_path: str, gsp_id: int, output_dir: str = "./output"):
    """Train LSTM without weather inputs for one GSP."""
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_feather(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in data.columns]
    data_gsp = data[data["gsp_id"] == gsp_id].drop(columns=cols_to_drop)

    train = data_gsp[data_gsp["Year"].isin(range(2015, 2022))]
    test = data_gsp[data_gsp["Year"] == 2022]

    variables = train.columns.difference(["datetime_hour"])
    df_train = train[variables].astype(float)

    target_col = "generation_mw"
    feature_cols = [c for c in df_train.columns if c != target_col]

    X = df_train[feature_cols].astype(float)
    y = df_train[[target_col]].astype(float)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Create sequences (no weather branch)
    trainX_seq, trainY = [], []
    for i in range(N_PAST, len(X_scaled) - N_FUTURE + 1):
        trainX_seq.append(X_scaled[i - N_PAST:i, :])
        trainY.append(y_scaled[i + N_FUTURE - 1:i + N_FUTURE, :])

    trainX_seq = np.array(trainX_seq)
    trainY = np.array(trainY).reshape(-1, 1)

    # Build model
    n_features = trainX_seq.shape[2]
    seq_inp = Input(shape=(N_PAST, n_features))
    x = LSTM(128, return_sequences=True, dropout=0.2)(seq_inp)
    x = LSTM(64, dropout=0.2)(x)
    x = Dense(48, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(seq_inp, out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary()

    # Split and train
    val_size = int(0.1 * len(trainX_seq))
    X_train, X_val = trainX_seq[:-val_size], trainX_seq[-val_size:]
    y_train, y_val = trainY[:-val_size], trainY[-val_size:]

    model_path = os.path.join(output_dir, f"lstm_no_weather_gsp_{gsp_id}.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1,
    )

    # Test
    df_test = test[variables].astype(float)
    X_test = scaler_X.transform(df_test[feature_cols].astype(float))
    y_test_scaled = scaler_y.transform(df_test[[target_col]].astype(float))

    testX_seq, testY = [], []
    for i in range(N_PAST, len(X_test) - N_FUTURE + 1):
        testX_seq.append(X_test[i - N_PAST:i, :])
        testY.append(y_test_scaled[i + N_FUTURE - 1:i + N_FUTURE, :])

    testX_seq = np.array(testX_seq)
    testY = np.array(testY).reshape(-1, 1)

    model.load_weights(model_path)
    y_pred = scaler_y.inverse_transform(model.predict(testX_seq))
    y_true = scaler_y.inverse_transform(testY)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- Results (No Weather) GSP {gsp_id} ---")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")

    pd.DataFrame([{"gsp_id": gsp_id, "MAE": mae, "RMSE": rmse}]).to_csv(
        os.path.join(output_dir, f"metrics_no_weather_gsp_{gsp_id}.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM without weather (ablation)")
    parser.add_argument("data_path", help="Path to the feather data file")
    parser.add_argument("--gsp-id", type=int, default=47)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    train_lstm_no_weather(args.data_path, args.gsp_id, args.output_dir)
