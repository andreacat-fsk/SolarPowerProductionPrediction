"""
LSTM with Cross-Attention weather fusion — experimental variant.

This architecture replaces the concatenation-based fusion with a
Multi-Head Attention mechanism where the LSTM sequence output attends
to the weather embedding. This allows the model to dynamically weight
which parts of the historical sequence are most relevant given current
weather conditions.

Architecture:
    - Two stacked LSTM layers (64 → 32, return_sequences=True)
    - Weather branch (64 → 32) reshaped to sequence for attention
    - Multi-Head Attention (2 heads, key_dim=32): LSTM queries weather
    - Fusion dense layers (64 → 32 → 1)

This is an experimental variant that was tested but did not outperform
the simpler concatenation approach on this dataset.
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, Reshape, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


WEATHER_COLS = [
    "AIR_TEMPERATURE", "CLD_TTL_AMT_ID", "HAS_RAINED", "MAX_GUST_DIR",
    "MAX_GUST_SPEED", "MEAN_WIND_DIR", "MEAN_WIND_SPEED", "MSL_PRESSURE",
    "PRCP_AMT", "SUN_DUR", "solar_azimuth", "solar_elevation",
    "time_after_sunrise", "time_to_sunset",
]

N_PAST = 18
N_FUTURE = 1
DROP_COLS = ["gsp_id", "Longitude", "Latitude", "Elevation", "Season", "is_daylight"]


def train_lstm_cross_attention(data_path: str, gsp_id: int, output_dir: str = "./output"):
    """Train LSTM with cross-attention weather fusion for one GSP."""
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

    weather_idx = [df_train.columns.get_loc(c) for c in WEATHER_COLS if c in df_train.columns]

    # Create sequences
    trainX_seq, trainX_weather, trainY = [], [], []
    for i in range(N_PAST, len(X_scaled) - N_FUTURE + 1):
        trainX_seq.append(X_scaled[i - N_PAST:i, :])
        trainX_weather.append(X_scaled[i, weather_idx])
        trainY.append(y_scaled[i + N_FUTURE - 1:i + N_FUTURE, :])

    trainX_seq = np.array(trainX_seq)
    trainX_weather = np.array(trainX_weather)
    trainY = np.array(trainY).reshape(-1, 1)

    n_features = trainX_seq.shape[2]
    weather_dim = len(weather_idx)

    # Build model with cross-attention
    seq_inp = Input(shape=(N_PAST, n_features))
    x = LSTM(64, return_sequences=True, dropout=0.2)(seq_inp)
    x = LSTM(32, return_sequences=True, dropout=0.2)(x)

    weather_inp = Input(shape=(weather_dim,))
    w = Dense(64, activation="relu")(weather_inp)
    w = Dense(32, activation="relu")(w)
    w = Dropout(0.3)(w)
    w_seq = Reshape((1, 32))(w)

    # Cross-attention: LSTM sequence queries attend to weather embedding
    attn_out = MultiHeadAttention(num_heads=2, key_dim=32)(
        query=x, value=w_seq, key=w_seq
    )
    attn_vec = attn_out[:, -1, :]  # Take last timestep

    # Fusion
    fusion = Dense(64, activation="relu")(attn_vec)
    fusion = Dense(32, activation="relu")(fusion)
    fusion = Dropout(0.2)(fusion)
    out = Dense(1)(fusion)

    model = Model(inputs=[seq_inp, weather_inp], outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary()

    # Train
    val_size = int(0.1 * len(trainX_seq))
    model_path = os.path.join(output_dir, f"lstm_cross_attn_gsp_{gsp_id}.h5")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        [trainX_seq[:-val_size], trainX_weather[:-val_size]],
        trainY[:-val_size],
        validation_data=(
            [trainX_seq[-val_size:], trainX_weather[-val_size:]],
            trainY[-val_size:],
        ),
        epochs=100, batch_size=128, callbacks=callbacks, verbose=1,
    )

    # Evaluate
    df_test = test[variables].astype(float)
    X_test_scaled = scaler_X.transform(df_test[feature_cols].astype(float))
    y_test_scaled = scaler_y.transform(df_test[[target_col]].astype(float))

    testX_seq, testX_weather, testY = [], [], []
    for i in range(N_PAST, len(X_test_scaled) - N_FUTURE + 1):
        testX_seq.append(X_test_scaled[i - N_PAST:i, :])
        testX_weather.append(X_test_scaled[i, weather_idx])
        testY.append(y_test_scaled[i + N_FUTURE - 1:i + N_FUTURE, :])

    testX_seq = np.array(testX_seq)
    testX_weather = np.array(testX_weather)
    testY = np.array(testY).reshape(-1, 1)

    model.load_weights(model_path)
    y_pred = scaler_y.inverse_transform(model.predict([testX_seq, testX_weather]))
    y_true = scaler_y.inverse_transform(testY)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- Results (Cross-Attention) GSP {gsp_id} ---")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")

    pd.DataFrame([{"gsp_id": gsp_id, "MAE": mae, "RMSE": rmse}]).to_csv(
        os.path.join(output_dir, f"metrics_cross_attn_gsp_{gsp_id}.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM with cross-attention (experimental)")
    parser.add_argument("data_path", help="Path to the feather data file")
    parser.add_argument("--gsp-id", type=int, default=47)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    train_lstm_cross_attention(args.data_path, args.gsp_id, args.output_dir)
