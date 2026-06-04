"""
LSTM with Weather Fusion for hourly solar power generation forecasting.

Architecture:
    - Two stacked LSTM layers (64 → 32 units) encode 18 hours of past sequences
    - A dense branch (64 → 32 units) transforms current weather covariates
    - A skip connection preserves raw weather inputs
    - Fusion dense layers (64 → 32 → 1) combine both branches for prediction

This is the primary model described in the thesis. It is trained independently
per GSP to capture local dynamics while sharing architecture across sites.

Key design choices:
    - 18-hour lookback (daytime hours only, nighttime removed)
    - AdamW optimizer with ReduceLROnPlateau and early stopping
    - Dropout (0.2) across all layers to prevent overfitting
    - MSE loss (penalises large errors at peak generation)
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import AdamW


# --- Configuration ---

WEATHER_COLS = [
    "AIR_TEMPERATURE", "CLD_TTL_AMT_ID", "HAS_RAINED", "MAX_GUST_DIR",
    "MAX_GUST_SPEED", "MEAN_WIND_DIR", "MEAN_WIND_SPEED", "MSL_PRESSURE",
    "PRCP_AMT", "SUN_DUR", "solar_azimuth", "solar_elevation",
    "time_after_sunrise", "time_to_sunset",
]

N_PAST = 18       # Lookback window (hours)
N_FUTURE = 1      # Forecast horizon
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
PATIENCE_ES = 10  # Early stopping patience
PATIENCE_LR = 5   # ReduceLROnPlateau patience

# Columns to drop (static per GSP or unused)
DROP_COLS = [
    "gsp_id", "Longitude", "Latitude", "Elevation", "Season", "is_daylight",
]


def load_and_prepare_data(data_path: str, gsp_id: int) -> tuple:
    """Load data and split into train/test for a specific GSP."""
    data = pd.read_feather(data_path)

    # Filter to target GSP and drop static columns (keep only those that exist)
    cols_to_drop = [c for c in DROP_COLS if c in data.columns]
    data_gsp = data[data["gsp_id"] == gsp_id].drop(columns=cols_to_drop)

    # Also drop capacity breakdown columns if present
    capacity_cols = [c for c in data_gsp.columns if "cumul_capacity" in c]
    if capacity_cols:
        data_gsp = data_gsp.drop(columns=capacity_cols)

    # Temporal split
    train = data_gsp[data_gsp["Year"].isin(range(2015, 2022))]
    test = data_gsp[data_gsp["Year"] == 2022]

    return train, test


def create_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    weather_idx: list,
) -> tuple:
    """Create input sequences for the LSTM model."""
    X_seq, X_weather, Y = [], [], []

    for i in range(N_PAST, len(X_scaled) - N_FUTURE + 1):
        seq_x = X_scaled[i - N_PAST:i, :]
        weather_x = X_scaled[i, weather_idx]
        seq_y = y_scaled[i + N_FUTURE - 1:i + N_FUTURE, :]

        X_seq.append(seq_x)
        X_weather.append(weather_x)
        Y.append(seq_y)

    return (
        np.array(X_seq),
        np.array(X_weather),
        np.array(Y).reshape(-1, 1),
    )


def build_model(n_features: int, n_weather: int) -> Model:
    """Build the LSTM + weather fusion model."""
    # LSTM branch (sequence encoder)
    seq_inp = Input(shape=(N_PAST, n_features), name="sequence_input")
    x = LSTM(64, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=0)(seq_inp)
    x = LSTM(32, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=0)(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Weather branch (current timestep)
    weather_inp = Input(shape=(n_weather,), name="weather_input")
    w = Dense(64, activation="relu")(weather_inp)
    w = Dense(32, activation="relu")(w)
    w = Dropout(DROPOUT_RATE)(w)

    # Fusion with skip connection (raw weather preserved)
    fusion = Concatenate()([x, w, weather_inp])
    fusion = Dense(64, activation="relu")(fusion)
    fusion = Dense(32, activation="relu")(fusion)
    fusion = Dropout(DROPOUT_RATE)(fusion)
    out = Dense(1, activation="linear")(fusion)

    model = Model(inputs=[seq_inp, weather_inp], outputs=out)
    model.compile(optimizer=AdamW(learning_rate=LEARNING_RATE), loss="mse")

    return model


def train_and_evaluate(data_path: str, gsp_id: int, output_dir: str = "./output"):
    """Full training and evaluation pipeline for one GSP."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training LSTM model for GSP {gsp_id}")
    print(f"{'='*60}\n")

    # Load data
    train_df, test_df = load_and_prepare_data(data_path, gsp_id)
    train_dates = pd.to_datetime(train_df["datetime_hour"])
    test_dates = pd.to_datetime(test_df["datetime_hour"])

    # Prepare features
    variables = train_df.columns.difference(["datetime_hour"])
    df_train = train_df[variables].astype(float)

    target_col = "generation_mw"
    feature_cols = [c for c in df_train.columns if c != target_col]

    X_train_raw = df_train[feature_cols].astype(float)
    y_train_raw = df_train[[target_col]].astype(float)

    # Fit scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train_raw)
    y_scaled = scaler_y.fit_transform(y_train_raw)

    # Weather column indices
    weather_idx = [df_train.columns.get_loc(c) for c in WEATHER_COLS if c in df_train.columns]

    # Create sequences
    trainX_seq, trainX_weather, trainY = create_sequences(X_scaled, y_scaled, weather_idx)
    print(f"Training sequences: {trainX_seq.shape[0]}")
    print(f"Sequence shape: {trainX_seq.shape}")
    print(f"Weather shape: {trainX_weather.shape}")

    # Train/validation split (last 10%, preserving temporal order)
    val_size = int(0.1 * len(trainX_seq))
    X_seq_train, X_seq_val = trainX_seq[:-val_size], trainX_seq[-val_size:]
    X_weather_train, X_weather_val = trainX_weather[:-val_size], trainX_weather[-val_size:]
    y_train, y_val = trainY[:-val_size], trainY[-val_size:]

    # Build and train model
    model = build_model(
        n_features=trainX_seq.shape[2],
        n_weather=trainX_weather.shape[1],
    )
    model.summary()

    model_path = os.path.join(output_dir, f"lstm_gsp_{gsp_id}.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=PATIENCE_LR, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        [X_seq_train, X_weather_train],
        y_train,
        validation_data=([X_seq_val, X_weather_val], y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    df_test = test_df[variables].astype(float)
    X_test_raw = df_test[feature_cols].astype(float)
    y_test_raw = df_test[[target_col]].astype(float)

    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    testX_seq, testX_weather, testY = create_sequences(X_test_scaled, y_test_scaled, weather_idx)

    model.load_weights(model_path)
    y_pred_scaled = model.predict([testX_seq, testX_weather])
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(testY)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- Results for GSP {gsp_id} ---")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")

    # Save predictions and metrics
    pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()}).to_csv(
        os.path.join(output_dir, f"predictions_gsp_{gsp_id}.csv"), index=False
    )
    pd.DataFrame([{"gsp_id": gsp_id, "MAE": mae, "RMSE": rmse}]).to_csv(
        os.path.join(output_dir, f"metrics_gsp_{gsp_id}.csv"), index=False
    )

    # --- Diagnostic plots ---
    test_dates_aligned = test_dates.iloc[N_PAST:N_PAST + len(y_true)]
    if hasattr(test_dates_aligned, "dt"):
        test_dates_aligned = test_dates_aligned.dt.tz_localize(None)

    # Full test set
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(test_dates_aligned, y_true.flatten(), label="True", alpha=0.7)
    ax.plot(test_dates_aligned, y_pred.flatten(), label="Predicted", alpha=0.7)
    ax.set_title(f"GSP {gsp_id} — Full Test Set Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Generation (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"full_test_gsp_{gsp_id}.png"), dpi=200)
    plt.close()

    # Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title(f"GSP {gsp_id} — Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_curve_gsp_{gsp_id}.png"), dpi=200)
    plt.close()

    return {"gsp_id": gsp_id, "MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM solar forecasting model for a specific GSP"
    )
    parser.add_argument("data_path", help="Path to the feather data file")
    parser.add_argument(
        "--gsp-id", type=int, default=199,
        help="GSP ID to train on (default: 199)"
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Directory for outputs (models, plots, metrics)"
    )
    args = parser.parse_args()

    train_and_evaluate(args.data_path, args.gsp_id, args.output_dir)
