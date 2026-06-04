"""
Linear Regression baseline for hourly solar power forecasting (PyTorch).

This model predicts next-hour solar generation for individual GSPs using
weather covariates, cyclical time encodings, and geographic features. It
serves as an interpretable baseline before moving to LSTM architectures.

Features:
- Cyclical encoding of hour, day-of-year, and month
- One-hot encoding for season and near-sunrise/sunset indicator
- StandardScaler normalisation on features and target
- GPU-accelerated training via PyTorch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def prepare_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine cyclical encodings for time variables."""
    df = df.copy()
    df["Hour"] = df["Hour"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Year"] = df["Year"].astype(int)

    # Hour encoding (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Day-of-year encoding (period = 365)
    df["day_of_year"] = df["datetime_hour"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Month encoding (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df


def prepare_dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded features for categorical variables."""
    df = df.copy()
    season_dummies = pd.get_dummies(df["Season"], prefix="Season", drop_first=True)
    near_sun_dummies = pd.get_dummies(
        df["near_sunrise_sunset"], prefix="NEAR_SUNRISE_SUNSET", drop_first=True
    )
    df = pd.concat([df, season_dummies, near_sun_dummies], axis=1)
    return df


def train_linear_model(data_path: str, output_plot: str = "predicted_vs_true.png"):
    """
    Train a linear regression model and evaluate on held-out test year.

    Parameters
    ----------
    data_path : str
        Path to the feather file with preprocessed GSP-level data.
    output_plot : str
        Filename for the predicted-vs-true scatter plot.
    """
    # Load data
    df = pd.read_feather(data_path)
    df = prepare_cyclical_features(df)
    df = prepare_dummy_features(df)

    # Train/test split (temporal: train < 2022, test >= 2022)
    train_df = df[df["Year"] < 2022]
    test_df = df[df["Year"] >= 2022]

    # Feature definitions
    features = [
        "SUN_DUR", "AIR_TEMPERATURE", "PRCP_AMT",
        "CLD_TTL_AMT_ID", "MEAN_WIND_SPEED", "MSL_PRESSURE",
        "MEAN_WIND_DIR", "MAX_GUST_DIR", "MAX_GUST_SPEED", "HAS_RAINED",
        "Longitude", "Latitude", "Elevation",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "Season_Spring", "Season_Summer", "Season_Winter", "Year",
        "NEAR_SUNRISE_SUNSET_True",
    ]
    target = "generation_mw"

    continuous_features = [
        col for col in features
        if not col.startswith("Season_") and not col.startswith("NEAR_SUNRISE_SUNSET_")
    ]
    dummy_columns = [
        "Season_Spring", "Season_Summer", "Season_Winter", "NEAR_SUNRISE_SUNSET_True"
    ]

    # Scale continuous features
    scaler_X = StandardScaler()
    X_train_cont = scaler_X.fit_transform(train_df[continuous_features])
    X_test_cont = scaler_X.transform(test_df[continuous_features])

    X_train = np.hstack([X_train_cont, train_df[dummy_columns].values])
    X_test = np.hstack([X_test_cont, test_df[dummy_columns].values])

    # Scale target
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(
        train_df[target].values.reshape(-1, 1)
    ).flatten()
    y_test = y_scaler.transform(test_df[target].values.reshape(-1, 1)).flatten()

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)

    # Model: single linear layer
    model = nn.Linear(X_train_t.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_test_t), y_test_t)
            print(
                f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | "
                f"Test Loss: {val_loss.item():.6f}"
            )

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t)

    y_pred_np = y_pred.cpu().numpy().flatten()
    y_test_np = y_test_t.cpu().numpy().flatten()

    # Print coefficients
    all_features = continuous_features + dummy_columns
    coefficients = model.weight.data.cpu().numpy().flatten()
    intercept = model.bias.data.cpu().item()

    print("\n--- Learned Coefficients ---")
    for feat, coef in zip(all_features, coefficients):
        print(f"  {feat:30s}: {coef:+.6f}")
    print(f"  {'Intercept':30s}: {intercept:+.6f}")
    print(f"\n  R² score: {r2_score(y_test_np, y_pred_np):.4f}")

    # Plot in original units
    preds_orig = y_scaler.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
    targets_orig = y_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(targets_orig, preds_orig, alpha=0.3, edgecolors="k", s=10)
    plt.plot(
        [targets_orig.min(), targets_orig.max()],
        [targets_orig.min(), targets_orig.max()],
        "r--", lw=2,
    )
    plt.xlabel("True Generation (MW)")
    plt.ylabel("Predicted Generation (MW)")
    plt.title("Linear Regression: Predicted vs True Solar Power Output")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"\nPlot saved to {output_plot}")


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "lin_regr_data.feather"
    train_linear_model(data_path)
