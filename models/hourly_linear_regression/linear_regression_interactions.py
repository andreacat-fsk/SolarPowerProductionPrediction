"""
Linear Regression with interaction terms for hourly solar power forecasting.

Extends the base linear model by adding hand-crafted interaction features
that capture non-linear relationships between sunshine duration and temporal/
spatial variables. This helps the linear model capture effects like:
- Sun duration × month (seasonal sun angle)
- Sun duration × hour (diurnal sun position)
- Sun duration × cloud cover (effective irradiance)
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

    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["day_of_year"] = df["datetime_hour"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-motivated interaction features."""
    df = df.copy()
    df["SUN_DUR_x_Month"] = df["SUN_DUR"] * df["Month"]
    df["SUN_DUR_x_hour_cos"] = df["SUN_DUR"] * df["hour_cos"]
    df["SUN_DUR_x_CLD_TTL_AMT_ID"] = df["SUN_DUR"] * df["CLD_TTL_AMT_ID"]
    df["SUN_DUR_x_Longitude"] = df["SUN_DUR"] * df["Longitude"]
    df["SUN_DUR_x_hour_cos_x_month_cos"] = (
        df["SUN_DUR"] * df["hour_cos"] * df["month_cos"]
    )
    return df


def train_linear_model_interactions(
    data_path: str, output_plot: str = "predicted_vs_true_interactions.png"
):
    """
    Train a linear regression model with interaction terms.

    Parameters
    ----------
    data_path : str
        Path to the feather file with preprocessed GSP-level data.
    output_plot : str
        Filename for the predicted-vs-true scatter plot.
    """
    df = pd.read_feather(data_path)
    df = prepare_cyclical_features(df)

    # Add dummy variables
    season_dummies = pd.get_dummies(df["Season"], prefix="Season", drop_first=True)
    near_sun_dummies = pd.get_dummies(
        df["near_sunrise_sunset"], prefix="NEAR_SUNRISE_SUNSET", drop_first=True
    )
    df = pd.concat([df, season_dummies, near_sun_dummies], axis=1)

    # Add interaction terms
    df = add_interaction_terms(df)

    # Temporal split
    train_df = df[df["Year"] < 2022]
    test_df = df[df["Year"] >= 2022]

    features = [
        "SUN_DUR", "AIR_TEMPERATURE", "PRCP_AMT",
        "CLD_TTL_AMT_ID", "MEAN_WIND_SPEED", "MSL_PRESSURE",
        "MEAN_WIND_DIR", "MAX_GUST_DIR", "MAX_GUST_SPEED", "HAS_RAINED",
        "Longitude", "Latitude", "Elevation",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "Season_Spring", "Season_Summer", "Season_Winter", "Year",
        "NEAR_SUNRISE_SUNSET_True",
        "SUN_DUR_x_Month", "SUN_DUR_x_hour_cos",
        "SUN_DUR_x_CLD_TTL_AMT_ID", "SUN_DUR_x_Longitude",
        "SUN_DUR_x_hour_cos_x_month_cos",
    ]
    target = "generation_mw"

    continuous_features = [
        col for col in features
        if not col.startswith("Season_") and not col.startswith("NEAR_SUNRISE_SUNSET_")
    ]
    dummy_columns = [
        "Season_Spring", "Season_Summer", "Season_Winter", "NEAR_SUNRISE_SUNSET_True"
    ]

    # Scaling
    scaler_X = StandardScaler()
    X_train_cont = scaler_X.fit_transform(train_df[continuous_features])
    X_test_cont = scaler_X.transform(test_df[continuous_features])

    X_train = np.hstack([X_train_cont, train_df[dummy_columns].values])
    X_test = np.hstack([X_test_cont, test_df[dummy_columns].values])

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(
        train_df[target].values.reshape(-1, 1)
    ).flatten()
    y_test = y_scaler.transform(test_df[target].values.reshape(-1, 1)).flatten()

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)

    model = nn.Linear(X_train_t.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

    all_features = continuous_features + dummy_columns
    coefficients = model.weight.data.cpu().numpy().flatten()
    intercept = model.bias.data.cpu().item()

    print("\n--- Learned Coefficients ---")
    for feat, coef in zip(all_features, coefficients):
        print(f"  {feat:40s}: {coef:+.6f}")
    print(f"  {'Intercept':40s}: {intercept:+.6f}")
    print(f"\n  R² score: {r2_score(y_test_np, y_pred_np):.4f}")

    # Plot
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
    plt.title("Linear Regression + Interactions: Predicted vs True")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"\nPlot saved to {output_plot}")


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "lin_regr_data.feather"
    train_linear_model_interactions(data_path)
