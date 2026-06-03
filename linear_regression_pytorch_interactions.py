import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

df = pd.read_feather('/home/andreacatucci/lin_regr_data.feather')

df['Hour'] = df['Hour'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)

#Prepare cyclical variable for hour

df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

#Prepare cyclical variable for day

df['day_of_year'] = df['datetime_hour'].dt.dayofyear
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

#Prepare cyclical variable for month

df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

#Prepare one hot encoding for quarter

season_dummies = pd.get_dummies(df['Season'], prefix='Season', drop_first=True)
near_sun_dummies = pd.get_dummies(df['near_sunrise_sunset'], prefix='NEAR_SUNRISE_SUNSET', drop_first=True)
df = pd.concat([df, season_dummies, near_sun_dummies], axis=1)

#Add interaction terms manually

df['SUN_DUR_x_Month'] = df['SUN_DUR'] * df['Month']
df['SUN_DUR_x_hour_cos'] = df['SUN_DUR'] * df['hour_cos']
df['SUN_DUR_x_CLD_TTL_AMT_ID'] = df['SUN_DUR'] * df['CLD_TTL_AMT_ID']
df['SUN_DUR_x_Longitude'] = df['SUN_DUR'] * df['Longitude']
df['SUN_DUR_x_hour_cos_x_month_cos'] = df['SUN_DUR'] * df['hour_cos'] * df['month_cos']

train_df = df[df['Year'] < 2022]
test_df  = df[df['Year'] >= 2022]

df.drop(columns=['Month', 'Hour', 'datetime_hour', 'day_of_year', 'Season'], inplace=True)

features = ['SUN_DUR', 'AIR_TEMPERATURE', 'PRCP_AMT',
            'CLD_TTL_AMT_ID', 'MEAN_WIND_SPEED', 'MSL_PRESSURE',
            'MEAN_WIND_DIR', 'MAX_GUST_DIR', 'MAX_GUST_SPEED', 'HAS_RAINED',
            'Longitude', 'Latitude', 'Elevation', 
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'doy_sin', 'doy_cos',
            'Season_Spring', 'Season_Summer', 'Season_Winter', 'Year',
            'NEAR_SUNRISE_SUNSET_True',
            'SUN_DUR_x_Month', 'SUN_DUR_x_hour_cos',
            'SUN_DUR_x_CLD_TTL_AMT_ID', 'SUN_DUR_x_Longitude', 'SUN_DUR_x_hour_cos_x_month_cos']

target = 'generation_mw'

#Scale continuous variables

continuous_features = [
    col for col in features 
    if not col.startswith('Season_') and not col.startswith('NEAR_SUNRISE_SUNSET_')
]

dummy_columns = ['Season_Spring', 'Season_Summer', 'Season_Winter', 'NEAR_SUNRISE_SUNSET_True']

scaler_X = StandardScaler()
X_train_cont = scaler_X.fit_transform(train_df[continuous_features])
X_test_cont  = scaler_X.transform(test_df[continuous_features])

X_train = np.hstack([X_train_cont, train_df[dummy_columns].values])
X_test  = np.hstack([X_test_cont,  test_df[dummy_columns].values])

y_train_orig = train_df[target]
y_test_orig = test_df[target]

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_orig.values.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test_orig.values.reshape(-1, 1)).flatten()

#Training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test, dtype=torch.float32).to(device)
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

model = nn.Linear(X_train.shape[1], 1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(1, num_epochs + 1):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Test Loss: {val_loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test)
    print(f"Test MSE: {mse.item():.6f}")

y_pred_np = y_pred.cpu().numpy().flatten()
y_test_np = y_test.cpu().numpy().flatten()

coefficients = model.weight.data.cpu().numpy().flatten()
intercept = model.bias.data.cpu().item()

all_features = continuous_features + dummy_columns
for feat, coef in zip(all_features, coefficients):
    print(f"{feat}: {coef:.6f}")
print(f"Intercept: {intercept:.6f}")

print("R²:", r2_score(y_test_np, y_pred_np))

preds_orig = y_scaler.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
targets_orig = y_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 6))
plt.scatter(targets_orig, preds_orig, alpha=0.3, edgecolors='k')
plt.plot([targets_orig.min(), targets_orig.max()], [targets_orig.min(), targets_orig.max()], 'r--', lw=2)
plt.xlabel('True MWh')
plt.ylabel('Predicted MWh')
plt.title('Predicted vs. True Solar Power Output (Original Units)')
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_true_interactions.png", dpi=300)
