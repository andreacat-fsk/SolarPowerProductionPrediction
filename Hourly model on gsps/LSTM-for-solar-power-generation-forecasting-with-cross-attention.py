import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Reshape
from keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam 

from matplotlib import pyplot as plt

data = pd.read_feather("/home/andreacatucci/data_for_lstm.feather")

# Our sample gsps for comparison are: 47, 152, 199, 202, 324.
# Let's extract them.

#### Testing only: use gsp 47 for fitting ####
data_gsp_47 = data[data['gsp_id'].isin([47])].drop(columns=['gsp_id', 'Longitude', 'Latitude', 'Elevation', 'Season', 'is_daylight'])

# First step: prepare input data for training
## LSTM expects 3D data (samples, time steps, features)

# Split data into train/test sets

train_47 = data_gsp_47[data_gsp_47['Year'].isin([2015, 2016, 2017,2018,2019,2020,2021])]
test_47 = data_gsp_47[data_gsp_47['Year'] == 2022]

#Separate dates for future plotting
train_dates = pd.to_datetime(train_47['datetime_hour'])

variables = train_47.columns.difference(['datetime_hour'])  # returns a valid Index object
print(variables)
df_for_training = train_47[variables].astype(float)

# Separate target and features
target_col = "generation_mw"
feature_cols = [c for c in df_for_training.columns if c != target_col]

X = df_for_training[feature_cols].astype(float)
#y = df_for_training[[target_col]].astype(float)  # keep as DataFrame for scaler

#### SCALE GEN BY CAPACITY

# Separate target and features
capacity_col = "total_capacity_mwp"
feature_cols = [c for c in df_for_training.columns if c not in [target_col]]

# Compute capacity factor as target
y_capacity_factor = df_for_training[[target_col]].values.flatten() / df_for_training[[capacity_col]].values.flatten()
y_capacity_factor = y_capacity_factor.reshape(-1, 1)  

# Fit separate scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Recombine into full scaled matrix if you want to keep same structure
df_for_training_scaled = np.concatenate([X_scaled, y_scaled], axis=1)

# Make sure columns are in the same order as before
scaled_cols = feature_cols + [target_col]

# Reshape data into 3D

n_future = 1
n_past = 18

weather_cols = [
    'AIR_TEMPERATURE', 'CLD_TTL_AMT_ID', 'HAS_RAINED', 'MAX_GUST_DIR',
    'MAX_GUST_SPEED', 'MEAN_WIND_DIR', 'MEAN_WIND_SPEED', 'MSL_PRESSURE',
    'PRCP_AMT', 'SUN_DUR', 'solar_azimuth', 'solar_elevation',
    'time_after_sunrise', 'time_to_sunset'
]

# Find indices of weather columns in df_for_training
weather_idx = [df_for_training.columns.get_loc(c) for c in weather_cols]

trainX_seq, trainX_weather, trainY = [], [], []

for i in range(n_past, len(X_scaled) - n_future + 1):
    # Sequence: past n_past timesteps of ALL features (including weather)
    seq_x = X_scaled[i - n_past:i, :]  # shape (n_past, n_features)
    
    # Weather: only the forecasted weather vars at time t (the "current" step)
    weather_x = X_scaled[i, weather_idx]  # shape (len(weather_idx),)
    
    # Target: generation at time t (already scaled separately)
    seq_y = y_scaled[i + n_future - 1:i + n_future, :]  # shape (1, 1)
    
    trainX_seq.append(seq_x)
    trainX_weather.append(weather_x)
    trainY.append(seq_y)

# Convert to numpy arrays
trainX_seq = np.array(trainX_seq)            # (samples, n_past, n_features)
trainX_weather = np.array(trainX_weather)    # (samples, n_weather_features)
trainY = np.array(trainY).reshape(-1, 1)     # (samples, 1)

print('trainX_seq shape: ', trainX_seq.shape)
print('trainX_weather shape: ', trainX_weather.shape)
print('trainY shape: ', trainY.shape)


#trainX_seq shape:  (46008, 18, 66)
#trainX_weather shape:  (46008, 14)
#trainY shape:  (46008, 1)

# LSTM encoder + fusion dense layer

n_features = trainX_seq.shape[2]  

# --- LSTM Encoder ---
seq_inp = Input(shape=(n_past, n_features))
x_1 = LSTM(64, return_sequences=True, dropout=0.2)(seq_inp)
x_2 = LSTM(32, return_sequences=True, dropout=0.2)(x_1)  # keep sequence for attention

# --- Weather branch ---
weather_dim = len(weather_cols)
weather_inp = Input(shape=(weather_dim,))
w = Dense(64, activation="relu")(weather_inp)
w = Dense(32, activation="relu")(w)
w = Dropout(0.3)(w)

# reshape weather vector to sequence length = 1 for attention
w_seq = Reshape((1, 32))(w)

# --- Cross-Attention ---
# Let the LSTM sequence attend to weather embedding
attn_out = MultiHeadAttention(num_heads=2, key_dim=32)(
    query=x_2,        # sequence as query
    value=w_seq,      # weather as value
    key=w_seq         # weather as key
)
attn_vec = attn_out[:, -1, :]  #last time step


# --- Fusion ---
fusion_2 = Dense(64, activation='relu')(attn_vec)
fusion_3 = Dense(32, activation='relu')(fusion_2)
fusion_4 = Dropout(0.2)(fusion_3)
out = Dense(trainY.shape[1])(fusion_4)

model = Model(inputs= [seq_inp, weather_inp], outputs=out)
custom_optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=custom_optimizer, loss='mse')
model.summary()

# Split training data into training/validation (10%) sets

n_samples = len(trainX_seq)
val_size = int(0.1 * n_samples)

X_seq_train, X_seq_val = trainX_seq[:-val_size], trainX_seq[-val_size:]
X_weather_train, X_weather_val = trainX_weather[:-val_size], trainX_weather[-val_size:]
y_train, y_val = trainY[:-val_size], trainY[-val_size:]

# Train model

save_dir = "./models"
os.makedirs(save_dir, exist_ok = True)
model_path = os.path.join(save_dir, "lstm_weather_fusion.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights = True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(
        monitor='val_loss',       
        factor=0.5,               
        patience=5,              
        min_lr=1e-6,              
        verbose=1
    )
]

history = model.fit(
    [X_seq_train, X_weather_train],
    y_train,
    validation_data = ([X_seq_val, X_weather_val], y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# Test loop

test_dates = pd.to_datetime(test_47['datetime_hour'])
df_for_test = test_47[variables].astype(float)

X_test = df_for_test[feature_cols].astype(float)
y_test = df_for_test[[target_col]].astype(float)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

testX_seq, testX_weather, testY = [], [], []

for i in range(n_past, len(X_test_scaled) - n_future + 1):
    seq_x = X_test_scaled[i - n_past:i, :]
    weather_x = X_test_scaled[i, weather_idx]
    seq_y = y_test_scaled[i + n_future - 1:i + n_future, :]
    
    testX_seq.append(seq_x)
    testX_weather.append(weather_x)
    testY.append(seq_y)

testX_seq = np.array(testX_seq)
testX_weather = np.array(testX_weather)
testY = np.array(testY).reshape(-1, 1)

print("testX_seq shape:", testX_seq.shape)
print("testX_weather shape:", testX_weather.shape)
print("testY shape:", testY.shape)

# Evaluate

model.load_weights(model_path)

test_loss = model.evaluate([testX_seq, testX_weather], testY)

#y_pred_scaled = model.predict([testX_seq, testX_weather])
#y_pred = scaler_y.inverse_transform(y_pred_scaled) 
#y_true = scaler_y.inverse_transform(testY)


# Inference scaling back
y_pred_scaled = model.predict([testX_seq, testX_weather])
y_pred_capacity_factor = scaler_y.inverse_transform(y_pred_scaled)

# Multiply by capacity at each timestep
capacity_test = X_test[capacity_col].values[n_past:]  # align with testY length
y_pred = y_pred_capacity_factor.flatten() * capacity_test
y_true = y_test.values.flatten()[n_past:]  # align true with predictions

#y_true = scaler_y.inverse_transform(testY)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Test MAE: {mae:.2f} MW")
print(f"Test RMSE: {rmse:.2f} MW")

pred_df = pd.DataFrame({
    "y_true": y_true.flatten(),
    "y_pred": y_pred.flatten()
})
pred_df.to_csv("predictions.csv", index=False)

metrics = {"MAE": mae, "RMSE": rmse}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("metrics.csv", index=False)

