import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ===================================
# LOAD DATA
# ===================================

data = pd.read_csv("hour.csv")

# ===================================
# PREPROCESSING
# ===================================

# convert dteday to datetime
data['dteday'] = pd.to_datetime(data['dteday'])

# keep only day number
data['day'] = data['dteday'].dt.day

# remove old column
data.drop(columns=['dteday'], inplace=True)

# remove leakage columns
data.drop(columns=['instant', 'casual', 'registered'], inplace=True)

# ===================================
# FEATURES / TARGET
# ===================================

X = data.drop(columns=['cnt'])
# Log transformation to prevent negative predictions
y = np.log1p(data['cnt']) 

# ===================================
# STANDARDIZATION
# ===================================

scale_cols = ['temp', 'atemp', 'hum', 'windspeed']

scaler = StandardScaler()

X[scale_cols] = scaler.fit_transform(X[scale_cols])

# save scaler
joblib.dump(scaler, "scaler.pkl")

# ===================================
# TRAIN TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# BEST MODEL
# ===================================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=5,
    random_state=42
)

# ===================================
# TRAIN
# ===================================

model.fit(X_train, y_train)

# ===================================
# PREDICTION (Transform back)
# ===================================
y_pred_log = model.predict(X_test)
# Convert back from log space to actual bike counts
y_pred = np.expm1(y_pred_log)

# IMPORTANT: Convert y_test back to original scale for metric calculation
y_test_original = np.expm1(y_test)

# ===================================
# METRICS
# ===================================

# Compare original scale values to get real-world errors
r2 = r2_score(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)

print("\nFINAL MODEL PERFORMANCE")

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# ===================================
# SAVE MODEL
# ===================================

joblib.dump(model, "gradient_boosting_model.pkl")

print("\nModel saved successfully!")