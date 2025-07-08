"""
Train regression models for batters and pitchers using normalized baseball stats.
- Loads normalized data from clean_data
- Trains linear regression models for WAR prediction
- Saves trained models to disk
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.clean_data import get_batters_df_normalized, get_pitchers_df_normalized

# ----------------------
# Model Training
# ----------------------
def train_models():
    """
    Trains linear regression models for batters and pitchers WAR prediction.
    Saves models to data/models/ directory.
    """
    # Load normalized numeric data
    batters_normalized_numeric = get_batters_df_normalized()
    pitchers_normalized_numeric = get_pitchers_df_normalized()

    # -----------
    # Batters Model
    # -----------
    # We'll predict 'b_war' using all other numeric columns except 'b_war' itself
    X_batters = batters_normalized_numeric.drop(columns=['b_war'])
    y_batters = batters_normalized_numeric['b_war']

    # Split into train and test sets
    X_train_batters, X_test_batters, y_train_batters, y_test_batters = train_test_split(
        X_batters, y_batters, test_size=0.2, random_state=42
    )

    # Train the linear regression model
    batters_model = LinearRegression()
    batters_model.fit(X_train_batters, y_train_batters)

    # Make predictions on the test set
    predictions_batters = batters_model.predict(X_test_batters)

    # -----------
    # Pitchers Model
    # -----------
    # We'll predict 'p_war' using all other numeric columns except 'p_war' itself
    X_pitchers = pitchers_normalized_numeric.drop(columns=['p_war'])
    y_pitchers = pitchers_normalized_numeric['p_war']

    # Split into train and test sets
    X_train_pitchers, X_test_pitchers, y_train_pitchers, y_test_pitchers = train_test_split(
        X_pitchers, y_pitchers, test_size=0.2, random_state=42
    )

    # Train the linear regression model
    pitchers_model = LinearRegression()
    pitchers_model.fit(X_train_pitchers, y_train_pitchers)

    # Make predictions on the test set
    predictions_pitchers = pitchers_model.predict(X_test_pitchers)

    # -----------
    # Save Models
    # -----------
    joblib.dump(batters_model, "data/models/batters_model.joblib")
    joblib.dump(pitchers_model, "data/models/pitchers_model.joblib")

if __name__ == "__main__":
    train_models()