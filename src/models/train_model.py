from src.data.clean_data import get_batters_df, get_pitchers_df
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

def train_models():
    batters_df = get_batters_df()
    pitchers_df = get_pitchers_df()

    # Normalize batters_df
    batters_numeric = batters_df.select_dtypes(include=['Int64', 'float'])
    batters_non_numeric = batters_df.select_dtypes(exclude=['Int64', 'float'])
    batters_normalized_numeric = (batters_numeric - batters_numeric.min()) / (batters_numeric.max() - batters_numeric.min())
    batters_normalized = pd.concat([batters_normalized_numeric, batters_non_numeric.reset_index(drop=True)], axis=1)

    # Normalize pitchers_df
    pitchers_numeric = pitchers_df.select_dtypes(include=['Int64', 'float'])
    pitchers_non_numeric = pitchers_df.select_dtypes(exclude=['Int64', 'float'])
    pitchers_normalized_numeric = (pitchers_numeric - pitchers_numeric.min()) / (pitchers_numeric.max() - pitchers_numeric.min())
    pitchers_normalized = pd.concat([pitchers_normalized_numeric, pitchers_non_numeric.reset_index(drop=True)], axis=1)

    # Drop the year column
    batters_normalized_numeric = batters_normalized_numeric.drop(columns=['year'])
    pitchers_normalized_numeric = pitchers_normalized_numeric.drop(columns=['year'])
    pitchers_normalized_numeric = pitchers_normalized_numeric.dropna()

    # Define features and target variable
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
    predictions = batters_model.predict(X_test_batters)

    # Define features and target variable
    # We'll predict 'b_war' using all other numeric columns except 'b_war' itself
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
    predictions = pitchers_model.predict(X_test_pitchers)

    # Save the batters model
    joblib.dump(batters_model, "data/models/batters_model.joblib")

    # Save the pitchers model
    joblib.dump(pitchers_model, "data/models/pitchers_model.joblib")

if __name__ == "__main__":
    train_models()