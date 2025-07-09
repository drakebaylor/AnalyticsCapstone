import sys
import os
import pandas as pd
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.clean_data import get_batters_df, get_pitchers_df, get_batters_df_normalized, get_pitchers_df_normalized

# Load models
BATTERS_MODEL_PATH = 'data/models/batters_model.joblib'
PITCHERS_MODEL_PATH = 'data/models/pitchers_model.joblib'

batters_model = joblib.load(BATTERS_MODEL_PATH)
pitchers_model = joblib.load(PITCHERS_MODEL_PATH)

# Load data
batters_df = get_batters_df()
pitchers_df = get_pitchers_df()
batters_normalized = get_batters_df_normalized()
pitchers_normalized = get_pitchers_df_normalized()

# Get min and max for denormalization
b_war_min = batters_df['b_war'].min()
b_war_max = batters_df['b_war'].max()
p_war_min = pitchers_df['p_war'].min()
p_war_max = pitchers_df['p_war'].max()

def denormalize(normalized_val, min_val, max_val):
    if pd.isna(normalized_val):
        return None
    return normalized_val * (max_val - min_val) + min_val

# Helper to get actual and predicted WAR for a player
def get_predicted_war(player_name, player_type='batter'):
    if player_type == 'batter':
        df = batters_df
        norm_df = batters_normalized
        model = batters_model
        war_col = 'b_war'
        war_min, war_max = b_war_min, b_war_max
    else:
        df = pitchers_df
        norm_df = pitchers_normalized
        model = pitchers_model
        war_col = 'p_war'
        war_min, war_max = p_war_min, p_war_max
    # Find player row
    player_row = df[df['fullName'] == player_name]
    norm_row = norm_df.loc[player_row.index]
    if player_row.empty or norm_row.empty:
        return None, None
    # Prepare input for prediction (drop WAR column)
    X = norm_row.drop(columns=[war_col], errors='ignore')
    predicted_war_norm = model.predict(X)[0]
    predicted_war = denormalize(predicted_war_norm, war_min, war_max)
    # Get actual WAR as a scalar value robustly
    actual_war_val = player_row[war_col]
    if isinstance(actual_war_val, pd.Series):
        actual_war = actual_war_val.iloc[0]
    elif isinstance(actual_war_val, np.ndarray):
        actual_war = actual_war_val[0]
    else:
        actual_war = actual_war_val
    return actual_war, predicted_war

# Update value_labels CSVs with actual and predicted WAR
def update_value_labels_csv(csv_path, player_type='batter'):
    df = pd.read_csv(csv_path)
    actual_wars = []
    predicted_wars = []
    for name in df['fullName']:
        actual, predicted = get_predicted_war(name, player_type)
        actual_wars.append(actual)
        predicted_wars.append(predicted)
    df['actual_war'] = actual_wars
    df['predicted_war'] = predicted_wars
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with actual and predicted WAR.")

if __name__ == "__main__":
    # Example usage: python get_predicted_war.py "Aaron Judge" batter
    if len(sys.argv) > 2:
        name = sys.argv[1]
        ptype = sys.argv[2]
        actual, predicted = get_predicted_war(name, ptype)
        print(f"Player: {name}\nType: {ptype}\nActual WAR: {actual}\nPredicted WAR: {predicted}")
    else:
        # Update both value_labels CSVs
        update_value_labels_csv('data/processed/batters_value_labels.csv', 'batter')
        update_value_labels_csv('data/processed/pitchers_value_labels.csv', 'pitcher') 