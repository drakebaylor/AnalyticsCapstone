"""
Data cleaning and database creation for baseball player stats.
- Parses raw HTML player files
- Extracts salary, batting, and pitching stats
- Saves processed data to SQLite database and CSV
- Provides functions to load and normalize dataframes
"""

import os
import pandas as pd
import sqlite3
import re
from bs4 import BeautifulSoup

# ----------------------
# Main Data Extraction
# ----------------------
def get_dataframes():
    """
    Parses all HTML files in data/raw, extracts player stats and salary, and saves to SQLite DB and CSV files.
    
    Returns:
        None. Saves processed data to 'data/processed/baseball_stats.db', 'batters.csv', and 'pitchers.csv'.
    """
    files = []
    path = 'data/raw'
    
    if not os.path.exists(path):
        print(f"Directory '{path}' does not exist. Please run load_data.py first to download the files.")
        return
    
    # Gather all HTML files in the raw data directory
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            files.append(full_path)
    
    print(f"Found {len(files)} files in {path}")
    for file in files:
        print(f"  - {file}")

    batters_df = pd.DataFrame()
    pitchers_df = pd.DataFrame()
    num_files = len(files)
    current_file = 0
    for file in files:
        current_file += 1
        print(f"Processing file {current_file} of {num_files}: {file}")
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        player_name = file.split(".html")[0].split("\\")[-1]
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # -----------
        # Salary Extraction
        # -----------
        salary = 0
        meta = soup.find(id="meta")
        if meta:
            for p in meta.find_all("p"):
                if "Contract Status" in p.text:
                    contract_text = p.text
                    # Look for contract pattern like '5 yr/$300M', '5 yrs/$300M', or just '$20.5M'
                    contract_match = re.search(r'(\d+) yr[s]?/\$(\d+\.?\d*[MBk]?)', contract_text)
                    if contract_match:
                        years = int(contract_match.group(1))
                        total_salary = contract_match.group(2)
                        # Convert total_salary to float
                        if total_salary.endswith('M'):
                            total_salary_val = float(total_salary[:-1]) * 1_000_000
                        elif total_salary.endswith('k'):
                            total_salary_val = float(total_salary[:-1]) * 1_000
                        elif total_salary.endswith('B'):
                            total_salary_val = float(total_salary[:-1]) * 1_000_000_000
                        else:
                            total_salary_val = float(total_salary)
                        salary = total_salary_val / years
                        print(f"Contract: {years} yr, Total: ${total_salary}, Annual Salary: ${salary:,.0f}")
                    else:
                        # Fallback: Look for salary pattern like $20.5M or $15.2M
                        salary_match = re.search(r'\$(\d+\.?\d*[MBk]?)', contract_text)
                        if salary_match:
                            salary = salary_match.group(1)
                            if salary.endswith('M'):
                                salary = float(salary[:-1]) * 1_000_000
                            elif salary.endswith('k'):
                                salary = float(salary[:-1]) * 1_000
                            elif salary.endswith('B'):
                                salary = float(salary[:-1]) * 1_000_000_000
                            else:
                                salary = float(salary)
                            print(f"Salary: ${salary:,.0f}")
                        else:
                            print("No salary found in contract status")
                            salary = 0
        # -----------
        # Pitching Stats Extraction
        # -----------
        div = soup.find(id="div_players_standard_pitching")
        if div:
            row = div.find(id="players_standard_pitching.2024")
            if row:
                stats_dict = {}
                stats_dict['fullName'] = player_name
                year_th = row.find("th", {"data-stat": "year_id"})
                if year_th:
                    year = year_th.text.strip()
                else:
                    year = None
                stats_dict["year"] = year
                for cell in row.find_all("td"):
                    stat_name = cell.get("data-stat")
                    stat_value = cell.text.strip()
                    stats_dict[stat_name] = stat_value
                stats_dict['salary'] = salary
                df = pd.DataFrame([stats_dict])
                pitchers_df = pd.concat([pitchers_df, df], ignore_index=True)
            else:
                print("2025 pitching row not found.")
        else:
            print("Pitching table not found.")
        
        # -----------
        # Batting Stats Extraction
        # -----------
        div = soup.find(id="div_players_standard_batting")
        if div:
            row = div.find(id="players_standard_batting.2024")
            if row:
                stats_dict = {}
                stats_dict['fullName'] = player_name
                year_th = row.find("th", {"data-stat": "year_id"})
                if year_th:
                    year = year_th.text.strip()
                else:
                    year = None
                stats_dict["year"] = year
                for cell in row.find_all("td"):
                    stat_name = cell.get("data-stat")
                    stat_value = cell.text.strip()
                    stats_dict[stat_name] = stat_value
                stats_dict['salary'] = salary
                df = pd.DataFrame([stats_dict])
                batters_df = pd.concat([batters_df, df], ignore_index=True)
            else:
                print("2025 batting row not found.")
        else:
            print("Batting table not found.")

    # ----------------------
    # Save to SQLite Database
    # ----------------------
    db_dir = 'data/processed'
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'baseball_stats.db')
    
    # Delete existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)

    # Save pitchers dataframe to database
    if not pitchers_df.empty:
        pitchers_df = load_and_fix_dtypes(pitchers_df)
        if 'awards' in pitchers_df.columns:
            pitchers_df = pitchers_df.drop(columns=['awards'])
        pitchers_df.to_sql('pitchers', conn, if_exists='replace', index=False)
        print(f"Saved {len(pitchers_df)} pitcher records to database")

    # Save batters dataframe to database 
    if not batters_df.empty:
        batters_df = load_and_fix_dtypes(batters_df)
        if 'awards' in batters_df.columns:
            batters_df = batters_df.drop(columns=['awards'])
        batters_df.to_sql('batters', conn, if_exists='replace', index=False)
        print(f"Saved {len(batters_df)} batter records to database")

    # Close the connection
    conn.close()
    print(f"Database saved to: {db_path}")

    # Save DataFrames to CSV
    batters_csv_path = os.path.join(db_dir, 'batters.csv')
    pitchers_csv_path = os.path.join(db_dir, 'pitchers.csv')
    batters_df.to_csv(batters_csv_path, index=False)
    print(f"Batters DataFrame saved to: {batters_csv_path}")
    pitchers_df.to_csv(pitchers_csv_path, index=False)
    print(f"Pitchers DataFrame saved to: {pitchers_csv_path}")

# ----------------------
# DataFrame Loaders
# ----------------------
def get_batters_df():
    """
    Loads batters table from SQLite database, drops 'year' column, and returns DataFrame.

    Returns:
        pd.DataFrame: Batters data with salary > 0, without the 'year' column.
    """
    conn = sqlite3.connect('data/processed/baseball_stats.db')
    batters_df = pd.read_sql_query("SELECT * FROM batters WHERE salary > 0", conn)
    if 'year' in batters_df.columns:
        batters_df = batters_df.drop(columns=['year'])
    conn.close()
    return batters_df

def get_pitchers_df():
    """
    Loads pitchers table from SQLite database, drops 'year' column, drops NA, and returns DataFrame.

    Returns:
        pd.DataFrame: Pitchers data with salary > 0, without the 'year' column, NA rows dropped.
    """
    conn = sqlite3.connect('data/processed/baseball_stats.db')
    pitchers_df = pd.read_sql_query("SELECT * FROM pitchers WHERE salary > 0", conn)
    if 'year' in pitchers_df.columns:
        pitchers_df = pitchers_df.drop(columns=['year'])
    pitchers_df = pitchers_df.dropna()
    conn.close()
    return pitchers_df

# ----------------------
# Normalization Helpers
# ----------------------
def get_batters_df_normalized():
    """
    Returns normalized numeric columns of batters DataFrame (min-max scaling).

    Returns:
        pd.DataFrame: Normalized numeric columns of batters data.
    """
    batters_df = get_batters_df()
    batters_numeric = batters_df.select_dtypes(include=['Int64', 'float'])
    batters_non_numeric = batters_df.select_dtypes(exclude=['Int64', 'float'])
    batters_normalized_numeric = (batters_numeric - batters_numeric.min()) / (batters_numeric.max() - batters_numeric.min())
    # Optionally, you can return the full DataFrame with non-numeric columns
    # batters_normalized = pd.concat([batters_normalized_numeric, batters_non_numeric.reset_index(drop=True)], axis=1)
    return batters_normalized_numeric

def get_pitchers_df_normalized():
    """
    Returns normalized numeric columns of pitchers DataFrame (min-max scaling).

    Returns:
        pd.DataFrame: Normalized numeric columns of pitchers data.
    """
    pitchers_df = get_pitchers_df()
    pitchers_numeric = pitchers_df.select_dtypes(include=['Int64', 'float'])
    pitchers_non_numeric = pitchers_df.select_dtypes(exclude=['Int64', 'float'])
    pitchers_normalized_numeric = (pitchers_numeric - pitchers_numeric.min()) / (pitchers_numeric.max() - pitchers_numeric.min())
    # Optionally, you can return the full DataFrame with non-numeric columns
    # pitchers_normalized = pd.concat([pitchers_normalized_numeric, pitchers_non_numeric.reset_index(drop=True)], axis=1)
    return pitchers_normalized_numeric

# ----------------------
# Data Type Fixer
# ----------------------
def load_and_fix_dtypes(df):
    """
    Converts columns of a DataFrame to the correct pandas dtypes.
    Uses pandas' convert_dtypes, but can be extended for manual overrides if needed.

    Args:
        df (pd.DataFrame): DataFrame to convert.
    Returns:
        pd.DataFrame: DataFrame with corrected dtypes.
    """
    # Use pandas' convert_dtypes for best-guess
    df = df.convert_dtypes()
    # Manual overrides for known columns (example: year, salary)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
    if 'salary' in df.columns:
        def parse_salary(val):
            if pd.isna(val):
                return None
            val = str(val).replace('$', '').replace(',', '').strip()
            if val.endswith('M'):
                return float(val[:-1]) * 1000000
            elif val.endswith('k'):
                return float(val[:-1]) * 1000
            elif val.endswith('B'):
                return float(val[:-1]) * 1000000000
            else:
                try:
                    return float(val)
                except Exception:
                    return None
        df['salary'] = df['salary'].map(parse_salary).astype('float')
    # Batting float columns
    float_cols = [
        'b_war', 'b_batting_avg', 'b_onbase_perc', 'b_slugging_perc', 'b_onbase_plus_slugging',
        'b_roba'
    ]
    # Pitching float columns (updated to match corrected pitcher columns)
    float_cols += [
        'p_war', 'p_ip', 'p_earned_run_avg', 'p_win_loss_perc', 'p_fip', 'p_whip',
        'p_hits_per_nine', 'p_hr_per_nine', 'p_bb_per_nine', 'p_so_per_nine', 'p_strikeouts_per_base_on_balls'
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('^\\.', '0.', regex=True), errors='coerce').astype('float')
    # Batting integer columns
    int_cols = [
        'b_games', 'b_pa', 'b_ab', 'b_r', 'b_h', 'b_doubles', 'b_triples', 'b_hr', 'b_rbi',
        'b_sb', 'b_cs', 'b_bb', 'b_so', 'b_onbase_plus_slugging_plus', 'b_rbat_plus', 'b_tb',
        'b_gidp', 'b_hbp', 'b_sh', 'b_sf', 'b_ibb', 'awards', 'b_rbat_plus'
    ]
    # Pitching integer columns (updated to match corrected pitcher columns)
    int_cols += [
        'p_g', 'p_gs', 'p_r', 'p_bfp', 'p_earned_run_avg_plus'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    # Categorical/text columns: fullName, team_name_abbr, comp_name_abbr, pos, awards
    for col in ['fullName', 'team_name_abbr', 'comp_name_abbr', 'pos', 'awards']:
        if col in df.columns:
            df[col] = df[col].astype('string')
    return df

if __name__ == "__main__":
    get_dataframes()