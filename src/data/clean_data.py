import os
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup

def main():
    files = []
    path = 'data/raw'
    
    if not os.path.exists(path):
        print(f"Directory '{path}' does not exist. Please run load_data.py first to download the files.")
        return
    
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

        # Find the player's salary
        meta = soup.find(id="meta")
        for p in meta.find_all("p"):
            if "Contract Status" in p.text:
                # Extract salary value using regex
                import re
                contract_text = p.text
                # Look for salary pattern like $20.5M or $15.2M
                salary_match = re.search(r'\$(\d+\.?\d*[MBK]?)', contract_text)
                if salary_match:
                    salary = salary_match.group(1)  # Get just the value without the $
                    print(f"Salary: ${salary}")
                else:
                    print("No salary found in contract status")

            # Find the table wrapper for pitching
        div = soup.find(id="div_players_value_pitching")
        if div:
            row = div.find(id="players_value_pitching.2024")
            if row:
                stats_dict = {}
                stats_dict['fullName'] = player_name
                year = row.find("th", {"data-stat": "year_id"}).text.strip()
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
        
        # Now handle batting
        div = soup.find(id="div_players_standard_batting")
        if div:
            row = div.find(id="players_standard_batting.2024")
            if row:
                stats_dict = {}
                stats_dict['fullName'] = player_name
                year = row.find("th", {"data-stat": "year_id"}).text.strip()
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

    # Create database directory if it doesn't exist
    db_dir = 'data/processed'
    os.makedirs(db_dir, exist_ok=True)

    # Create database connection
    db_path = os.path.join(db_dir, 'baseball_stats.db')
    
    # Delete existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)

    # Save pitchers dataframe to database
    if not pitchers_df.empty:
        pitchers_df.to_sql('pitchers', conn, if_exists='replace', index=False)
        print(f"Saved {len(pitchers_df)} pitcher records to database")

    # Save batters dataframe to database  
    if not batters_df.empty:
        batters_df.to_sql('batters', conn, if_exists='replace', index=False)
        print(f"Saved {len(batters_df)} batter records to database")

    # Close the connection
    conn.close()
    print(f"Database saved to: {db_path}")

def get_batters_df():
    conn = sqlite3.connect('data/processed/baseball_stats.db')
    batters_df = pd.read_sql_query("SELECT * FROM batters WHERE salary > 0", conn)
    conn.close()
    return batters_df

def get_pitchers_df():
    conn = sqlite3.connect('data/processed/baseball_stats.db')
    pitchers_df = pd.read_sql_query("SELECT * FROM pitchers WHERE salary > 0", conn)
    conn.close()
    return pitchers_df


if __name__ == "__main__":
    main()