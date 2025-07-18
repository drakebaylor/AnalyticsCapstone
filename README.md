# Moneyball 2.0 – Predicting Player Value from Performance Metrics (Data Analytics Capstone Project)

This capstone project explores how player performance metrics can predict Wins Above Replacement (WAR) and identify undervalued MLB players based on salary. The workflow includes scraping, cleaning, modeling, clustering, and visualization to uncover actionable insights for player valuation.

## Motivation & Summary of Findings

Major League Baseball teams are always searching for undervalued talent. By leveraging modern data analytics, we can predict player value (WAR) and compare it to salary, revealing market inefficiencies. This project demonstrates that performance metrics can reliably predict WAR and that clustering can segment players into value categories, highlighting both overvalued and undervalued players.

## Project Objectives
- Build a regression model to predict WAR
- Normalize WAR against salary to find under/overvalued players
- Cluster players based on value and performance
- Visualize insights through interactive dashboards and figures

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AnalyticsCapstone
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow & Usage

The project is organized into modular scripts and Jupyter notebooks. The main workflow is as follows:

### 1. Data Gathering
Scrape player HTML pages for the 2025 MLB season.
```bash
python src/data/load_data.py
```
- Downloads raw HTML files to `data/raw/`.
- **Expected output:** Console messages for each player, files in `data/raw/`.

### 2. Data Cleaning & Database Creation
Parse HTML files, extract stats and salary, and save to SQLite and CSV.
```bash
python src/data/clean_data.py
```
- Outputs: `data/processed/baseball_stats.db`, `data/processed/batters.csv`, `data/processed/pitchers.csv`
- **Expected output:** Console messages for each file processed, summary of records saved.

### 3. Modeling
Train regression models to predict WAR for batters and pitchers.
```bash
python src/models/train_model.py
```
- Outputs: `data/models/batters_model.joblib`, `data/models/pitchers_model.joblib`
- **Expected output:** Models saved, optionally add print statements for metrics.

### 4. Clustering & Value Segmentation
Cluster players by value and visualize results.
```bash
python src/clustering/clustering.py
```
- Outputs: `reports/figures/batters_value_segments.png`, `reports/figures/pitchers_value_segments.png`,
  `data/processed/batters_value_labels.csv`, `data/processed/pitchers_value_labels.csv`
- **Expected output:** Console cluster profiling, figures saved, CSVs with value labels.

## Usage Examples

### Using Data Loaders in Python
```python
from src.data.clean_data import get_batters_df, get_pitchers_df
batters = get_batters_df()
pitchers = get_pitchers_df()
print(batters.head())
```

### Loading and Using Trained Models
```python
import joblib
batters_model = joblib.load('data/models/batters_model.joblib')
# Predict using the model
predictions = batters_model.predict(batters.drop(columns=['b_war']))
```

## Notebooks
- `01_data_gathering.ipynb`: Scrape and collect player HTML data
- `02_data_cleaning.ipynb`: Parse, clean, and store player stats
- `03_exploratory_analysis.ipynb`: EDA, visualizations, and feature analysis
- `04_modeling.ipynb`: Regression modeling for WAR prediction
- `05_clustering.ipynb`: Clustering and value segmentation analysis

## Outputs
- **Processed Data:** `data/processed/baseball_stats.db`, `batters.csv`, `pitchers.csv`
- **Trained Models:** `data/models/batters_model.joblib`, `pitchers_model.joblib`
- **Figures:** Correlation matrices, WAR distributions, value segment plots in `reports/figures/`
- **Cluster Labels:** `data/processed/batters_value_labels.csv`, `pitchers_value_labels.csv`

## Example Figures
- ![Batter WAR Distribution](reports/figures/batter_war_distribution.png)
- ![Batter Value Segments](reports/figures/batters_value_segments.png)

## Repository Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for EDA, modeling, and visualization
- `src/`: Modular Python code for reusable pipelines
- `reports/`: Final report and figures

