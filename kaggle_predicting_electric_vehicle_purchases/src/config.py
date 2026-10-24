"""Shared settings for every notebook: paths, seed, column names."""
from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "playground-series-s6e9"   # raw Kaggle files (never edited)
ORIG_FILE = ROOT / "original_data" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"  # 10k real-source rows
FOLDS_DIR = ROOT / "folds"                   # fixed cross-validation splits
OOF_DIR = ROOT / "oof"                       # out-of-fold predictions on train
PREDS_DIR = ROOT / "preds"                   # predictions on test
SUB_DIR = ROOT / "submissions"               # files ready to upload to Kaggle
EXPERIMENTS_FILE = ROOT / "experiments.csv"  # log of every model and its score

# Fixed settings
SEED = 42
N_FOLDS = 5
ID_COL = "id"
TARGET = "Will_Buy_EV"

# Columns
NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
FEATURES = NUM_COLS + CAT_COLS
