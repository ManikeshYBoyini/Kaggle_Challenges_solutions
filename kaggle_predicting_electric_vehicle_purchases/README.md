# Predicting Electric Vehicle Purchases

Kaggle **Playground Series – Season 6, Episode 9**
https://www.kaggle.com/competitions/playground-series-s6e9

---

## 🎯 The problem

- Each row is one person.
- We predict the **chance (0 to 1) that the person will buy an electric vehicle (EV)**.
- This is **binary classification**: the answer is Yes or No.

**Scoring:** **ROC AUC** on `Will_Buy_EV`.
- AUC measures how well we rank likely buyers above non-buyers.
- 1.0 is perfect. 0.5 is random guessing.
- ⚠️ This comes from one notebook write-up ([cnblogs post](https://www.cnblogs.com/Bony1029/p/23142803)), not from the official page. The write-up's data facts match ours exactly. Still, check the competition's "Evaluation" tab. We also track **logloss**, just in case.

---

## 📁 The data

Raw files are in `playground-series-s6e9/`. Never edit them.

| File | Rows | What it has |
|---|---|---|
| `train.csv` | 668,665 | Person details **plus the answer** (`Will_Buy_EV`) |
| `test.csv` | 286,571 | Person details, **no answer**. We predict these. |
| `sample_submission.csv` | 286,571 | The upload format: `id, Will_Buy_EV` |

**Columns**
- **Numbers:** Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned, Charging_Stations_Near_Home, Charging_Stations_Near_Work, Environmental_Concern_Level (1–5)
- **Text:** Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level

**What we learned about the data (notebook 01)**
- **Only 17.5% said "Yes".** The two answers are uneven.
- **No missing values.**
- **Train and test look the same.** A model trying to tell them apart scored AUC 0.4998, which is no better than chance. So our CV score can be trusted.
- **Odd repeated values:**
  - Income = 30000 in 9% of rows. These people rarely buy (4.4% vs 18.8% for everyone else).
  - Commute = 5.0 km in 22% of rows.
  - These look like minimum values ("floors") from the program that made the data.

---

## 📜 Key rules (full text in `rules.txt`)

- **10 submissions per day.** Pick **2 final** ones.
- **The private leaderboard decides the final rank**, not the public one. So trust our own CV score.
- Teams of up to 3. One Kaggle account only.
- Outside data is OK if it's free and public. No labeling test data by hand.
- No private code sharing. Public sharing on Kaggle is fine.

---

## 🗂️ Folder map

```
├── README.md                  ← this file
├── rules.txt                  ← competition rules
├── requirements.txt           ← Python libraries used
├── experiments.csv            ← every model + its score (auto-filled)
├── best_lgbm_params.json      ← best LightGBM settings found by Optuna
├── playground-series-s6e9/    ← raw Kaggle data
├── original_data/             ← the real 10k-row dataset the competition was made from
├── src/
│   ├── config.py              ← paths, seed, column lists
│   ├── utils.py               ← load data, folds, scoring, saving, submissions
│   └── train.py               ← 5-fold training loops for LightGBM / XGBoost / CatBoost
├── notebooks/                 ← run in order: 01, 02, 03, 04, 06, then 05
├── folds/folds.csv            ← fixed 5-fold split, shared by ALL models
├── oof/                       ← each model's predictions on train (for blending)
├── preds/                     ← each model's predictions on test
└── submissions/               ← CSV files ready to upload to Kaggle
```

---

## 🧪 Our solution, step by step

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Explore the data: charts, buy rate per column, train vs test check |
| `02_baseline.ipynb` | Makes the **5 fixed CV folds**. Trains logistic regression as a first score to beat. |
| `03_gbdt_models.ipynb` | Trains **LightGBM, XGBoost, CatBoost**. Then **Optuna** tunes LightGBM for 15 minutes. |
| `04_feature_engineering.ipynb` | Tests 4 groups of new columns. Keeps only the ones that raise CV. |
| `06_original_data.ipynb` | Adds the **original 10k-row dataset**. Tests 3 ways to use it and keeps the best. |
| `05_ensemble.ipynb` | Combines models: average, rank average and stacking. Picks the final 2 submissions. **Run it last**, after 06. |

**Key words**
- **Cross-validation (CV):** split train into 5 parts. Train on 4, score on the 5th, and repeat. The average is our trusted score.
- **Out-of-fold (OOF) predictions:** a prediction for every train row, made by a model that never saw that row. Used to score and blend models fairly.
- **Gradient boosted trees** (LightGBM / XGBoost / CatBoost): many small decision trees, each fixing the last one's mistakes. Usually the best choice for table data.
- **Optuna:** a tool that tries many model settings and keeps the best.

**How to re-run:** open the notebooks in order and run all cells. Notebook 03 takes about 50 minutes and notebook 04 about 40 (on 20 CPU cores). To change how long Optuna runs, edit `OPTUNA_MINUTES` in notebook 03.

---

## 📊 Results (5-fold CV, run on 2026-10-17)

| Model | CV AUC | CV logloss |
|---|---|---|
| Logistic regression (baseline) | 0.93809 | 0.23330 |
| LightGBM (default settings) | 0.94177 | 0.22657 |
| XGBoost (default settings) | 0.94186 | 0.22643 |
| CatBoost (default settings) | 0.94177 | 0.22659 |
| CatBoost + new columns | 0.94178 | 0.22658 |
| **LightGBM tuned by Optuna** | **0.94283** | **0.22468** |
| LightGBM tuned + "floor" columns | 0.94281 | 0.22473 |
| **LightGBM tuned + original-data hint columns** | **0.94288** | **0.22459** |
| Blend: simple average (`lgbm_orig` + `lgbm_optuna`) | **0.94299** | – |
| Blend: rank average (same 2 models) | 0.94299 | – |
| Blend: stacking (same 2 models) | 0.94299 | 0.22440 |

**Original dataset tested (notebook 06)**, all with the tuned LightGBM at learning rate 0.05:

Source: [EV Adoption Behavior & Range Anxiety](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety). It has 10,000 rows, the same columns and the same 17.5% "Yes" rate. About 2% of its number values are missing.

| Way | What | CV AUC | Kept? |
|---|---|---|---|
| none (raw) | – | 0.94280 | – |
| A. Add rows | 10k original rows added to each fold's training part | 0.94279 | ❌ |
| B. Hint columns | per column: the "Yes" rate for that value in the original data | **0.94288** | ✅ |
| C. Both | A + B | 0.94273 | ❌ |

- Adding the rows didn't help. 10k rows is tiny next to 535k training rows per fold.
- The hint columns gave a small gain (+0.00008).
- The new model also blends well with `lgbm_optuna`, raising the blend to 0.94299.

**New columns tested (notebook 04)**, all with the tuned LightGBM:

| Group | Columns | CV AUC | Kept? |
|---|---|---|---|
| none (raw) | – | 0.94280 | – |
| A. Floors | income = 30000 flag, commute = 5.0 flag | 0.94281 | ✅ (tiny gain) |
| B. Totals & ratios | total stations, income per car, commute ÷ stations | 0.94260 | ❌ |
| C. Text combos | home × subsidy, city × car, anxiety × home | 0.94263 | ❌ |
| D. Numbers as categories | cars, stations, concern level as text | 0.94258 | ❌ |

**What this tells us**
- Tree models beat the baseline by about 0.004.
- **Tuning LightGBM helped the most** (+0.001).
- New columns **didn't really help**. The tree models already find these patterns on their own.
- **Blending added almost nothing** (+0.00002, which is noise). The blend only picked the two LightGBM models, and their predictions are nearly identical. XGBoost and CatBoost were not added because they did not raise the score.

---

## ✅ Which file to submit

1. **`submissions/blend_mean.csv`**: average of `lgbm_orig` + `lgbm_optuna` (CV AUC 0.94299).
2. **`submissions/lgbm_orig.csv`**: best single model, which uses the original-data hint columns (CV AUC 0.94288).

These two score almost the same, and both hold real probabilities, so they work for either metric.
- `blend_rank` scored 0.000002 higher, which is noise. We skip it because its values are ranks, not probabilities, and it would score badly if the metric turns out to be logloss.

**Small notes on the scores**
- Optuna tuned on fold 0, which is also one of the 5 scoring folds. So `lgbm_optuna`'s CV is a little too optimistic.
- `lgbm_optuna` used learning rate 0.02 and `lgbm_fe` used 0.05, so those two rows aren't a fair comparison. The feature test inside notebook 04 *was* fair, since every run there used the same settings.

**Status:** nothing has been uploaded to Kaggle yet. Upload manually on the competition page and note the public score here.

---

## 🔜 Ideas to try next

- **More hint columns from the original data:** "Yes" rates for *pairs* of columns (for example, home charging × subsidy), and train XGBoost / CatBoost with the hint columns too.
- **Tune XGBoost and CatBoost** with Optuna too. Only LightGBM was tuned.
- **Run Optuna longer.** It only ran 46 trials in 15 minutes.
- **Average several seeds:** train the same model with 3–5 different random seeds and average the results. This makes predictions steadier.
- **Different models for blending:** for example, a neural network or a CatBoost with very different settings. Blends work best when models are different.
