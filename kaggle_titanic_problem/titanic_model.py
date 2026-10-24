"""Titanic survival model: clean data, compare two models, write a Kaggle submission."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "titanic"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")


def prepare(df):
    """Turn raw columns into numbers the model can use."""
    out = pd.DataFrame(index=df.index)
    out["Pclass"] = df["Pclass"]
    out["IsFemale"] = (df["Sex"] == "female").astype(int)

    # Title from the name, e.g. "Braund, Mr. Owen" -> "Mr"
    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0]
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    for t in ["Mr", "Mrs", "Miss", "Master", "Rare"]:
        out[f"Title_{t}"] = (title == t).astype(int)

    # Fill missing age with the typical age for that title (learned from train)
    out["Age"] = df["Age"].fillna(title.map(AGE_BY_TITLE))

    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["Fare"] = df["Fare"].fillna(FARE_MEDIAN)
    out["HasCabin"] = df["Cabin"].notna().astype(int)

    embarked = df["Embarked"].fillna("S")  # most common port
    for port in ["C", "Q", "S"]:
        out[f"Embarked_{port}"] = (embarked == port).astype(int)
    return out


# Values used to fill gaps come from train only, so test data does not leak in
train_title = train["Name"].str.extract(r",\s*([^\.]+)\.")[0]
train_title = train_title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
train_title = train_title.where(train_title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
AGE_BY_TITLE = train.groupby(train_title)["Age"].median()
FARE_MEDIAN = train["Fare"].median()

X = prepare(train)
y = train["Survived"]
X_test = prepare(test)

models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random forest": RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=42
    ),
}

# Baseline: the sample file's rule "women live, men die"
print(f"Baseline (women live)  : {(X['IsFemale'] == y).mean():.3f}")

scores = {}
for name, model in models.items():
    s = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    scores[name] = s.mean()
    print(f"{name:<23}: {s.mean():.3f} (+/- {s.std():.3f})")

best = max(scores, key=scores.get)
print(f"Best model: {best}")

model = models[best].fit(X, y)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)})
submission.to_csv(Path(__file__).parent / "submission.csv", index=False)
print(f"Wrote submission.csv ({len(submission)} rows, {submission['Survived'].mean():.0%} predicted survived)")
