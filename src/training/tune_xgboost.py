from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)


df = pd.read_csv(DATA_PATH)

categorical = [
    "month",
    "season",
    "state",
    "district"
]

encoder = LabelEncoder()

for column in categorical:

    df[column] = encoder.fit_transform(
        df[column]
    )


df["date_of_record"] = pd.to_datetime(
    df["date_of_record"]
)

df["day"] = df["date_of_record"].dt.day
df["month_num"] = df["date_of_record"].dt.month
df["year"] = df["date_of_record"].dt.year

df = df.drop(columns=["date_of_record", "station_name"])


X = df.drop(columns=["RainTomorrow"])

y = df["RainTomorrow"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


param_grid = {

    "n_estimators": [200, 300],

    "max_depth": [4, 6, 8],

    "learning_rate": [0.03, 0.05, 0.1],

    "subsample": [0.8, 1.0],

    "colsample_bytree": [0.8, 1.0]

}


model = XGBClassifier(

    random_state=42,

    eval_metric="logloss"

)


grid = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    scoring="f1",

    cv=3,

    verbose=2,

    n_jobs=-1

)


grid.fit(

    X_train,
    y_train
)


print()

print("===========================")

print("BEST PARAMETERS")

print("===========================")

print(grid.best_params_)

print()

print("BEST CV F1")

print(grid.best_score_)