from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from evaluate_models import evaluate_models

from model_selection import (
    select_best_model,
    save_model_comparison
)

from save_model import save_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)


def load_dataset():

    return pd.read_csv(DATA_PATH)


def preprocess_data(df):

    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=[
            "month",
            "season",
            "state",
            "district"
        ],
        drop_first=True,
        dtype=int
    )

    df["date_of_record"] = pd.to_datetime(
        df["date_of_record"]
    )

    df["day"] = df["date_of_record"].dt.day
    df["month_num"] = df["date_of_record"].dt.month
    df["year"] = df["date_of_record"].dt.year

    df = df.drop(
        columns=[
            "date_of_record",
            "station_name"
        ]
    )

    return df


def split_dataset(df):

    X = df.drop(columns=["RainTomorrow"])

    y = df["RainTomorrow"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_models(X_train, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    models = {

        "Logistic Regression":

            LogisticRegression(
                max_iter=5000,
                random_state=42
            ),

        "Decision Tree":

            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":

            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":

            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=1.0,
                eval_metric="logloss",
                random_state=42
            )
    }

    models["Logistic Regression"].fit(
        X_train_scaled,
        y_train
    )

    for model_name, model in models.items():

        if model_name != "Logistic Regression":

            model.fit(
                X_train,
                y_train
            )

    return models, scaler



if __name__ == "__main__":

    dataset = load_dataset()

    dataset = preprocess_data(dataset)

    X_train, X_test, y_train, y_test = split_dataset(
        dataset
    )

    models, scaler = train_models(
        X_train,
        y_train
    )

    results = evaluate_models(
        models,
        scaler,
        X_test,
        y_test
)
    best_model_name = select_best_model(
        results
)
    save_model_comparison(
        results
)
    saved_path = save_model(
        models[best_model_name],
        "baseline_model"
)
    print()
    print("==============================")
    print("MODEL COMPARISON")
    print("==============================")
    for model_name, metrics in results.items():
        print()
        print(model_name)
        print("--------------------------")
        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
    )
        print(
            f"Precision : {metrics['precision']:.4f}"
    )
        print(
            f"Recall    : {metrics['recall']:.4f}"
    )
        print(
            f"F1 Score  : {metrics['f1_score']:.4f}"
    )
        print(
            f"ROC AUC   : {metrics['roc_auc']:.4f}"
    )
    print()
    print("==============================")
    print(f"BEST MODEL : {best_model_name}")
    print("==============================")
    print()
    print(f"Model saved at: {saved_path}")