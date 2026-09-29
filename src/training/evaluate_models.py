from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_models(
    models,
    scaler,
    X_test,
    y_test
):

    results = {}

    X_test_scaled = scaler.transform(
        X_test
    )

    for model_name, model in models.items():

        if model_name == "Logistic Regression":

            prediction = model.predict(
                X_test_scaled
            )

            probability = model.predict_proba(
                X_test_scaled
            )[:, 1]

        else:

            prediction = model.predict(
                X_test
            )

            probability = model.predict_proba(
                X_test
            )[:, 1]

        results[model_name] = {

            "accuracy": accuracy_score(
                y_test,
                prediction
            ),

            "precision": precision_score(
                y_test,
                prediction
            ),

            "recall": recall_score(
                y_test,
                prediction
            ),

            "f1_score": f1_score(
                y_test,
                prediction
            ),

            "roc_auc": roc_auc_score(
                y_test,
                probability
            )
        }

    return results