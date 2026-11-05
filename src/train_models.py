import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from model_pipeline import load_and_clean_data, split_data, build_models


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    return metrics


def main():
    df = load_and_clean_data()

    X_train, X_test, y_train, y_test = split_data(df)

    models = build_models()

    results = []

    os.makedirs("models", exist_ok=True)

    test_data = X_test.copy()
    test_data["experimental class"] = y_test.values
    test_data.to_csv("test_data.csv", index=False)


    print("QSAR BIODEGRADATION - MODEL TRAINING")

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        model_filename = model_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, f"models/{model_filename}")

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = model_name
        results.append(metrics)

        print(f"Completed. Saved as models/{model_filename}")

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]
    ]

    print("MODEL COMPARISON")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        "model_results.csv",
        index=False
    )

    print("\nResults saved to model_results.csv")
    print("Test data saved to test_data.csv")


if __name__ == "__main__":
    main()