import joblib
import pandas as pd

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm"
]

TEST_PATH = "test_data.csv"
TARGET_NAME = "experimental class"


def main():
    test_data = pd.read_csv(TEST_PATH)

    X_test = test_data.drop(columns=[TARGET_NAME])
    y_test = test_data[TARGET_NAME]

    print("SAVED MODEL VERIFICATION")

    print(f"\nTest data shape: {test_data.shape}")

    for model_name in MODEL_NAMES:
        model_path = f"models/{model_name}.pkl"

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        print(
            f"{model_name}: "
            f"loaded successfully | "
            f"predictions={len(predictions)}"
        )

    print("\nAll saved models loaded and predicted successfully.")


if __name__ == "__main__":
    main()