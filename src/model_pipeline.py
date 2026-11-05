import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


INPUT_PATH = "data/biodeg.csv"

FEATURE_NAMES = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]",
    "NssssC", "nCb-", "C%", "nCp", "nO", "F03[C-N]", "SdssC",
    "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]", "Me", "Mi", "nN-N",
    "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]",
    "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]",
    "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon",
    "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX"
]

TARGET_NAME = "experimental class"


def load_and_clean_data():
    df = pd.read_csv(
        INPUT_PATH,
        sep=";",
        header=None
    )

    df.columns = FEATURE_NAMES + [TARGET_NAME]

    df = df.drop_duplicates().reset_index(drop=True)

    df[TARGET_NAME] = df[TARGET_NAME].map({
        "RB": 1,
        "NRB": 0
    })

    return df


def split_data(df):
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=2026,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=2026
            ))
        ]),

        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                random_state=2026
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),

        "SVM": Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            random_state=2026
        ),
        method="sigmoid",
        cv=5
    ))
])
    }

    return models


if __name__ == "__main__":
    df = load_and_clean_data()

    X_train, X_test, y_train, y_test = split_data(df)

    models = build_models()

    print("=" * 60)
    print("TRAIN / TEST PIPELINE")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nModels configured:")
    for model_name in models:
        print(f"- {model_name}")