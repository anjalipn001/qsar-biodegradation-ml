import pandas as pd

INPUT_PATH = "data/biodeg.csv"

FEATURE_NAMES = [
    "SpMax_L",
    "J_Dz(e)",
    "nHM",
    "F01[N-N]",
    "F04[C-N]",
    "NssssC",
    "nCb-",
    "C%",
    "nCp",
    "nO",
    "F03[C-N]",
    "SdssC",
    "HyWi_B(m)",
    "LOC",
    "SM6_L",
    "F03[C-O]",
    "Me",
    "Mi",
    "nN-N",
    "nArNO2",
    "nCRX3",
    "SpPosA_B(p)",
    "nCIR",
    "B01[C-Br]",
    "B03[C-Cl]",
    "N-073",
    "SpMax_A",
    "Psi_i_1d",
    "B04[C-Br]",
    "SdO",
    "TI2_L",
    "nCrt",
    "C-026",
    "F02[C-N]",
    "nHDon",
    "SpMax_B(m)",
    "Psi_i_A",
    "nN",
    "SM6_B(m)",
    "nArCOOR",
    "nX",
]

TARGET_NAME = "experimental class"


def load_dataset():
    df = pd.read_csv(
        INPUT_PATH,
        sep=";",
        header=None
    )

    df.columns = FEATURE_NAMES + [TARGET_NAME]

    return df


def clean_dataset(df):
    initial_rows = len(df)

    # Remove exact duplicate observations
    df = df.drop_duplicates().reset_index(drop=True)

    removed_rows = initial_rows - len(df)

    # Encode target: RB = 1, NRB = 0
    df[TARGET_NAME] = df[TARGET_NAME].map({
        "RB": 1,
        "NRB": 0
    })

    if df[TARGET_NAME].isnull().any():
        raise ValueError("Unexpected target value found.")

    return df, removed_rows


if __name__ == "__main__":
    df = load_dataset()

    print("QSAR BIODEGRADATION DATA PREPARATION")

    print(f"\nOriginal shape: {df.shape}")

    df, removed_rows = clean_dataset(df)

    print(f"Duplicate rows removed: {removed_rows}")
    print(f"Cleaned shape: {df.shape}")

    print("\nTarget distribution:")
    print(df[TARGET_NAME].value_counts())

    print("\nTarget mapping:")
    print("RB  -> 1")
    print("NRB -> 0")

    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows remaining:", df.duplicated().sum())

    print("\nFeature names:")
    print(df.columns.tolist())