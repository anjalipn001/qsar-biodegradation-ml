import pandas as pd

DATA_PATH = "data/biodeg.csv"

df = pd.read_csv(DATA_PATH, sep=";", header=None)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nShape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts())

print("\nUnique target values:")
print(df.iloc[:, -1].unique())

print("\nDuplicate rows with all columns:")
duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

print(duplicates.to_string(index=False))

print("\nDuplicate feature rows with conflicting targets:")
feature_columns = df.columns[:-1]
target_column = df.columns[-1]

duplicate_features = df[
    df.duplicated(subset=feature_columns, keep=False)
].sort_values(by=list(feature_columns))

conflicting = (
    duplicate_features
    .groupby(list(feature_columns))[target_column]
    .nunique()
)

print(
    "Number of duplicated feature groups with multiple target labels:",
    (conflicting > 1).sum()
)

print("\nFeature uniqueness:")
feature_columns = df.columns[:-1]

for column in feature_columns:
    unique_count = df[column].nunique()
    if unique_count <= 1:
        print(f"Constant feature: {column} -> {unique_count} unique value(s)")

print("\nFeature ranges:")
for column in feature_columns:
    print(
        f"{column}: "
        f"min={df[column].min():.4f}, "
        f"max={df[column].max():.4f}, "
        f"unique={df[column].nunique()}"
    )