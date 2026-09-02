from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA = Path("data/raw/california_housing.csv")
PROCESSED_DIR = Path("data/processed")

TARGET = "MedHouseVal"


def preprocess_data():
    """Load and split the California housing dataset."""

    df = pd.read_csv(RAW_DATA)

    print("Dataset shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget:")
    print(TARGET)


if __name__ == "__main__":
    preprocess_data()