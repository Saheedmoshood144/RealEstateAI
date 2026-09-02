from pathlib import Path

from sklearn.datasets import fetch_california_housing


RAW_DIR = Path("data/raw")


def download_california_housing() -> None:
    """Download and save the California Housing dataset."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    housing = fetch_california_housing(
        data_home=RAW_DIR,
        as_frame=True,
    )

    df = housing.frame.copy()

    output_path = RAW_DIR / "california_housing.csv"

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    download_california_housing()