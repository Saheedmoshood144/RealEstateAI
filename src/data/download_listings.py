"""
Download the verified CC0 synthetic housing listings dataset.

Dataset:
    Stratum Housing Market Dataset

Purpose:
    Used for property search and recommendation features.
"""

from pathlib import Path

import requests


DATA_URL = "https://stratumstats.com/datasets/housing-market/housing_market.csv"

OUTPUT_PATH = Path("data/raw/housing_market.csv")


def download_dataset() -> None:
    """Download the housing listings dataset."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading listings dataset...")
    print(f"Source: {DATA_URL}")

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    OUTPUT_PATH.write_bytes(response.content)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    download_dataset()