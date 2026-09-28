from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_german_credit_data():
    # Fetch dataset
    dataset = fetch_ucirepo(id=144)

    # Get features and target
    X = dataset.data.features
    y = dataset.data.targets

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    return df


if __name__ == "__main__":
    df = load_german_credit_data()

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save dataset
    output_path = data_dir / "german_credit.csv"
    df.to_csv(output_path, index=False)

    print("Dataset shape:", df.shape)
    print("Dataset saved to:", output_path)