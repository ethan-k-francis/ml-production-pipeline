"""
Synthetic Credit Card Fraud Dataset Generator
==============================================
Generates a realistic synthetic dataset for credit card fraud detection.
Uses sklearn's make_classification to create a binary classification problem
with features that mimic real fraud data (PCA-transformed components + amount/time).

Why synthetic data?
- Real fraud datasets have privacy constraints and redistribution issues
- Synthetic data lets anyone reproduce the full pipeline without external downloads
- We control the class imbalance ratio to match real-world fraud rates (~1-2%)
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_fraud_dataset(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic credit fraud dataset.

    The dataset mimics real credit card transaction data with:
    - PCA-like features (v1-v10): simulated principal components from original features
    - amount: transaction amount (log-normal distribution, realistic for purchases)
    - time: seconds elapsed from first transaction (simulates temporal ordering)

    Args:
        n_samples: total number of transactions to generate
        fraud_ratio: proportion of fraudulent transactions (default 2%)
        random_state: seed for reproducibility
    """
    rng = np.random.RandomState(random_state)

    # Calculate class split — fraud is the minority class
    n_fraud = int(n_samples * fraud_ratio)
    n_samples - n_fraud

    # make_classification generates a linearly separable dataset with some noise.
    # n_informative=8 means 8 features carry real signal, 2 are redundant noise.
    # flip_y=0.03 adds 3% label noise — realistic since fraud labels aren't perfect.
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.03,
        random_state=random_state,
    )

    # Name features v1-v10 to mimic PCA-transformed components (like the real
    # Kaggle credit card dataset where original features are hidden for privacy)
    feature_names = [f"v{i}" for i in range(1, 11)]
    df = pd.DataFrame(features, columns=feature_names)

    # Generate realistic transaction amounts using log-normal distribution.
    # Most transactions are small ($5-$50), with a long tail of large purchases.
    # Fraudulent transactions tend to be larger on average.
    amounts = rng.lognormal(mean=3.0, sigma=1.2, size=n_samples)
    fraud_mask = labels == 1
    amounts[fraud_mask] *= rng.uniform(1.5, 5.0, size=fraud_mask.sum())
    df["amount"] = np.round(amounts, 2)

    # Simulate time as seconds from the start of the observation window.
    # Spans ~48 hours (172800 seconds), sorted to mimic temporal ordering.
    df["time"] = np.sort(rng.uniform(0, 172800, size=n_samples)).astype(int)

    # Target column: 0 = legitimate, 1 = fraud
    df["is_fraud"] = labels

    return df


def main() -> None:
    """Generate the dataset and save to CSV in the data/ directory."""
    # Ensure output directory exists relative to project root
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "transactions.csv")

    print("Generating synthetic credit card fraud dataset...")
    df = generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02)

    # Report dataset statistics before saving
    n_fraud = df["is_fraud"].sum()
    n_total = len(df)
    print(f"  Total transactions: {n_total:,}")
    print(f"  Fraudulent: {n_fraud:,} ({n_fraud / n_total * 100:.1f}%)")
    print(
        f"  Legitimate: {n_total - n_fraud:,} ({(n_total - n_fraud) / n_total * 100:.1f}%)"
    )
    print(f"  Features: {[c for c in df.columns if c != 'is_fraud']}")

    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
