import os
import json
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(filepath)

def scale_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Standard-scale selected numeric columns."""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(
    df: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label-encode categorical columns and return mapping dicts."""
    encoders: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    return df, encoders

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """End-to-end preprocessing and saving to CSV."""
    # 1. Load
    df = load_data(input_path)

    # 2. Scale numeric columns (updated)
    numeric_cols = [
        "Financial Loss (in Million $)",
        "Number of Affected Users",
        "Incident Resolution Time (in Hours)"
    ]
    df = scale_numeric(df, numeric_cols)

    # 3. Encode categoricals
    categorical_cols = [
        "Attack Type",
        "Country",
        "Target Industry",
        "Attack Source",
        "Security Vulnerability Type",
        "Defense Mechanism Used",
    ]
    df, encoders = encode_categorical(df, categorical_cols)

    # 4. Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 5. Save processed data
    df.to_csv(output_path, index=False)
    with open(output_path.replace(".csv", "_encoders.json"), "w") as f:
        json.dump(encoders, f, indent=2)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for ML pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to raw input CSV")
    parser.add_argument("-o", "--output", required=True, help="Path to save processed CSV")
    args = parser.parse_args()

    processed = preprocess_data(args.input, args.output)
    print(
        f"✅ Preprocessing complete. {processed.shape[0]} rows × {processed.shape[1]} "
        f"columns saved to {args.output}"
    )
