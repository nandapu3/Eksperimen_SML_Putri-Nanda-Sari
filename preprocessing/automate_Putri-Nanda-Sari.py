import os
import json
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(filepath)

def bin_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply binning to selected numeric columns."""
    df["Mood Score"] = pd.cut(
        df["Mood Score"], bins=[0, 4, 7, 10], labels=["Low", "Medium", "High"], include_lowest=True
    )
    df["Stress Level"] = pd.cut(
        df["Stress Level"], bins=[0, 4, 7, 10], labels=["Low", "Medium", "High"], include_lowest=True
    )
    df["Screen Time Before Bed (mins)"] = pd.cut(
        df["Screen Time Before Bed (mins)"],
        bins=[0, 30, 60, 180],
        labels=["<30 menit", "30–60 menit", ">60 menit"],
        include_lowest=True,
    )
    if "Sleep Quality" in df.columns:
        df["Sleep Quality Category"] = pd.cut(
            df["Sleep Quality"], bins=[0, 4, 7, 10], labels=["Low", "Medium", "High"], include_lowest=True
        )
    return df

def encode_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label-encode categorical columns and return mapping dicts."""
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    return df, encoders

def scale_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Standard-scale selected numeric columns."""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns if they exist."""
    cols_to_drop = [
        "Date", "Person_ID", "Gender",
        "Productivity Score", "Exercise (mins/day)", "Caffeine Intake (mg)"
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_data(input_path)
    df = drop_unused_columns(df)
    df = bin_columns(df)

    categorical_cols = [
        "Mood Score", "Stress Level", "Screen Time Before Bed (mins)"
    ]
    if "Sleep Quality Category" in df.columns:
        categorical_cols.append("Sleep Quality Category")

    df, encoders = encode_categorical(df, categorical_cols)

    numeric_cols = [
        "Age", "Sleep Start Time", "Sleep End Time",
        "Total Sleep Hours", "Screen Time Before Bed (mins)",
        "Work Hours (hrs/day)", "Stress Level", "Mood Score"
    ]
    df = scale_numeric(df, [col for col in numeric_cols if col in df.columns])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    with open(output_path.replace(".csv", "_encoders.json"), "w") as f:
        json.dump(encoders, f, indent=2)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sleep Quality dataset for ML.")
    parser.add_argument("-i", "--input", required=True, help="Path to raw input CSV")
    parser.add_argument("-o", "--output", required=True, help="Path to save processed CSV")
    args = parser.parse_args()

    processed = preprocess_data(args.input, args.output)
    print(f"✅ Preprocessing complete. {processed.shape[0]} rows × {processed.shape[1]} columns saved to {args.output}")
