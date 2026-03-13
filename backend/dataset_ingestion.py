import uuid
import pandas as pd
from pathlib import Path

def ensure_data_dirs(base_dir="data"):
    """
    Create data/raw and data/processed directories if they do not exist.
    """
    base_path = Path(base_dir)
    (base_path / "raw").mkdir(parents=True, exist_ok=True)
    (base_path / "processed").mkdir(parents=True, exist_ok=True)

def load_dataset(file) -> pd.DataFrame:

    """
    Load dataset from file path or file-like object (e.g. Streamlit UploadedFile).
    Automatically normalizes column names.
    
    """

    filename = file.name if hasattr(file, "name") else str(file)

    if filename.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)

    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(file)

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    df = normalize_columns(df)

    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    """
    Convert column names to lowercase, replace spaces with underscores, and strip whitespace.
    """
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    return df

def save_raw_dataset(df: pd.DataFrame, base_dir="data"):
    """
    Save the dataset into data/raw/.
    Generate a unique dataset_id using uuid.
    Return dataset_id and saved file path.
    """
    ensure_data_dirs(base_dir)
    dataset_id = str(uuid.uuid4())
    save_path = Path(base_dir) / "raw" / f"{dataset_id}.csv"
    
    # Save the dataframe as CSV
    df.to_csv(save_path, index=False)
    
    return dataset_id, str(save_path)
