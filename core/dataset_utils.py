# core/dataset_utils.py

import os
import glob
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from utils.progress import tqdm_bar
from utils.file_saver import safe_save_path
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads a dataset from a file or directory containing CSVs.
    
    Parameters:
        path (str): Path to a CSV file or directory of CSV files.
        
    Returns:
        pd.Dataframe: Combined DataDrame from the CSVs.
    """
    if os.path.isfile(path):
        return pd.read_csv(path)
    elif os.path.isdir(path):
        all_csvs = glob.glob(os.path.join(path, '*.csv'))
        return pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def build_combined_dataset(sources: list[str], output_path: str) -> pd.DataFrame:
    """
    Builds and saves a combined dataset from multiple CSV sources.
    
    Parameters:
        sources (list[str]): List of file paths or directories to load datasets from.
        output_path (str): Destination file path for the combined CSV.
    
    Returns:
        pd.DataFrame: The combined and cleaned dataset.
    """
    dataframes = []

    for src in tqdm_bar(sources, desc="Building dataset", unit="file"):
        try:
            df = load_dataset(src)
            df['source'] = os.path.basename(src)
            dataframes.append(df)
        except Exception as e:
            logger.warning("Skipped %s due to error: %s", src, e)

    if dataframes:
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info("Combined dataset shape before cleaning: %s", combined.shape)
        
        # Remove source column first so cross-file identical rows are caught by dedup
        combined.drop(columns=['source'], inplace=True, errors='ignore')
        combined.drop_duplicates(inplace=True)

        # Ensure all values are numeric
        try:
            combined = combined.apply(pd.to_numeric, errors='raise')
        except Exception as e:
            logger.error("Combined dataset contains non-numeric values: %s", e)
            raise

        logger.info("Cleaned combined dataset shape: %s", combined.shape)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info("Combined dataset saved to: %s", output_path)
        return combined
    else:
        logger.warning("No valid datasets found to combine.")
        return pd.DataFrame()


def split_dataset(df: pd.DataFrame, label_col: str = 'label', train_size: float = 0.8, stratify: bool = True, random_state: int = 42)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into test and train sets.

    Parameters:
        df (pd.DataFrame): the dataset to split.
        label_col (str): Column to use for stratification.
        train_size (float): Proportion of data to use for training.
        stratify (bool): Whether to stratify the split based on label column.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Dataframe]: Training and test datasets.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    stratify_labels = df[label_col] if stratify else None

    df_train, df_test = train_test_split(
        df, train_size=train_size, stratify=stratify_labels, random_state=random_state
    )

    logger.info("Split complete: train=%d, test=%d", len(df_train), len(df_test))
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def balance_labels(df: pd.DataFrame, label_col: str = 'label', random_state: int = 42) -> pd.DataFrame:
    """
    Undersamples the majority class to create a balanced dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a label column.
        label_col (str): Column name for the binary class label.
        random_state (42): Seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced dataset with equal samples from each class.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    class_counts = df[label_col].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Label column must have exactly two classes to balance.")

    logger.info("Class distribution before balancing:\n%s", class_counts)

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    df_minority = df[df[label_col] == minority_class]
    df_majority = df[df[label_col] == majority_class]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=random_state
    )

    df_balanced = pd.concat([df_minority, df_majority_downsampled])
    logger.info("Balanced dataset created with %d samples per class", len(df_minority))

    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def run_dataset_utils(args) -> None:
    """
    Command-line interface handler for dataset operations.

    Parameters:
        args: Parsed command-line arguments containing 'combine', 'balance', 'split', and 'output' options.

    This dispatcher handles the flow of the following dataset transformations:
        - Combining multiple datasets.
        - Balancing class distribution.
        - Spltting into test/train sets.

    Uses config defaults and safe file naming when needed.
    """
    default_output = config['dataset_utils']['output_path']
    output_path = args.output or safe_save_path(default_output)

    df = None

    if args.combine:
        logger.info("Combining datasets from: %s", args.combine)
        build_combined_dataset(args.combine, output_path)
        df = pd.read_csv(output_path)

    if args.balance:
        if df is None:
            df = pd.read_csv(output_path)
        df = balance_labels(df)
        output_path = safe_save_path(output_path.replace(".csv", "_balanced.csv"))
        df.to_csv(output_path, index=False)
        logger.info("Saved balanced dataset to: %s", output_path)

    if args.split:
        if df is None:
            df = pd.read_csv(output_path)
        train_df, test_df = split_dataset(df)
        base = output_path.replace(".csv", "")
        train_path = safe_save_path(f"{base}_train.csv")
        test_path = safe_save_path(f"{base}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Saved split datasets:\n- %s\n- %s", train_path, test_path)

    if not (args.combine or args.balance or args.split):
        logger.warning("No dataset operation specified.")
    else:
        logger.info("Dataset transformation complete.")
