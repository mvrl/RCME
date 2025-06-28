import pandas as pd
import numpy as np

def filter_catalog_csv(csv_path: str, split: str, save_path: str):
    """
    Filters a catalog CSV file based on the specified split and ensure entire taxonomic labels are available.

    Args:
        csv_path (str): Path to the input CSV file.
        split (str): The split to filter by (e.g., 'train', 'val', 'train_small').
        save_path (str): Path to save the filtered CSV file.
    """
    cv = pd.read_csv(csv_path)
    filtered_cv = cv[(cv["split"]==split) & (pd.notna(cv['kingdom'])) & (pd.notna(cv['phylum'])) & (pd.notna(cv['class'])) & (pd.notna(cv['order']))& (pd.notna(cv['family'])) & (pd.notna(cv['genus'])) & (pd.notna(cv['species']))]
    filtered_cv = filtered_cv.reset_index(drop=True)
    filtered_cv.to_csv(save_path, index=False)
