#!/usr/bin/env python3
"""Script to extract one sample per book from raw audio metadata.

This script reads raw_audio_metadata.csv and creates a new CSV file with one
entry per book_id, maintaining all the original fields.
"""

import pandas as pd
from pathlib import Path


def load_raw_metadata(file_path: str) -> pd.DataFrame:
    """Loads the raw audio metadata CSV file.
    
    Args:
        file_path: Path to the raw audio metadata CSV file.
        
    Returns:
        DataFrame containing the raw audio metadata.
    """
    return pd.read_csv(file_path)


def extract_book_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts one sample per book_id from the input DataFrame.
    
    Args:
        df: Input DataFrame containing all audio metadata.
        
    Returns:
        DataFrame containing one sample per book_id.
    """
    # Group by book_id and take the first entry for each book
    return df.groupby('book_id').first().reset_index()


def main() -> None:
    """Main function to process the audio metadata."""
    data_dir = Path('data')
    input_file = data_dir / 'raw_audio_metadata.csv'
    output_file = data_dir / 'book_samples.csv'
    
    # Load and process the data
    raw_data = load_raw_metadata(input_file)
    book_samples = extract_book_samples(raw_data)
    
    # Save the results
    book_samples.to_csv(output_file, index=False)
    print(f'Successfully created {output_file} with {len(book_samples)} book samples')


if __name__ == '__main__':
    main() 