"""Script to download LibriVox audio files and extract first 8 seconds.

This script downloads audio files from LibriVox and creates 8-second clips using ffmpeg.
"""

import os
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import Tuple


def ensure_directories() -> Tuple[Path, Path]:
    """Creates necessary directories if they don't exist.
    
    Returns:
        Tuple containing paths to data and cut directories.
    """
    data_dir = Path('sample/data')
    cut_dir = Path('sample/small_cut')
    
    data_dir.mkdir(parents=True, exist_ok=True)
    cut_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir, cut_dir


def download_audio(url: str, output_path: Path) -> bool:
    """Downloads audio file from given URL.
    
    Args:
        url: Audio file URL.
        output_path: Path where the file will be saved.
        
    Returns:
        bool: True if download successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def cut_audio(input_path: Path, output_path: Path, duration: int = 8) -> bool:
    """Cuts audio file to specified duration using ffmpeg.
    
    Args:
        input_path: Path to input audio file.
        output_path: Path to save cut audio file.
        duration: Duration in seconds to cut (default: 8).
        
    Returns:
        bool: True if cutting successful, False otherwise.
    """
    try:
        cmd = [
            'ffmpeg', '-i', str(input_path), '-t', str(duration),
            '-acodec', 'copy', str(output_path), '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cutting {input_path}: {e}")
        return False


def process_audio(row: pd.Series, data_dir: Path, cut_dir: Path) -> None:
    """Processes a single audio file: downloads and cuts it.
    
    Args:
        row: Pandas Series containing audio metadata.
        data_dir: Directory to save downloaded files.
        cut_dir: Directory to save cut files.
    """
    file_id = row['file_id']
    audio_link = row['audio_link']
    
    download_path = data_dir / f"{file_id}.mp3"
    cut_path = cut_dir / f"{file_id}_8sec.mp3"
    
    if not download_path.exists():
        if download_audio(audio_link, download_path):
            print(f"Downloaded: {file_id}")
    
    if download_path.exists() and not cut_path.exists():
        if cut_audio(download_path, cut_path):
            print(f"Cut: {file_id}")


def main():
    """Main function to orchestrate the download and cutting process."""
    # Create necessary directories
    data_dir, cut_dir = ensure_directories()
    
    # Read CSV file
    df = pd.read_csv('data/book_samples.csv')
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _, row in df.iterrows():
            executor.submit(process_audio, row, data_dir, cut_dir)


if __name__ == '__main__':
    main() 