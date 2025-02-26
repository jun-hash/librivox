import re
import shutil
from pathlib import Path

def extract_number(filename: str) -> str:
    """Extracts trailing numeric portion from filename."""
    match = re.search(r'\d+$', filename)
    return match.group() if match else ''

def merge_url_files(file1: Path, file2: Path, output_dir: Path) -> None:
    """Merges two URL files and names result with merged numbers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num1 = extract_number(file1.stem)
    num2 = extract_number(file2.stem)
    merged_name = f"merged_{num1}_{num2}_urls.txt"
    
    merged_path = output_dir / merged_name
    
    with merged_path.open('w') as out_f:
        for f in [file1, file2]:
            with f.open() as in_f:
                shutil.copyfileobj(in_f, out_f)

def main() -> None:
    """Processes URL files for merging."""
    url_dir = Path('url_list')
    merged_url_dir = Path('merged_url_list')
    
    url_files = sorted(
        url_dir.glob('merged_not_processed_urls_part_*.txt'),
        key=lambda f: int(extract_number(f.stem))
    )
    
    for i in range(0, len(url_files), 2):
        if i+1 >= len(url_files):
            break
        file1, file2 = url_files[i], url_files[i+1]
        merge_url_files(file1, file2, merged_url_dir)

if __name__ == '__main__':
    main() 