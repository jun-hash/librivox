import re
from pathlib import Path
import shutil

def extract_number(folder_name: str) -> str:
    """Extracts trailing numeric portion from folder name."""
    match = re.search(r'\d+$', folder_name)
    return match.group() if match else ''

def merge_folders(source1: Path, source2: Path, merged_dir: Path) -> None:
    """Merges contents of two source directories into a new directory."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    for source in [source1, source2]:
        for item in source.glob('*'):
            if item.is_file():
                shutil.copy(item, merged_dir / item.name)

def main() -> None:
    """Processes VAD folder pairs for merging."""
    base_dir = Path('vad_lists')
    vad_pairs = [
        ('vad_1', 'vad_2'),
        ('vad_3', 'vad_4'),
        ('vad_5', 'vad_6'),
        ('vad_7', 'vad_8'),
        ('vad_9', 'vad_10'),
        ('vad_11', 'vad_12'),
        ('vad_13', 'vad_14'),
        ('vad_15', 'vad_17'),
        ('vad_19', 'vad_20'),
    ]
    
    for a, b in vad_pairs:
        merged_name = f'merged_{extract_number(a)}_{extract_number(b)}'
        merge_folders(base_dir/a, base_dir/b, base_dir/merged_name)

if __name__ == '__main__':
    main() 