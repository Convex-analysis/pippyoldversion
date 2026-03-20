#!/usr/bin/env python3
"""
Fix Python 3.8 type annotation compatibility issues.

This script converts PEP 585 type hints (Python 3.9+) to PEP 484 type hints (Python 3.8+)
by replacing:
- List[...] -> List[...]
- Dict[...] -> Dict[...]
- Tuple[...] -> Tuple[...]
- Set[...] -> Set[...]
- FrozenSet[...] -> FrozenSet[...]
"""
import os
import re
from pathlib import Path
from typing import List, Set

def fix_type_annotations(content: str) -> str:
    """Fix PEP 585 type hints to PEP 484 type hints."""
    # Type mapping
    type_mapping = {
        'list': 'List',
        'dict': 'Dict',
        'tuple': 'Tuple',
        'set': 'Set',
        'frozenset': 'FrozenSet',
    }

    # Fix each type
    for old_type, new_type in type_mapping.items():
        # Pattern: \btype\[ -> \bNewType[
        # But we need to avoid matching already capitalized versions
        pattern = r'\b' + old_type + r'\['
        content = re.sub(pattern, new_type + '[', content)

    return content

def process_file(file_path: Path) -> bool:
    """Process a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix type annotations
        new_content = fix_type_annotations(content)

        # Write back if changed
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function."""
    # Get EVO1 directory
    evo1_dir = Path(__file__).parent

    # Find all Python files
    py_files: List[Path] = []
    for pattern in ['*.py', '**/*.py']:
        py_files.extend(evo1_dir.rglob(pattern))

    # Process each file
    fixed_count = 0
    for py_file in py_files:
        if process_file(py_file):
            print(f"Fixed: {py_file.relative_to(evo1_dir)}")
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
