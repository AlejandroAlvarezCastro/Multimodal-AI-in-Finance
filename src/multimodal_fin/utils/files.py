"""
File and path utilities for the multimodal_fin package.

Includes helpers for reading CSVs, JSONs, and locating conference files.
"""
from pathlib import Path
from typing import List, Any
import pandas as pd
import json


def read_paths_csv(csv_path: str) -> List[str]:
    """
    Read a CSV file with a 'path' column and return the list of paths.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        List[str]: List of directory paths for conferences.

    Raises:
        ValueError: If the CSV does not contain a 'path' column.
    """
    df = pd.read_csv(csv_path)
    if 'path' not in df.columns:
        raise ValueError("Input CSV must contain a 'path' column.")
    return df['path'].dropna().tolist()


def make_processed_path(original: Path) -> Path:
    """
    Determine the output directory for processed data based on the original path.

    If the segment 'companies' exists in the path, it is replaced with 'processed_companies'.
    Otherwise, appends '_processed' to the original directory name under the same parent.

    Args:
        original (Path): Original conference directory path.

    Returns:
        Path: Output directory path for processed results.
    """
    parts = list(original.parts)
    try:
        idx = parts.index('companies')
        parts[idx] = 'processed_companies'
        return Path(*parts)
    except ValueError:
        # Fallback: append suffix
        return original.parent / f"{original.name}_processed"


def read_json_file(json_path: Path) -> Any:
    """
    Read a JSON file and return the parsed object.

    Args:
        json_path (Path): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_level3_json(directory: Path) -> Path:
    """
    Locate the 'LEVEL_3.json' file within a conference directory.

    Args:
        directory (Path): Conference directory.

    Returns:
        Path: Full path to LEVEL_3.json.

    Raises:
        FileNotFoundError: If the file is not present.
    """
    candidate = directory / 'LEVEL_3.json'
    if not candidate.exists():
        raise FileNotFoundError(f"LEVEL_3.json not found in {directory}")
    return candidate


def find_audio_file(directory: Path) -> Path:
    """
    Locate an audio file (mp3, wav, or flac) within a conference directory.

    Args:
        directory (Path): Conference directory.

    Returns:
        Path: Path to the first matching audio file.

    Raises:
        FileNotFoundError: If no audio file is found.
    """
    for ext in ('mp3', 'wav', 'flac'):
        for file in directory.glob(f'*.{ext}'):
            return file
    raise FileNotFoundError(f"No audio file found in {directory}")