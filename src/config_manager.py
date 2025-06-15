"""
Handle loading of puzzle definitions from JSON and solver profiles from INI files.
"""
import configparser
import json
import os
from typing import Dict


def load_puzzle_definition(puzzle_number: int, puzzles_file_path: str) -> Dict:
    """
    Loads a specific puzzle's parameters from a JSON file.

    The JSON file is expected to be a list of puzzle objects.

    Args:
        puzzle_number (int): The number of the puzzle to load.
        puzzles_file_path (str): The path to the JSON file containing puzzles.

    Returns:
        Dict: A dictionary containing the parameters for the specified puzzle.

    Raises:
        FileNotFoundError: If the puzzles file cannot be found.
        ValueError: If the specified puzzle number is not found in the file.
    """
    try:
        with open(puzzles_file_path, 'r') as f:
            puzzles = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Puzzles file not found at: {puzzles_file_path}")

    for puzzle in puzzles:
        if puzzle.get("puzzle_number") == puzzle_number:
            return puzzle

    raise ValueError(f"Puzzle number {puzzle_number} not found in {puzzles_file_path}")


def load_profile(profile_name: str, profiles_dir_path: str) -> Dict:
    """
    Loads a specific solver profile's parameters from an INI file.

    The INI file is expected to have a [Solver] section.

    Args:
        profile_name (str): The name of the profile to load (e.g., 'verify').
        profiles_dir_path (str): The path to the directory containing profile INI files.

    Returns:
        Dict: A dictionary of the settings from the [Solver] section.

    Raises:
        FileNotFoundError: If the profile file cannot be found.
        ValueError: If the [Solver] section is missing from the profile.
    """
    profile_file_path = os.path.join(profiles_dir_path, f"{profile_name}.ini")
    config = configparser.ConfigParser()

    if not config.read(profile_file_path):
        raise FileNotFoundError(f"Profile file not found at: {profile_file_path}")

    if 'Solver' not in config:
        raise ValueError(f"[Solver] section not found in profile: {profile_file_path}")

    return dict(config['Solver'])
