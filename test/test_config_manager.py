import json
import pytest

import config_manager as cm


@pytest.fixture
def puzzles_json_path(tmp_path):
    """Creates a temporary puzzles.json file for testing."""
    puzzles_data = [
        {
            "puzzle_number": 5,
            "public_key": "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
            "range_start": "0x2",
            "range_end": "0x3",
        },
        {
            "puzzle_number": 8,
            "public_key": "03c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
            "range_start": "0x80",
            "range_end": "0xff",
        }
    ]
    file_path = tmp_path / "puzzles.json"
    with open(file_path, 'w') as f:
        json.dump(puzzles_data, f)
    return str(file_path)


@pytest.fixture
def profiles_dir(tmp_path):
    """Creates a temporary directory with profile .ini files."""
    profiles_path = tmp_path / "profiles"
    profiles_path.mkdir()

    # Valid profile
    verify_ini = profiles_path / "verify.ini"
    verify_ini.write_text(
        "[Solver]\n"
        "num_walkers = 16\n"
        "distinguished_point_threshold = 10\n"
    )

    # Profile missing [Solver] section
    invalid_ini = profiles_path / "invalid.ini"
    invalid_ini.write_text(
        "[OtherSection]\n"
        "key = value\n"
    )

    return str(profiles_path)


class TestLoadPuzzleDefinition:
    def test_load_puzzle_success(self, puzzles_json_path):
        """Tests successfully loading a puzzle definition."""
        puzzle_data = cm.load_puzzle_definition(8, puzzles_json_path)
        assert puzzle_data["puzzle_number"] == 8
        assert puzzle_data["range_start"] == "0x80"

    def test_load_puzzle_not_found(self, puzzles_json_path):
        """Tests that a ValueError is raised for a non-existent puzzle."""
        with pytest.raises(ValueError, match="Puzzle number 99 not found"):
            cm.load_puzzle_definition(99, puzzles_json_path)

    def test_load_puzzle_file_not_found(self):
        """Tests that a FileNotFoundError is raised for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            cm.load_puzzle_definition(5, "nonexistent/puzzles.json")


class TestLoadProfile:
    def test_load_profile_success(self, profiles_dir):
        """Tests successfully loading a solver profile."""
        profile_data = cm.load_profile("verify", profiles_dir)
        assert isinstance(profile_data, dict)
        assert profile_data["num_walkers"] == "16"
        assert profile_data["distinguished_point_threshold"] == "10"

    def test_load_profile_file_not_found(self, profiles_dir):
        """Tests that a FileNotFoundError is raised for a non-existent profile."""
        with pytest.raises(FileNotFoundError, match="Profile file not found"):
            cm.load_profile("nonexistent", profiles_dir)

    def test_load_profile_missing_section(self, profiles_dir):
        """Tests that a ValueError is raised if [Solver] section is missing."""
        with pytest.raises(ValueError, match="\\[Solver\\] section not found"):
            cm.load_profile("invalid", profiles_dir)
