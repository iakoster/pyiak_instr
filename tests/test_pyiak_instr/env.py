import shutil
from pathlib import Path


__all__ = [
    "ROOT_DIR",
    "TEST_DATA_DIR",
    "get_local_test_data_dir",
    "remove_test_data_dir",
]


ROOT_DIR = Path(__file__).parent.parent.parent
TEST_DATA_DIR = ROOT_DIR / "test_data"


def get_local_test_data_dir(__name__: str) -> Path:
    """
    Get test data path for __name__ module.

    Parameters
    ----------
    __name__ : str
        the value of the __name__ variable in the corresponding module.

    Returns
    -------
    Path
        Test data placement path for this module
    """
    return TEST_DATA_DIR / __name__.split(".")[-1]


def remove_test_data_dir() -> None:
    """Iteratively delete the test data folder with all its contents."""
    shutil.rmtree(TEST_DATA_DIR)
