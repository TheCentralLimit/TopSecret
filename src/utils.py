"""
Utility functions.
"""

from os import makedirs
from os.path import isdir


def make_sure_path_exists(path):
    """
    Creates the supplied *path* if it does not exist.
    Raises *OSError* if the *path* cannot be created.

    **Parameters**
    path : str
        Path to create.

    **Returns**
    None
    """
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise
