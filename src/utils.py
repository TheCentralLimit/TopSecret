"""
Utility functions.
"""

from itertools import tee

from os import makedirs
from os.path import isdir


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
