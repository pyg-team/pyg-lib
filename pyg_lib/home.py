import os
import os.path as osp
from typing import Optional

ENV_PYG_LIB_HOME = 'PYG_LIB_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'pyg_lib')

_home_dir: Optional[str] = None


def get_home_dir() -> str:
    r"""Gets the cache directory used for storing all :obj:`pyg-lib` data.

    If :meth:`set_home_dir` is not called, the path is given by the environment
    variable :obj:`$PYG_LIB_HOME` which defaults to :obj:`"~/.cache/pyg_lib"`.

    Returns:
        (str): The cache directory.
    """
    if _home_dir is not None:
        return _home_dir

    home_dir = os.getenv(ENV_PYG_LIB_HOME, DEFAULT_CACHE_DIR)
    home_dir = osp.expanduser(home_dir)
    return home_dir


def set_home_dir(path: str):
    r"""Sets the cache directory used for storing all :obj:`pyg-lib` data.

    Args:
        path (str): The path to a local folder.
    """
    global _home_dir
    _home_dir = path
