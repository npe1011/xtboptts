from pathlib import Path
import subprocess as sp
import tempfile
from typing import Union

import numpy as np

import config
from xtboptts import xyzutils, utils


def view_xyz_file(xyz_file: Union[str, Path]) -> None:
    sp.Popen([config.VIEWER_PATH, str(xyz_file)])


@utils.async_func
def view_xyz_structure(atoms: np.ndarray, coordinates: np.ndarray, title: str = ''):
    _, file = tempfile.mkstemp(suffix='.xyz', text=True)
    xyzutils.save_xyz_file(file, atoms, coordinates, title)
    sp.run([config.VIEWER_PATH, str(file)])
    try:
        Path(file).unlink()
    except:
        pass
