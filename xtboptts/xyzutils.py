from typing import List, Tuple, Union
from pathlib import Path

import numpy as np

from config import FLOAT, XYZ_FORMAT


def save_xyz_file(xyz_file: Union[str, Path], atoms: np.ndarray, coordinates: np.ndarray, title: str = '') -> None:
    xyz_file = Path(xyz_file)
    data = [
        str(len(atoms)) + '\n',
        title.rstrip() + '\n'
    ]
    for i in range(len(atoms)):
        data.append(XYZ_FORMAT.format(atoms[i], coordinates[i][0], coordinates[i][1], coordinates[i][2]))
    with xyz_file.open(mode='w', encoding='utf-8', newline='\n') as f:
        f.writelines(data)


def read_single_xyz_file(xyz_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: tuple(atoms: np.ndarray, coordinates: np.ndarray)
    """
    xyz_file = Path(xyz_file)
    with xyz_file.open(mode='r') as f:
        data = f.readlines()
    atoms = []
    coordinates = []
    num_atoms = int(data[0].strip())
    for line in data[2:2+num_atoms]:
        temp = line.strip().split()
        atom = temp[0].capitalize()
        x = float(temp[1])
        y = float(temp[2])
        z = float(temp[3])
        atoms.append(atom)
        coordinates.append([x, y, z])
    return np.array(atoms), np.array(coordinates, dtype=FLOAT)


def read_sequential_xyz_file(file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    return: tuple(atoms, coordinates_list, title_list)
    atoms: np.ndarray
    coordinates_list np.ndarray 3d (num_conf x num_atoms x 3)
    title_list list of str
    """

    with Path(file).open() as f:
        data = f.readlines()

    atoms = []
    coordinates_list = []
    title_list = []

    num_atoms = int(data[0].strip())

    # check if validity
    start_lines = []
    for i, line in enumerate(data):
        if i % (num_atoms + 2) == 0:
            if line.strip() == str(num_atoms):
                start_lines.append(i)
            else:
                if line.strip() == '':
                    break
                else:
                    raise ValueError(str(file) + 'is not a valid xyz file.')

    # Read each coordinates and title
    for start_line in start_lines:
        title_list.append(data[start_line+1].strip())
        coordinates = []
        for line in data[start_line+2:start_line+num_atoms+2]:
            temp = line.strip().split()
            x = float(temp[1])
            y = float(temp[2])
            z = float(temp[3])
            coordinates.append([x, y, z])
        coordinates_list.append(np.array(coordinates, dtype=FLOAT))

    # Read atoms
    for line in data[2:2+num_atoms]:
        atoms.append(line.strip().split()[0].capitalize())

    return np.array(atoms), np.array(coordinates_list, dtype=FLOAT), title_list


def save_sequential_xyz_file(xyz_file: Union[str, Path],
                             atoms: np.ndarray,
                             coordinates_list: Union[np.ndarray, List[np.ndarray]],
                             title_list: List[str]) -> None:
    """
    atoms: np.ndarray
    coordinates_list np.ndarray 3d (num_conf x num_atoms x 3)
    title_list: str list
    """
    data = []
    num_atoms = len(atoms)
    for (coordinates, title) in zip(coordinates_list, title_list):
        data.append('{:5d}\n'.format(num_atoms))
        data.append(title.rstrip() + '\n')
        for i in range(len(atoms)):
            data.append(XYZ_FORMAT.format(atoms[i], coordinates[i][0], coordinates[i][1], coordinates[i][2]))

    with Path(xyz_file).open(mode='w', encoding='utf-8', newline='\n') as f:
        f.writelines(data)


def get_xyz_string(atoms: Union[List[str], np.ndarray], coordinates: np.ndarray) -> str:
    data = []
    for i in range(len(atoms)):
        data.append(XYZ_FORMAT.format(atoms[i], coordinates[i][0], coordinates[i][1], coordinates[i][2]))
    return ''.join(data)
