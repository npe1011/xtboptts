from typing import Any
from pathlib import Path

import numpy as np
from geometric.molecule import PeriodicTable

import config
from xtboptts import xyzutils


class XTBFreqResult:
    def __init__(self, structure_xyz_file: Path, hessian_file: Path):
        self.atoms, self.coordinates = xyzutils.read_single_xyz_file(structure_xyz_file)
        self.hess = np.loadtxt(hessian_file, dtype=float)

        # calculate mass-weighted Hessian
        weight_vec = []
        for atom in self.atoms:
            mass = PeriodicTable[atom.capitalize()]
            weight_vec.extend([1.0 / np.sqrt(mass), 1.0 / np.sqrt(mass), 1.0 / np.sqrt(mass)])
        weight_vec = np.array(weight_vec, dtype=float)
        self.mw_hess = np.copy(self.hess)
        self.mw_hess *= weight_vec
        self.mw_hess *= weight_vec.reshape((-1, 1))

        # check this molecule is linear or not
        self.linear = self._check_linear()
        if self.linear:
            num_omit = 5
        else:
            num_omit = 6

        # calculate eigen values in cm-1 and eigen vectors
        self.frequencies = []
        self.eigen_vectors = []
        eigen_values, eigen_matrix = np.linalg.eigh(self.mw_hess)
        # omit num_omit eigen values with small abs value
        omit_indices = np.argsort(np.abs(eigen_values))[0:num_omit]
        # list up
        eigen_sorted_indices = np.argsort(eigen_values)
        for i in eigen_sorted_indices:
            if i in omit_indices:
                continue
            if eigen_values[i] < 0.0:
                freq = -1.0 * _convert_eigen_to_inverse_cm(-eigen_values[i])
            else:
                freq = 1.0 * _convert_eigen_to_inverse_cm(eigen_values[i])
            self.frequencies.append(freq)
            self.eigen_vectors.append(eigen_matrix[:, i])

    def _check_linear(self) -> bool:
        num_atoms = len(self.atoms)
        if num_atoms <= 2:
            return True
        ref_vec = self.coordinates[1, :] - self.coordinates[0, :]
        for i in range(2, num_atoms):
            vec = self.coordinates[i, :] - self.coordinates[0, :]
            if not np.isclose(abs(np.dot(ref_vec, vec)) - (np.linalg.norm(ref_vec) * np.linalg.norm(vec)), 0.0):
                return False
        return True

    def save_freq_xyz(self, file: Path, index: int, step=20, max_shift=0.5):
        """
        save a xyz file for visualization of normal mode.
        """

        # weight: move move_matrix * weight in each step
        num_atoms = len(self.atoms)
        move_matrix = self.eigen_vectors[index].reshape((num_atoms, 3))
        max_vector_size = np.sqrt(np.max(np.sum(move_matrix * move_matrix, axis=1)))
        weight = max_shift / (max_vector_size * step)

        with open(file, 'w') as f:
            forward_coordinates = [self.coordinates + s * weight * move_matrix for s in range(1, step + 1)]
            backward_coordinates = [self.coordinates - s * weight * move_matrix for s in range(1, step + 1)]

            # forward
            f.write(str(num_atoms) + '\n')
            f.write('Initial Structure\n')
            f.write(_structure_string_from_atoms_and_coordinates(self.atoms, self.coordinates))
            for coordinate in forward_coordinates:
                f.write(str(num_atoms) + '\n')
                f.write('\n')
                f.write(_structure_string_from_atoms_and_coordinates(self.atoms, coordinate))
            for coordinate in reversed(forward_coordinates):
                f.write(str(num_atoms) + '\n')
                f.write('\n')
                f.write(_structure_string_from_atoms_and_coordinates(self.atoms, coordinate))

            # backward
            f.write(str(num_atoms) + '\n')
            f.write('Initial Structure\n')
            f.write(_structure_string_from_atoms_and_coordinates(self.atoms, self.coordinates))
            for coordinate in backward_coordinates:
                f.write(str(num_atoms) + '\n')
                f.write('\n')
                f.write(_structure_string_from_atoms_and_coordinates(self.atoms, coordinate))
            for coordinate in reversed(backward_coordinates):
                f.write(str(num_atoms) + '\n')
                f.write('\n')
                f.write(_structure_string_from_atoms_and_coordinates(self.atoms, coordinate))

            f.write(str(num_atoms) + '\n')
            f.write('Initial Structure\n')
            f.write(_structure_string_from_atoms_and_coordinates(self.atoms, self.coordinates))


def _convert_eigen_to_inverse_cm(eigen: Any) -> Any:
    """
    :param eigen:  in atomic unit
    :return: vib freq in cm-1 (float)
    """
    return np.sqrt(eigen) * 5140.486777894163


def _structure_string_from_atoms_and_coordinates(atoms: np.ndarray, coordinates: np.ndarray) -> str:
    structure_string = ''
    assert len(atoms) == coordinates.shape[0]
    for i in range(len(atoms)):
        structure_string += config.XYZ_FORMAT.format(atoms[i], coordinates[i, 0], coordinates[i, 1], coordinates[i, 2])
    return structure_string
