import os
import shutil
import subprocess as sp
from decimal import Decimal
from pathlib import Path
import tempfile
from typing import List, Union, Optional

import numpy as np

import config
from xtboptts import utils, xyzutils

# Global variable to check whether setenv run or not
CHECK_SETENV = False


class XTBTerminationError(Exception):
    pass


class XTBParams:
    def __init__(self,
                 method: str = 'gfn2',
                 charge: int = 0,
                 uhf: int = 0,
                 solvation: Optional[str] = None,
                 solvent: Optional[str] = None):
        # check parameters
        try:
            method = method.lower()
            assert method in ['gfn0', 'gfn1', 'gfn2', 'gfnff']
            assert uhf >= 0
            if solvation is not None:
                solvation = solvation.lower()
                assert solvation in ['alpb', 'gbsa']
                assert solvent is not None
                assert solvent.lower() != 'none'
                assert len(solvent) != ''
        except AssertionError as e:
            raise ValueError('Given XTB parameters are not valid. ' + str(e.args))

        self.method = method
        self.charge = charge
        self.uhf = uhf
        self.solvation = solvation
        self.solvent = solvent

    @property
    def args(self) -> List[str]:
        _args = ['--' + self.method, '--chrg', str(self.charge), '--uhf', str(self.uhf)]
        if self.solvation is not None:
            _args.extend(['--' + self.solvation, self.solvent])
        return _args


def setenv(num_threads: int = 1, memory_per_thread: Optional[str] = None) -> None:
    assert int(num_threads) > 0
    num_threads = str(int(num_threads))

    if memory_per_thread is None:
        memory_per_thread = '500M'
    memory_per_thread = str(memory_per_thread)
    if memory_per_thread.lower().endswith('b'):
        memory_per_thread = memory_per_thread[:-1]

    os.environ['XTBPATH'] = str(config.XTB_PARAM_DIR)
    os.environ['OMP_NUM_THREADS'] = num_threads + ',1'
    os.environ['OMP_STACKSIZE'] = memory_per_thread
    os.environ['MKL_NUM_THREADS'] = num_threads

    current_path = os.environ.get('PATH', '')
    xtb_bin_dir = str(Path(config.XTB_BIN).parent)
    if xtb_bin_dir not in current_path.split(os.pathsep):
        os.environ['PATH'] = current_path + os.pathsep + xtb_bin_dir

    current_path = os.environ.get('PATH', '')
    if config.XTB_OTHER_LIB_DIR is not None:
        if config.XTB_OTHER_LIB_DIR not in os.environ['PATH']:
            current_path + os.pathsep + config.XTB_OTHER_LIB_DIR

    global CHECK_SETENV
    CHECK_SETENV = True


def xtb_energy_and_gradient(atoms: np.ndarray, coordinates: np.ndarray, xtb_params: XTBParams, workdir: Path):
    # initial check
    if not CHECK_SETENV:
        setenv(num_threads=1, memory_per_thread='500M')

    workdir = Path(tempfile.mkdtemp(dir=workdir, prefix='xtb_energy_gradient_')).absolute()
    prevdir = os.getcwd()

    try:
        os.chdir(str(workdir))
        xyzutils.save_xyz_file(config.INIT_XYZ_FILE, atoms, coordinates, 'for energy and gradient')
        command = [config.XTB_BIN, config.INIT_XYZ_FILE, '--grad']
        command.extend(xtb_params.args)
        with open(config.XTB_LOG_FILE, 'w', encoding='utf-8') as f:
            proc = utils.popen_bg(command, universal_newlines=True, encoding='utf-8', stdout=f, stderr=sp.STDOUT)
            proc.wait()
        with open(config.XTB_LOG_FILE, 'r', encoding='utf-8') as f:
            if 'normal termination of xtb' not in f.read():
                raise RuntimeError('xtb optimization failed in {:}'.format(workdir))
        energy = _read_xtb_energy(config.XTB_ENERGY_FILE)
        gradient = _read_xtb_gradient(config.XTB_GRADIENT_FILE, len(atoms))

    except:
        raise

    else:
        return energy, gradient

    finally:
        os.chdir(prevdir)
        try:
            shutil.rmtree(workdir)
        except:
            pass


def xtb_hessian(atoms: np.ndarray, coordinates: np.ndarray, xtb_params: XTBParams, workdir: Path, temperature=None):
    # initial check
    if not CHECK_SETENV:
        setenv(num_threads=1, memory_per_thread='500M')

    workdir = Path(tempfile.mkdtemp(dir=workdir, prefix='xtb_energy_gradient_')).absolute()
    prevdir = os.getcwd()

    try:
        os.chdir(str(workdir))
        xyzutils.save_xyz_file(config.INIT_XYZ_FILE, atoms, coordinates, 'for hessian')
        command = [config.XTB_BIN, config.INIT_XYZ_FILE, '--hess']
        if temperature is not None:
            with open(config.XTB_INPUT_FILE, 'w', encoding='utf-8') as f:
                f.write('$thermo\n')
                f.write('  temp={:}\n'.format(temperature))
                f.write('$end\n')
            command.extend(['--input', config.XTB_INPUT_FILE])
        command.extend(xtb_params.args)
        with open(config.XTB_LOG_FILE, 'w', encoding='utf-8') as f:
            proc = utils.popen_bg(command, universal_newlines=True, encoding='utf-8', stdout=f, stderr=sp.STDOUT)
            proc.wait()
        with open(config.XTB_LOG_FILE, 'r', encoding='utf-8') as f:
            if 'normal termination of xtb' not in f.read():
                raise RuntimeError('xtb optimization failed in {:}'.format(workdir))
        hess = _read_xtb_hessian(config.XTB_HESSIAN_FILE, len(atoms))
        thermal_data = ThermalData(config.XTB_LOG_FILE)

    except:
        raise

    else:
        return hess, thermal_data

    finally:
        os.chdir(prevdir)
        try:
            shutil.rmtree(workdir)
        except:
            pass


def _read_xtb_energy(energy_file) -> float:
    energy_file = Path(energy_file)
    with energy_file.open(mode='r') as f:
        energy_data = f.readlines()
    assert energy_data[0].strip().startswith('$energy')
    return float(energy_data[1].strip().split()[1])


def _read_xtb_gradient(gradient_file: Union[str, Path], num_atom: int) -> np.ndarray:
    gradient_file = Path(gradient_file)
    with gradient_file.open(mode='r') as f:
        gradient_data = f.readlines()
    assert gradient_data[0].strip().startswith('$grad')
    gradient = []
    for line in gradient_data[2+num_atom:2+2*num_atom]:
        gradient.append(line.strip().split())

    return np.array(gradient, dtype=float)


def _read_xtb_hessian(hessian_file: Union[str, Path], num_atom: int) -> np.ndarray:
    hessian_file = Path(hessian_file)
    with hessian_file.open(mode='r') as f:
        hessian_data = f.readlines()
    assert hessian_data[0].strip().startswith('$hessian')

    hess_1d = []
    for line in hessian_data[1:]:
        if line.strip() == '':
            continue
        if line.strip().startswith('$end'):
            break
        hess_1d.extend(line.strip().split())

    assert len(hess_1d) == (num_atom * 3) ** 2

    return np.array(hess_1d, dtype=float).reshape((3*num_atom, 3*num_atom))


class ThermalData:
    _label = '# OUTPUT FROM XTB SADDLE PROGRAM #'

    def __init__(self, file: Union[str, Path]):
        self.ee: Optional[Decimal] = None
        self.zpe: Optional[Decimal] = None
        self.g: Optional[Decimal] = None
        self.g_corr: Optional[Decimal] = None
        self.h: Optional[Decimal] = None
        self.h_corr: Optional[Decimal] = None

        with open(file, mode='r', encoding='utf-8') as f:
            data = f.readlines()

        if data[0].strip() == self._label:
            self._read_saved_data(data)
        else:
            self._read_xtblog_data(data)

    def _read_saved_data(self, data: List[str]) -> None:
        for line in data:
            if line.startswith('EE ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.ee = None
                else:
                    self.ee = Decimal(value)
            if line.startswith('ZPE ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.zpe = None
                else:
                    self.zpe = Decimal(value)
            if line.startswith('G ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.g = None
                else:
                    self.g = Decimal(value)
            if line.startswith('G-EE ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.g_corr = None
                else:
                    self.g_corr = Decimal(value)
            if line.startswith('H ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.h = None
                else:
                    self.h = Decimal(value)
            if line.startswith('H-EE ='):
                value = line.split('=')[1].strip()
                if value.lower() == 'none':
                    self.h_corr = None
                else:
                    self.h_corr = Decimal(value)

    def _read_xtblog_data(self, data: List[str]) -> None:
        start_line = 0
        end_line = -1
        for (i, line) in enumerate(data):
            if '::' in line and 'THERMODYNAMIC' in line:
                start_line = i
            if line.strip().startswith(':::::::::::::::::::') and i > start_line + 3 > 3:
                end_line = i
                break

        for line in data[start_line:end_line]:
            if 'total free energy ' in line:
                # :: total free energy         -47.035962096909 Eh   ::
                self.g = Decimal(line.split('Eh')[0].strip().split()[-1])
            if 'total energy' in line:
                self.ee = Decimal(line.split('Eh')[0].strip().split()[-1])
            if 'zero point energy' in line:
                self.zpe = Decimal(line.split('Eh')[0].strip().split()[-1])

        # read enthalpy
        for line in data[end_line:]:
            if 'TOTAL ENTHALPY' in line:
                # | TOTAL ENTHALPY            -47.016062830788 Eh   |
                self.h = Decimal(line.split('Eh')[0].strip().split()[-1])

        # correction terms
        if (self.g is not None) and (self.ee is not None):
            self.g_corr = self.g - self.ee
        if (self.h is not None) and (self.ee is not None):
            self.h_corr = self.h - self.ee

    def save(self, file: Union[str, Path]) -> None:
        with open(file, mode='w', encoding='utf-8', newline='\n') as f:
            f.write(self._label + '\n')
            f.write('EE = {}\n'.format(self.ee))
            f.write('ZPE = {}\n'.format(self.zpe))
            f.write('G = {}\n'.format(self.g))
            f.write('G-EE = {}\n'.format(self.g_corr))
            f.write('H = {}\n'.format(self.h))
            f.write('H-EE = {}\n'.format(self.h_corr))
