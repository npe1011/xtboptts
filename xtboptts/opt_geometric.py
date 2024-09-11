import os
import shutil
import csv
import math
import re
import tempfile
import logging.config
from pathlib import Path
from typing import List, Tuple

import numpy as np
import geometric
from geometric.errors import GeomOptNotConvergedError

import config
from xtboptts import xtb, xyzutils


class MyXTBEngine(geometric.engine.Engine):

    def __init__(self, atoms: np.ndarray, coordinates: np.ndarray, xtb_params: xtb.XTBParams, stop_file: Path):
        self.atoms = atoms
        molecule = geometric.molecule.Molecule()
        molecule.xyzs = np.array([coordinates], dtype=float)
        molecule.elem = list(self.atoms)
        super(MyXTBEngine, self).__init__(molecule)
        self.xtb_params = xtb_params
        self.stop_file = stop_file

    def calc_new(self, coords, dirname):
        if self.stop_file.exists():
            raise xtb.XTBTerminationError('Stopped by user')
        workdir = Path(dirname).absolute()
        energy, gradient = xtb.xtb_energy_and_gradient(atoms=self.atoms,
                                                       coordinates=coords.reshape((-1, 3)) / config.ANG_TO_BOHR,
                                                       xtb_params=self.xtb_params,
                                                       workdir=workdir)
        return {'energy': energy, 'gradient': gradient.ravel()}

    def calc_wq_new(self, coords, dirname):
        raise NotImplemented('MyXTBEngine does not support work que parallelization.')


class GeometricOptHistory:
    def __init__(self, log_file: Path, xyz_file: Path):
        self.energy_list, \
        self.energy_change_list, \
        self.gradient_rms_list, \
        self.gradient_max_list, \
        self.displacement_rms_list, \
        self.displacement_max_list, \
        self.trust_list = GeometricOptHistory._read_log(log_file)

        self.atoms, self.coordinates_list, title_list = xyzutils.read_sequential_xyz_file(xyz_file)

        # check xyz and log files are consistent
        if not len(self.coordinates_list) == len(self.energy_list):
            raise RuntimeError('Step numbers of Geometric log and xyz files do not match.')
        try:
            for energy, title in zip(self.energy_list, title_list):
                assert math.isclose(energy, float(title.split('Energy')[1].strip()), abs_tol=1.0e-9)
        except AssertionError:
            raise RuntimeError('Energy values in Geometric log and xyz files do not match.')

    def extend(self, log_file: Path, xyz_file: Path) -> None:
        energy_list, \
        energy_change_list, \
        gradient_rms_list, \
        gradient_max_list, \
        displacement_rms_list, \
        displacement_max_list, \
        trust_list = GeometricOptHistory._read_log(log_file)

        # for extend, current final step should be same as the next step 0
        try:
            assert (math.isclose(self.energy_list[-1], energy_list[0], abs_tol=1.0e-9))
            assert (math.isclose(self.gradient_rms_list[-1], gradient_rms_list[0], rel_tol=1.0e-5))
            assert (math.isclose(self.gradient_max_list[-1], gradient_max_list[0], rel_tol=1.0e-5))
        except AssertionError:
            raise RuntimeError('Step 0 energy/gradient do not match the previous final results.')

        self.energy_list.extend(energy_list[1:])
        self.energy_change_list.extend(energy_change_list[1:])
        self.gradient_rms_list.extend(gradient_rms_list[1:])
        self.gradient_max_list.extend(gradient_max_list[1:])
        self.displacement_rms_list.extend(displacement_rms_list[1:])
        self.displacement_max_list.extend(displacement_max_list[1:])
        self.trust_list.extend(trust_list[1:])

        atoms, coordinates_list, title_list = xyzutils.read_sequential_xyz_file(xyz_file)
        # check xyz and log files are consistent
        if not len(coordinates_list) == len(energy_list):
            raise RuntimeError('Step numbers of Geometric log and xyz files do not match.')
        try:
            for energy, title in zip(energy_list, title_list):
                assert math.isclose(energy, float(title.split('Energy')[1].strip()), abs_tol=1.0e-9)
        except AssertionError:
            raise RuntimeError('Energy values in Geometric log and xyz files do not match.')

        self.coordinates_list = np.concatenate([self.coordinates_list, coordinates_list[1:]])

    def write_csv(self, csv_file: Path):
        with csv_file.open(mode='w', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'energy', 'energy change', 'gradient rms', 'gradient max',
                             'displacement rms', 'displacement max', 'trust'])
            for i in range(len(self.energy_list)):
                writer.writerow([i, self.energy_list[i], self.energy_change_list[i],
                                 self.gradient_rms_list[i], self.gradient_max_list[i],
                                 self.displacement_rms_list[i], self.displacement_max_list[i], self.trust_list[i]])

    def write_trajectory(self, xyz_file: Path):
        title_list = ['Step. {0:d} {1:.8f}'.format(i, energy) for (i, energy) in enumerate(self.energy_list)]
        xyzutils.save_sequential_xyz_file(xyz_file, self.atoms, self.coordinates_list, title_list)

    def write_current_xyz(self, xyz_file: Path):
        title = 'Step. {0:d} {1:.8f}'.format(self.num_step, self.current_energy)
        xyzutils.save_xyz_file(xyz_file, self.atoms, self.current_coordinates, title=title)

    @property
    def num_step(self) -> int:
        return len(self.coordinates_list)

    @property
    def current_coordinates(self) -> np.ndarray:
        return self.coordinates_list[-1]

    @property
    def current_energy(self) -> float:
        return self.energy_list[-1]

    @staticmethod
    def _read_log(log_file: Path) -> Tuple[List[float],List[float],List[float],List[float],List[float],List[float],List[float]]:
        """
        read geometric log file and return optimization patemers
        :param log_file: geometric log file
        """

        energy_list = []
        energy_change_list = []
        gradient_rms_list = []
        gradient_max_list = []
        displacement_rms_list = []
        displacement_max_list = []
        trust_list = []

        with log_file.open(mode='r') as f:
            data = f.readlines()

        next_step = 0
        for line in data:
            # remove color code
            color_pattern = r'\033\[[0-9;]*m'
            line = re.sub(color_pattern, '', line)
            # read Step   @@@: line
            if line.startswith('Step') and line.lstrip('Step').lstrip().startswith(str(next_step)):

                # for Step 0
                if next_step == 0:
                    # Step    0 : Gradient = 3.716e-03/7.707e-03 (rms/max) Energy = -12.2726375598
                    energy = line.split('Energy')[1].strip().split()[1]
                    gradient_rms, gradient_max = line.split('Gradient')[1].strip().split()[1].split('/')
                    energy_list.append(float(energy))
                    gradient_rms_list.append(float(gradient_rms))
                    gradient_max_list.append(float(gradient_max))
                    # None for unavailable parameters
                    energy_change_list.append(None)
                    displacement_rms_list.append(None)
                    displacement_max_list.append(None)
                    trust_list.append(None)

                # for other Steps
                else:
                    # Step    1 : Displace = 1.016e-02/2.050e-02 (rms/max) Trust = 1.000e-02 (=) Grad = 2.508e-03/5.516e-03 (rms/max) E (change) = -12.2723526693 (+2.849e-04) Quality = 1.000
                    energy = line.split('E (change)')[1].strip().split()[1]
                    energy_change = line.split('E (change)')[1].strip().split()[2].lstrip('(').rstrip(')')
                    gradient_rms, gradient_max = line.split('Grad')[1].strip().split()[1].split('/')
                    displacement_rms, displacement_max = line.split('Displace')[1].strip().split()[1].split('/')
                    trust = line.split('Trust')[1].strip().split()[1]
                    energy_list.append(float(energy))
                    energy_change_list.append(float(energy_change))
                    gradient_rms_list.append(float(gradient_rms))
                    gradient_max_list.append(float(gradient_max))
                    displacement_rms_list.append(float(displacement_rms))
                    displacement_max_list.append(float(displacement_max))
                    trust_list.append(float(trust))

                next_step += 1

        return energy_list, energy_change_list, gradient_rms_list, gradient_max_list, \
               displacement_rms_list, displacement_max_list, trust_list


class GeometricResult:
    """
    self.status
    'converged'
    'not converged
    'fail'
    """
    def __init__(self, status, error):
        self.status: str = status
        self.error: Exception = error


def optimization(input_xyz_file: Path,
                 job_name: str,
                 xtb_params: xtb.XTBParams,
                 temperature: float,
                 ts: bool,
                 converge: str,
                 maxcycles: int,
                 hess_step: int,
                 final_hess: bool,
                 workdir: Path,
                 keep_log: int) -> GeometricResult:
    """
    Run geometric optimization
    :param input_xyz_file: init xyz file path
    :param job_name: string used for output file names
    :param xtb_params: xtb calculation setting parameters
    :param temperature: temperature for thermochemistry. < 0: xtb default value (298.15)
    :param ts: True for transition state optimization
    :param converge: convergence criteria
    :param maxcycles: max optimization cycle
    :param hess_step: <0: no hess, 0: init hess, >=1: hessian calculation at every N step
    :param final_hess: True for final Hessian calculation to get freq and thermochemistry
    :param workdir:: working directory (child directories are made under this). Output files are saved here.
    :param keep_log: keep log mode
    :return: GeometricResult (containing status and error)
    """

    # Set paths
    input_xyz_file = input_xyz_file.absolute()
    workdir = workdir.absolute()
    output_trajectory_file = workdir / (job_name + config.OUTPUT_TRAJECTORY_FILE_SUFFIX)
    output_csv_file = workdir / (job_name + config.OUTPUT_CSV_FILE_SUFFIX)
    output_final_xyz_file = workdir / (job_name + config.OUTPUT_FINAL_XYZ_FILE_SUFFIX)
    output_final_hess_file = workdir / (job_name + config.OUTPUT_HESS_FILE_SUFFIX)
    output_final_thermal_file = workdir / (job_name + config.OUTPUT_THERMAL_FILE_SUFFIX)
    stop_file = input_xyz_file.parent / (job_name + config.STOP_FILE_SUFFIX)

    if stop_file.exists():
        stop_file.unlink()

    # initial check
    if not input_xyz_file.exists():
        raise FileNotFoundError(str(input_xyz_file) + ' not found.')
    if ts and hess_step < 0:
        raise RuntimeError('For TS optimization, initial Hessian calculation is required.')

    if not workdir.exists():
        workdir.mkdir(parents=True)

    init_dir = os.getcwd()

    geometric_dir = Path(tempfile.mkdtemp(dir=workdir, prefix=job_name+'geometric_')).absolute()
    shutil.copy(input_xyz_file, geometric_dir / config.INIT_XYZ_FILE)
    calc_result = 'fail'
    calc_error = None

    # check and translate converge
    if converge is None:
        converge = 'GAU'
    elif converge.lower() == 'normal':
        converge = 'GAU'
    elif converge.lower() == 'tight':
        converge = 'GAU_TIGHT'
    elif converge.lower() in ['vtight', 'very_tight', 'verytight']:
        converge = 'GAU_VERYTIGHT'
    elif converge.lower() in ['loose']:
        converge = 'GAU_LOOSE'
    else:
        converge = str(converge).upper()

    try:
        os.chdir(geometric_dir)
        intermediate_files = [Path(config.INIT_XYZ_FILE).absolute()]

        # case: hess_step < 0 or hess_step = 0 (one geometric optimization)
        if hess_step <= 0:
            prefix = 'geometric'
            atoms, init_coordinates = xyzutils.read_single_xyz_file(config.INIT_XYZ_FILE)
            # init Hessian if hess_step=0
            if hess_step == 0:
                hess, _ = xtb.xtb_hessian(atoms, init_coordinates, xtb_params, geometric_dir)
                hess_file = 'init_hess.txt'
                np.savetxt(hess_file, hess)
                intermediate_files.append(Path(hess_file).absolute())
                hessian = 'file:{:}'.format(hess_file)
            else:
                hessian = 'never'

            # main calculation
            try:
                logini = str(Path(__file__).parent / 'mylog.ini')
                engine = MyXTBEngine(atoms=atoms,
                                     coordinates=init_coordinates,
                                     xtb_params=xtb_params,
                                     stop_file=stop_file)
                geometric.optimize.run_optimizer(customengine=engine,
                                                 input=prefix,
                                                 transition=ts,
                                                 convergence_set=converge,
                                                 maxiter=maxcycles-1,
                                                 hessian=hessian,
                                                 logIni=logini)
            except GeomOptNotConvergedError:
                calc_result = 'not converged'

            except RuntimeError as e:
                calc_result = 'fail'
                calc_error = e

            else:
                calc_result = 'converged'

            opt_history = GeometricOptHistory(Path(prefix + '.log'), Path(prefix + '_optim.xyz'))
            intermediate_files.append(Path(prefix + '.log').absolute())
            intermediate_files.append(Path(prefix + '_optim.xyz').absolute())
            intermediate_files.append(Path(prefix + '.vdata_first').absolute())
            intermediate_files.append(Path(prefix + '.tmp').absolute())

        # case: hess_step > 0 (iterative geometric optimization)
        else:
            num_calc_hessian = 0
            total_step = 0
            remain_step = maxcycles
            atoms, current_coordinates = xyzutils.read_single_xyz_file(config.INIT_XYZ_FILE)
            opt_history = None

            while remain_step > 0:
                if total_step % hess_step == 0:
                    hess, _ = xtb.xtb_hessian(atoms, current_coordinates, xtb_params, geometric_dir)
                    hess_file = 'hess{:d}.txt'.format(num_calc_hessian)
                    np.savetxt(hess_file, hess)
                    intermediate_files.append(Path(hess_file).absolute())
                    hessian = 'file:{:}'.format(hess_file)
                    num_calc_hessian += 1
                else:
                    hessian = 'never'
                next_cycles = min(hess_step-1, remain_step)
                try:
                    prefix = 'geometric' + str(num_calc_hessian)
                    engine = MyXTBEngine(atoms=atoms,
                                         coordinates=current_coordinates,
                                         xtb_params=xtb_params,
                                         stop_file=stop_file)
                    logini = str(Path(__file__).parent / 'mylog.ini')
                    geometric.optimize.run_optimizer(customengine=engine,
                                                     input=prefix,
                                                     transition=ts,
                                                     convergence_set=converge,
                                                     maxiter=next_cycles,
                                                     hessian=hessian,
                                                     logIni=logini)
                except GeomOptNotConvergedError:
                    calc_result = 'not converged'

                except RuntimeError as e:
                    calc_result = 'fail'
                    calc_error = e

                else:
                    calc_result = 'converged'

                # read logs and concat results
                if opt_history is None:
                    opt_history = GeometricOptHistory(Path(prefix + '.log'), Path(prefix + '_optim.xyz'))
                else:
                    opt_history.extend(Path(prefix + '.log'), Path(prefix + '_optim.xyz'))

                # intermediate files
                intermediate_files.append(Path(prefix + '.log').absolute())
                intermediate_files.append(Path(prefix + '_optim.xyz').absolute())
                intermediate_files.append(Path(prefix + '.vdata_first').absolute())
                intermediate_files.append(Path(prefix + '.tmp').absolute())

                # update the current structure with the final structure
                current_coordinates = opt_history.current_coordinates

                remain_step -= (next_cycles + 1)
                if calc_result in ['fail', 'converged']:
                    break

        # Here, results are summarized in opt_history object, calc_result, calc_error
        # Output result
        opt_history.write_trajectory(output_trajectory_file)
        opt_history.write_csv(output_csv_file)
        opt_history.write_current_xyz(output_final_xyz_file)

        # Final Hessian/thermo chem calculations
        if final_hess and calc_result == 'converged':
            hess, thermal_data = xtb.xtb_hessian(atoms, opt_history.current_coordinates,
                                                 xtb_params, geometric_dir, temperature=temperature)
            np.savetxt(str(output_final_hess_file), hess)
            thermal_data.save(output_final_thermal_file)

    # common post-process
    finally:
        os.chdir(init_dir)
        if (keep_log == 0) or (keep_log == 1 and calc_result != 'fail'):
            # To release final log file from the logging
            temp_log_ini = Path(__file__).parent / 'mylog.ini'
            _, garbage_log_file = tempfile.mkstemp()
            logging.config.fileConfig(temp_log_ini, defaults={'logfilename': garbage_log_file},
                                      disable_existing_loggers=False)
            for file in intermediate_files:
                try:
                    if file.is_file():
                        file.unlink()
                    else:
                        shutil.rmtree(file)
                except:
                    pass

            try:
                shutil.rmtree(geometric_dir)
            except:
                pass

    return GeometricResult(status=calc_result, error=calc_error)
