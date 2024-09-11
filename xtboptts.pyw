import sys
import os
import csv
from threading import Thread
from pathlib import Path
from typing import Any, List, Optional

import wx
from wx import xrc
import wx.grid
import wx.lib.newevent
import numpy as np
import matplotlib.pyplot as plt

APP_DIR = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(APP_DIR)
import config
from xtboptts import utils, view, xyzutils, convert, xtb, xtbfreq, opt_geometric

CalcEndEvent, EVT_CALC_END = wx.lib.newevent.NewEvent()

VERSION = '0.5.0'


class CalculationThread(Thread):

    def __init__(self,
                 input_xyz_file: Path,
                 job_name: str,
                 xtb_params: xtb.XTBParams,
                 temperature: float,
                 ts: bool,
                 converge: str,
                 maxcycles: int,
                 hess_step: int,
                 final_hess: bool,
                 keep_log: int,
                 num_threads: int,
                 memory_per_thread: str,
                 parent_window: Any):

        Thread.__init__(self)
        self.input_xyz_file: Path = input_xyz_file.absolute()
        self.workdir: Path = self.input_xyz_file.parent
        self.job_name: str = job_name
        self.stop_file: Path = self.workdir / (job_name + config.STOP_FILE_SUFFIX)
        self.xtb_params: xtb.XTBParams = xtb_params
        self.temperature: float = temperature
        self.ts: bool = ts
        self.converge: str = converge
        self.maxcycles: int = maxcycles
        self.hess_step:int = hess_step
        self.final_hess: bool = final_hess
        self.keep_log: int = keep_log
        self.num_threads: int = num_threads
        self.memory_per_thread: str = memory_per_thread
        self.parent_window = parent_window
        self.start()

    def run(self):
        if self.stop_file.exists():
            self.stop_file.unlink()

        try:
            xtb.setenv(self.num_threads, self.memory_per_thread)
            result: opt_geometric.GeometricResult = \
                opt_geometric.optimization(
                    input_xyz_file=self.input_xyz_file,
                    job_name=self.job_name,
                    xtb_params=self.xtb_params,
                    temperature=self.temperature,
                    ts=self.ts,
                    converge=self.converge,
                    maxcycles=self.maxcycles,
                    hess_step=self.hess_step,
                    final_hess=self.final_hess,
                    workdir=self.workdir,
                    keep_log=self.keep_log)
        except Exception as e:
            result = opt_geometric.GeometricResult(status='fail', error=e)
            wx.PostEvent(self.parent_window, CalcEndEvent(job_name=self.job_name,
                                                          job_dir=self.workdir,
                                                          job_result=result))
        else:
            wx.PostEvent(self.parent_window, CalcEndEvent(job_name=self.job_name,
                                                          job_dir=self.workdir,
                                                          job_result=result))

    def terminate(self):
        self.stop_file.touch(exist_ok=True)


class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, file_names):
        file = Path(file_names[0]).absolute()
        if file.suffix.lower() == '.csv':
            self.window.load_result_csv_file(file)
        else:
            self.window.load_input_structure_file(file)
        return True


class CSVViewFrame(wx.Frame):
    def __init__(self, parent: Any, title: str, data: List[List]):
        wx.Frame.__init__(self, parent, -1, title)
        self.data = data
        self.init_frame()
        self.show_data()

    def init_frame(self):
        panel = wx.Panel(self, wx.ID_ANY)
        layout = wx.BoxSizer(wx.VERTICAL)
        self.grid_data = wx.grid.Grid(panel, wx.ID_ANY)
        font = wx.Font(12, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid_data.SetFont(font)
        layout.Add(self.grid_data, 1, wx.EXPAND | wx.ALL, border=3)
        panel.SetSizerAndFit(layout)

        self.grid_data.Bind(wx.EVT_KEY_DOWN, self.on_key_down_grid)

    def show_data(self):
        num_rows = len(self.data)
        num_cols = len(self.data[0])
        self.grid_data.CreateGrid(num_rows-1, num_cols-1)
        # Label for rows (number)
        for i in range(num_rows-1):
            self.grid_data.SetRowLabelValue(i, self.data[i+1][0])
        for i in range(num_cols-1):
            self.grid_data.SetColLabelValue(i, self.data[0][i+1])
        # show data
        for r in range(1, num_rows):
            for c in range(1, num_cols):
                self.grid_data.SetCellValue(r-1, c-1, self.data[r][c])
        self.grid_data.AutoSize()

        width = min(self.grid_data.Size[0] + 100, 1200)
        height = min(self.grid_data.Size[1] + 100, 800)
        self.SetSize((width, height))

    def copy_data(self):

        # Not selected but focused on some cell
        if len(self.grid_data.GetSelectionBlockBottomRight()) == 0:
            data = str(self.grid_data.GetCellValue(self.grid_data.GetGridCursorRow(),
                                                   self.grid_data.GetGridCursorCol()))

        else:
            rows = self.grid_data.GetSelectionBlockBottomRight()[0][0] - \
                   self.grid_data.GetSelectionBlockTopLeft()[0][0] + 1
            cols = self.grid_data.GetSelectionBlockBottomRight()[0][1] - \
                   self.grid_data.GetSelectionBlockTopLeft()[0][1] + 1

            data = ''
            for r in range(rows):
                for c in range(cols):
                    data = data + str(self.grid_data.GetCellValue(self.grid_data.GetSelectionBlockTopLeft()[0][0] + r,
                                                                  self.grid_data.GetSelectionBlockTopLeft()[0][1] + c))
                    if c < cols - 1:
                        data = data + '\t'
                data = data + '\n'

        clipboard = wx.TextDataObject()
        clipboard.SetText(data)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
        else:
            pass

    def on_key_down_grid(self, event):
        if event.ControlDown() and event.GetKeyCode() == 67:
            self.copy_data()


class ThermalDataViewFrame(wx.Frame):

    def __init__(self, parent: Any, title: str, thermal_data: xtb.ThermalData):
        wx.Frame.__init__(self, parent, -1, title)
        self.thermal_data = thermal_data
        self.init_frame()
        self.show_data()

    def init_frame(self):
        panel = wx.Panel(self, wx.ID_ANY)
        layout = wx.BoxSizer(wx.VERTICAL)
        self.grid_data = wx.grid.Grid(panel, wx.ID_ANY)
        font = wx.Font(12, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid_data.SetFont(font)
        layout.Add(self.grid_data, 1, wx.EXPAND | wx.ALL, border=3)
        panel.SetSizerAndFit(layout)

        self.grid_data.Bind(wx.EVT_KEY_DOWN, self.on_key_down_grid)

    def show_data(self):
        self.grid_data.CreateGrid(numRows=6, numCols=1)

        # show data
        self.grid_data.SetColLabelValue(0, 'Value (Hartree)')

        self.grid_data.SetRowLabelValue(0, 'EE')
        self.grid_data.SetCellValue(0, 0, str(self.thermal_data.ee))

        self.grid_data.SetRowLabelValue(1, 'ZPE')
        self.grid_data.SetCellValue(1, 0, str(self.thermal_data.zpe))

        self.grid_data.SetRowLabelValue(2, 'G')
        self.grid_data.SetCellValue(2, 0, str(self.thermal_data.g))

        self.grid_data.SetRowLabelValue(3, 'G-EE')
        self.grid_data.SetCellValue(3, 0, str(self.thermal_data.g_corr))

        self.grid_data.SetRowLabelValue(4, 'H')
        self.grid_data.SetCellValue(4, 0, str(self.thermal_data.h))

        self.grid_data.SetRowLabelValue(5, 'H-EE')
        self.grid_data.SetCellValue(5, 0, str(self.thermal_data.h_corr))

        self.grid_data.AutoSize()

        width = min(self.grid_data.Size[0] + 100, 1200)
        height = min(self.grid_data.Size[1] + 100, 800)
        self.SetSize((width, height))

    def copy_data(self):

        # Not selected but focused on some cell
        if len(self.grid_data.GetSelectionBlockBottomRight()) == 0:
            data = str(self.grid_data.GetCellValue(self.grid_data.GetGridCursorRow(),
                                                   self.grid_data.GetGridCursorCol()))

        else:
            rows = self.grid_data.GetSelectionBlockBottomRight()[0][0] - \
                   self.grid_data.GetSelectionBlockTopLeft()[0][0] + 1
            cols = self.grid_data.GetSelectionBlockBottomRight()[0][1] - \
                   self.grid_data.GetSelectionBlockTopLeft()[0][1] + 1

            data = ''
            for r in range(rows):
                for c in range(cols):
                    data = data + str(self.grid_data.GetCellValue(self.grid_data.GetSelectionBlockTopLeft()[0][0] + r,
                                                                  self.grid_data.GetSelectionBlockTopLeft()[0][1] + c))
                    if c < cols - 1:
                        data = data + '\t'
                data = data + '\n'

        clipboard = wx.TextDataObject()
        clipboard.SetText(data)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
        else:
            pass

    def on_key_down_grid(self, event):
        if event.ControlDown() and event.GetKeyCode() == 67:
            self.copy_data()


class XTBOptTSApp(wx.App):

    # initialization
    def OnInit(self):

        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        self.resource: xrc.XmlResource = xrc.XmlResource('./wxgui/gui.xrc')

        self.load_controls()
        self.init_controls()
        self.set_events()

        # Drug & Drop settings
        dt = MyFileDropTarget(self)
        self.frame.SetDropTarget(dt)

        # redirect
        sys.stdout = self.text_ctrl_log
        sys.stderr = self.text_ctrl_log

        self.frame.Show()

        # internal variables
        self.current_input_xyz_file: Optional[Path] = None
        self.current_result_csv_file: Optional[Path] = None
        self.current_result_trajectory_file: Optional[Path] = None
        self.current_result_optimized_xyz_file: Optional[Path] = None
        self.current_result_hess_file: Optional[Path] = None
        self.current_result_freq: Optional[xtbfreq.XTBFreqResult] = None
        self.current_result_thermal_data_file: Optional[Path] = None

        # calculation thread: None means vacant
        self.calculation_thread: Optional[CalculationThread] = None

        return True

    def load_controls(self):
        self.frame = self.resource.LoadFrame(None, 'frame')

        # Codes for loading objects
        self.text_ctrl_input_file: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_input_file')
        self.button_input_file: wx.Button = xrc.XRCCTRL(self.frame, 'button_input_file')
        self.button_view_input_structure: wx.Button = xrc.XRCCTRL(self.frame, 'button_view_input_structure')
        self.text_ctrl_job_name: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_job_name')
        self.choice_xtb_method: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_xtb_method')
        self.text_ctrl_xtb_charge: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_charge')
        self.text_ctrl_xtb_uhf: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_uhf')
        self.radio_box_xtb_solvation: wx.RadioBox = xrc.XRCCTRL(self.frame, 'radio_box_xtb_solvation')
        self.choice_xtb_solvent: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_xtb_solvent')
        self.text_ctrl_cpus: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_cpus')
        self.text_ctrl_memory: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_memory')
        self.choice_keep: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_keep')
        self.checkbox_optts: wx.CheckBox = xrc.XRCCTRL(self.frame, 'checkbox_optts')
        self.text_ctrl_opt_maxcycle: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_opt_maxcycle')
        self.text_ctrl_opt_calchess: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_opt_calchess')
        self.choice_opt_convergence: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_opt_convergence')
        self.checkbox_frq: wx.CheckBox = xrc.XRCCTRL(self.frame, 'checkbox_frq')
        self.text_ctrl_temperature: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_temperature')
        self.button_run: wx.Button = xrc.XRCCTRL(self.frame, 'button_run')
        self.text_ctrl_result_csv_file: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_result_csv_file')
        self.button_result_csv_file: wx.Button = xrc.XRCCTRL(self.frame, 'button_result_csv_file')
        self.button_result_structure_view_all: wx.Button = xrc.XRCCTRL(self.frame, 'button_result_structure_view_all')
        self.button_result_structure_view_final: wx.Button = xrc.XRCCTRL(self.frame,
                                                                         'button_result_structure_view_final')
        self.text_ctrl_result_structure_number: wx.TextCtrl = xrc.XRCCTRL(self.frame,
                                                                          'text_ctrl_result_structure_number')
        self.button_result_structure_view_selected: wx.Button = xrc.XRCCTRL(self.frame,
                                                                            'button_result_structure_view_selected')
        self.button_result_structure_copy_selected: wx.Button = xrc.XRCCTRL(self.frame,
                                                                            'button_result_structure_copy_selected')
        self.button_result_structure_save_selected: wx.Button = xrc.XRCCTRL(self.frame,
                                                                            'button_result_structure_save_selected')
        self.button_result_convergence_show_table: wx.Button = xrc.XRCCTRL(self.frame,
                                                                           'button_result_convergence_show_table')
        self.button_result_convergence_plot: wx.Button = xrc.XRCCTRL(self.frame, 'button_result_convergence_plot')
        self.list_box_freq: wx.ListBox = xrc.XRCCTRL(self.frame, 'list_box_freq')
        self.button_freq_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_freq_view')
        self.text_ctrl_freq_step: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_freq_step')
        self.text_ctrl_freq_shift: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_freq_shift')
        self.button_thermal_data_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_thermal_data_view')
        self.text_ctrl_log: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_log')

        # Codes for checking objects
        assert self.text_ctrl_input_file is not None
        assert self.button_input_file is not None
        assert self.button_view_input_structure is not None
        assert self.text_ctrl_job_name is not None
        assert self.choice_xtb_method is not None
        assert self.text_ctrl_xtb_charge is not None
        assert self.text_ctrl_xtb_uhf is not None
        assert self.radio_box_xtb_solvation is not None
        assert self.choice_xtb_solvent is not None
        assert self.text_ctrl_cpus is not None
        assert self.text_ctrl_memory is not None
        assert self.choice_keep is not None
        assert self.checkbox_optts is not None
        assert self.text_ctrl_opt_maxcycle is not None
        assert self.text_ctrl_opt_calchess is not None
        assert self.choice_opt_convergence is not None
        assert self.checkbox_frq is not None
        assert self.text_ctrl_temperature is not None
        assert self.button_run is not None
        assert self.text_ctrl_result_csv_file is not None
        assert self.button_result_csv_file is not None
        assert self.button_result_structure_view_all is not None
        assert self.button_result_structure_view_final is not None
        assert self.text_ctrl_result_structure_number is not None
        assert self.button_result_structure_view_selected is not None
        assert self.button_result_structure_copy_selected is not None
        assert self.button_result_structure_save_selected is not None
        assert self.button_result_convergence_show_table is not None
        assert self.button_result_convergence_plot is not None
        assert self.list_box_freq is not None
        assert self.button_freq_view is not None
        assert self.text_ctrl_freq_step is not None
        assert self.text_ctrl_freq_shift is not None
        assert self.button_thermal_data_view is not None
        assert self.text_ctrl_log is not None

    def init_controls(self):
        # XTB method
        for method in config.XTB_METHOD_LIST:
            self.choice_xtb_method.Append(method)
        self.choice_xtb_method.SetStringSelection('gfn2')

        # XTB solvent
        for solvent in config.XTB_SOLVENT_LIST:
            self.choice_xtb_solvent.Append(solvent)
        self.choice_xtb_solvent.SetStringSelection('None')

        # keep log
        for keep in ['No', 'Yes', 'when fail']:
            self.choice_keep.Append(keep)
            self.choice_keep.SetStringSelection('when fail')

        # Optimization Settings
        for converge in config.CONVERGENCE_CRITERIA_LIST:
            self.choice_opt_convergence.Append(converge)
        self.choice_opt_convergence.SetStringSelection('Tight')

    def set_events(self):
        # Input file
        self.button_input_file.Bind(wx.EVT_BUTTON, self.on_button_input_file)
        self.text_ctrl_input_file.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter_input_file)
        self.button_view_input_structure.Bind(wx.EVT_BUTTON, self.on_button_view_input_structure)

        # Results
        self.button_result_csv_file.Bind(wx.EVT_BUTTON, self.on_button_result_csv_file)
        self.text_ctrl_result_csv_file.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter_result_csv_file)
        self.button_result_structure_view_all.Bind(wx.EVT_BUTTON, self.on_button_result_structure_view_all)
        self.button_result_structure_view_final.Bind(wx.EVT_BUTTON, self.on_button_result_structure_view_final)
        self.button_result_structure_view_selected.Bind(wx.EVT_BUTTON, self.on_button_result_structure_view_selected)
        self.button_result_structure_copy_selected.Bind(wx.EVT_BUTTON, self.on_button_result_structure_copy_selected)
        self.button_result_structure_save_selected.Bind(wx.EVT_BUTTON, self.on_button_result_structure_save_selected)
        self.button_result_convergence_show_table.Bind(wx.EVT_BUTTON, self.on_button_result_convergence_show_table)
        self.button_result_convergence_plot.Bind(wx.EVT_BUTTON, self.on_button_result_convergence_plot)
        self.button_freq_view.Bind(wx.EVT_BUTTON, self.on_button_freq_view)
        self.button_thermal_data_view.Bind(wx.EVT_BUTTON, self.on_button_thermal_data_view)

        # Run
        self.button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        # Event when calculation finished
        self.Bind(EVT_CALC_END, self.on_calc_end)

    def load_input_structure_file(self, file: Path):
        # check file type by suffix and convert Gaussian file to xyz
        if file.suffix.lower() in ['.gjf', '.gjc', '.com']:
            file = convert.gaussian_input_to_xyz(file)
            atoms, coordinates = xyzutils.read_single_xyz_file(file)
            self.logging('Convert Gaussian input to xyz: {}'.format(file))
        elif file.suffix.lower() in ['.log', '.out']:
            file = convert.gaussian_log_to_xyz(file)
            atoms, coordinates = xyzutils.read_single_xyz_file(file)
            self.logging('Convert Gaussian log (final structure) to xyz: {}'.format(file))
        elif file.suffix.lower() != '.xyz':
            file = convert.other_to_xyz(file)
            atoms, coordinates = xyzutils.read_single_xyz_file(file)
            self.logging('Convert some log file (final structure) to xyz via cclib: {}'.format(file))
        else:
            atoms, coordinates_list, _ = xyzutils.read_sequential_xyz_file(file)
            # in case of xyz file, use final structure
            if len(coordinates_list) > 1:
                coordinates = coordinates_list[-1]
                file = file.absolute().parent / (file.stem + '_last.xyz')
                xyzutils.save_xyz_file(xyz_file=file, atoms=atoms, coordinates=coordinates)

        self.text_ctrl_input_file.SetValue(str(file.absolute()))
        self.current_input_xyz_file = file.absolute()

        # Default job name
        self.text_ctrl_job_name.SetValue(self.current_input_xyz_file.stem  + '_xtboptts')

    def load_result_csv_file(self, file: Path):
        self.text_ctrl_result_csv_file.SetValue(str(file.absolute()))
        self.current_result_csv_file = file.absolute()

        # check and read the corresponding other result files
        trajectory_file = file.absolute().parent / (file.stem + config.OUTPUT_TRAJECTORY_FILE_SUFFIX)
        optimized_xyz_file = file.absolute().parent / (file.stem + config.OUTPUT_FINAL_XYZ_FILE_SUFFIX)
        hess_file = file.absolute().parent / (file.stem + config.OUTPUT_HESS_FILE_SUFFIX)
        thermal_data_file = file.absolute().parent / (file.stem + config.OUTPUT_THERMAL_FILE_SUFFIX)

        # trajectory
        if not trajectory_file.exists():
            self.logging('The related result trajectory (xyz) file not found: {:}'.format(trajectory_file))
            self.current_result_trajectory_file = None
        else:
            self.logging('The related result xyz file is found: {:}'.format(trajectory_file))
            self.current_result_trajectory_file = trajectory_file

        # final
        if not optimized_xyz_file.exists():
            self.logging('The related result optimized (xyz) file not found: {:}'.format(optimized_xyz_file))
            self.current_result_optimized_xyz_file = None
        else:
            self.logging('The related result optimized (xyz) file is found: {:}'.format(optimized_xyz_file))
            self.current_result_optimized_xyz_file = optimized_xyz_file

        # hess (freq) file
        if not hess_file.exists():
            self.logging('The related result Hessian file not found: {:}'.format(hess_file))
            self.current_result_hess_file = None
        else:
            self.logging('The related result Hessian file is found: {:}'.format(hess_file))
            self.current_result_hess_file = hess_file
            if self.current_result_optimized_xyz_file is None:
                self.logging('Freq information is not available because no optimized structure file found.')
                self.current_result_freq = None
                self.list_box_freq.Clear()
            else:
                self.logging('Calculating freq information from the Hessian file...')
                self.current_result_freq = \
                    xtbfreq.XTBFreqResult(structure_xyz_file=self.current_result_optimized_xyz_file,
                                          hessian_file=self.current_result_hess_file)
                self.list_box_freq.Clear()
                for frequency in self.current_result_freq.frequencies:
                    self.list_box_freq.Append('{:.2f}'.format(frequency))
                self.logging('Done')

        # thermal data file
        if not thermal_data_file.exists():
            self.logging('The related result thermal data file not found: {:}'.format(thermal_data_file))
            self.current_result_thermal_data_file = None
        else:
            self.logging('The related result othermal data file is found: {:}'.format(thermal_data_file))
            self.current_result_thermal_data_file = thermal_data_file

    def logging(self, message):
        """
        output logs
        """
        log_string = (''.join(message)).rstrip()
        self.text_ctrl_log.write(log_string + '\n')

    # Event-handlers
    def on_button_input_file(self, event):
        dialog = wx.FileDialog(None, 'Select input structure file',
                               wildcard='XYZ or Gaussian file  (*.xyz;*.gjf;*.gjc;*.com;*.log;*.out)|'
                                        '*.xyz;*.gjf;*.gjc;*.com;*.log;*.out'
                                        '|All files (*.*)|*.*',
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.load_input_structure_file(file)
        dialog.Destroy()

    def on_text_enter_input_file(self, event):
        file = Path(self.text_ctrl_input_file.GetValue()).absolute()
        if file.exists():
            self.load_input_structure_file(file)
        else:
            self.logging('Input file not found: {}'.format(file))
            self.current_input_xyz_file = None

    def on_button_view_input_structure(self, event):
        if self.current_input_xyz_file is not None:
            view.view_xyz_file(self.current_input_xyz_file)
        else:
            self.logging('No input file')

    def on_button_run(self, event):
        if self.calculation_thread is not None:
            # Termination check
            msgbox = wx.MessageDialog(None,
                                      'Stop the running calculation?',
                                      'Calculation in running',
                                      style=wx.YES_NO)
            stop_check = msgbox.ShowModal()
            if stop_check == wx.ID_YES:
                msgbox.Destroy()
                self.calculation_thread.terminate()
                self.logging('Termination requested.')
            else:
                msgbox.Destroy()
                return

        else:
            # Input xyz
            if self.current_input_xyz_file is None:
                self.logging('No input file.')
                return

            # Settings for calculation
            # XTB Parameters
            xtb_method = self.choice_xtb_method.GetStringSelection()
            try:
                xtb_charge = int(self.text_ctrl_xtb_charge.GetValue().strip())
            except ValueError:
                self.logging('Charge is not valid.')
                return
            try:
                xtb_uhf = int(self.text_ctrl_xtb_uhf.GetValue().strip())
                assert xtb_uhf >= 0
            except:
                self.logging('UHF is not valid.')
                return
            xtb_solvation = self.radio_box_xtb_solvation.GetStringSelection().lower()
            if xtb_solvation == 'none':
                xtb_solvation = None
                xtb_solvent = ''
            else:
                xtb_solvent = self.choice_xtb_solvent.GetStringSelection().strip().lower()
                if not xtb_solvent:
                    self.logging('Solvent must be selected.')
                    return
            xtb_params = xtb.XTBParams(method=xtb_method, charge=xtb_charge, uhf=xtb_uhf,
                                       solvation=xtb_solvation, solvent=xtb_solvent)

            # cpus, memory, keep_log
            try:
                num_cpus = int(self.text_ctrl_cpus.GetValue().strip())
                assert num_cpus >= 1
            except:
                self.logging('The Number of CPUs is not valid. It must be positive integer.')
                return
            memory_per_cpu: str = self.text_ctrl_memory.GetValue().strip()
            memory_per_cpu = memory_per_cpu.upper().replace(' ', '').replace('　', '')
            keep_log = self.choice_keep.GetStringSelection().lower()
            if keep_log == 'yes':
                keep_log = 2
            elif keep_log == 'when fail':
                keep_log = 1
            else:
                keep_log = 0

            # Job name (used for output file names)
            job_name = self.text_ctrl_job_name.GetValue().strip()
            if job_name == '':
                job_name = self.current_input_xyz_file.stem + '_xtboptts'
                self.text_ctrl_job_name.SetValue(job_name)

            # Optimization Settings
            ts = self.checkbox_optts.GetValue()

            try:
                maxcycles = int(self.text_ctrl_opt_maxcycle.GetValue())
                assert maxcycles > 0
            except:
                self.logging('Max Cycle should be positive integer')
                return

            try:
                calc_hess = int(self.text_ctrl_opt_calchess.GetValue())
                assert calc_hess >= 0
            except:
                self.logging('Calc. Hess. should be non-negative integer')
                return

            converge = self.choice_opt_convergence.GetStringSelection()
            freq = self.checkbox_frq.GetValue()

            try:
                temperature = float(self.text_ctrl_temperature.GetValue())
                assert temperature > 0
            except:
                self.logging('Temp. should be > 0')
                return

            # Start Calculation
            self.calculation_thread = CalculationThread(input_xyz_file=self.current_input_xyz_file,
                                                        job_name=job_name,
                                                        xtb_params=xtb_params,
                                                        temperature=temperature,
                                                        ts=ts,
                                                        converge=converge,
                                                        maxcycles=maxcycles,
                                                        hess_step=calc_hess,
                                                        final_hess=freq,
                                                        keep_log=keep_log,
                                                        num_threads=num_cpus,
                                                        memory_per_thread=memory_per_cpu,
                                                        parent_window=self)

            self.button_run.SetLabelText('Stop')
            self.logging('Calculation started: {}'.format(job_name))

    def on_button_result_csv_file(self, event):
        dialog = wx.FileDialog(None, 'Select result csv file',
                               wildcard='CSV file (*.csv)|*.csv|All files (*.*)|*.*',
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.load_result_csv_file(file)
        dialog.Destroy()

    def on_text_enter_result_csv_file(self, event):
        file = Path(self.text_ctrl_result_csv_file.GetValue()).absolute()
        if file.exists():
            self.load_result_csv_file(file)
        else:
            self.logging('Input file not found: {}'.format(file))
            self.current_result_csv_file = None
            self.current_result_trajectory_file = None
            self.current_result_optimized_xyz_file = None
            self.current_result_hess_file = None
            self.current_result_freq = None
            self.current_result_thermal_data_file = None
            self.list_box_freq.Clear()

    def on_button_result_structure_view_all(self, event):
        if self.current_result_trajectory_file is None:
            self.logging('No result trajectory (xyz) file.')
            return
        view.view_xyz_file(self.current_result_trajectory_file)

    def on_button_result_structure_view_final(self, event):
        if self.current_result_optimized_xyz_file is None:
            self.logging('No result optimized xyz file.')
            return
        view.view_xyz_file(self.current_result_optimized_xyz_file)

    def on_button_result_structure_view_selected(self, event):
        if self.current_result_trajectory_file is None:
            self.logging('No result trajectory (xyz) file.')
            return

        # check input number
        try:
            num = int(self.text_ctrl_result_structure_number.GetValue())
        except:
            self.logging('The given number is not valid.')
            return

        # read xyz file
        atoms, coordinates, titles = xyzutils.read_sequential_xyz_file(self.current_result_trajectory_file)

        # check input number range
        num_structures = len(coordinates)
        if num < 0 or num >= num_structures:
            self.logging('The given number is out of range. Input 0 to {:}.'.format(num_structures-1))
            return

        view.view_xyz_structure(atoms=atoms,
                                coordinates=coordinates[num],
                                title=titles[num])

    def on_button_result_structure_copy_selected(self, event):
        if self.current_result_trajectory_file is None:
            self.logging('No result trajectory (xyz) file.')
            return

        # check input number
        try:
            num = int(self.text_ctrl_result_structure_number.GetValue())
        except:
            self.logging('The given number is not valid.')
            return

        # read xyz file
        atoms, coordinates, titles = xyzutils.read_sequential_xyz_file(self.current_result_trajectory_file)

        # check input number range
        num_structures = len(coordinates)
        if num < 0 or num >= num_structures:
            self.logging('The given number is out of range. Input 0 to {:}.'.format(num_structures - 1))
            return

        xyz_string = xyzutils.get_xyz_string(atoms, coordinates[num])

        clipboard = wx.TextDataObject()
        clipboard.SetText(xyz_string)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
            self.logging('XYZ coordinates copied to clipboard.')
        else:
            self.logging('Clipboard is not accessible.')

    def on_button_result_structure_save_selected(self, event):
        if self.current_result_trajectory_file is None:
            self.logging('No result trajectory (xyz) file.')
            return

        # check input number
        try:
            num = int(self.text_ctrl_result_structure_number.GetValue())
        except:
            self.logging('The given number is not valid.')
            return

        # read xyz file
        atoms, coordinates, titles = xyzutils.read_sequential_xyz_file(self.current_result_trajectory_file)

        # check input number range
        num_structures = len(coordinates)
        if num < 0 or num >= num_structures:
            self.logging('The given number is out of range. Input 0 to {:}.'.format(num_structures - 1))
            return

        dialog = wx.FileDialog(None, 'save file name',
                               wildcard='XYZ file (*.xyz)|*.xyz|All files (*.*)|*.*',
                               style=wx.FD_SAVE)
        dialog.SetDirectory(str(self.current_result_trajectory_file.parent))
        if dialog.ShowModal() == wx.ID_OK:
            file = dialog.GetPath()
            dialog.Destroy()
        else:
            dialog.Destroy()
            return

        xyzutils.save_xyz_file(xyz_file=Path(file), atoms=atoms, coordinates=coordinates[num], title=titles[num])

    def on_button_result_convergence_show_table(self, event):
        if self.current_result_csv_file is None:
            self.logging('No result csv file.')
            return

        with self.current_result_csv_file.open(mode='r') as f:
            reader = csv.reader(f)
            csv_data = list(reader)
            title = str(self.current_result_csv_file)

            # check: all lines have same length
            length = len(csv_data[0])
            for line in csv_data[1:]:
                if len(line) != length:
                    self.logging('CSV data is no valid. All data lines must have the same length.')
                    return
            # show
            csv_view_frame = CSVViewFrame(self.frame, title=title, data=csv_data)
            csv_view_frame.Show(True)

    def on_button_result_convergence_plot(self, event):
        if self.current_result_csv_file is None:
            self.logging('No result csv file.')
            return

        csv_data = np.genfromtxt(self.current_result_csv_file,
                                 dtype=float, skip_header=1, missing_values='NA', delimiter=',')
        xs = np.arange(csv_data.shape[0])

        plt.figure('Optimization', figsize=config.PLOT_SIZE)

        energies = csv_data[:,1]
        energy_changes = csv_data[:,2]
        gradient_rms = csv_data[:,3]
        gradient_max = csv_data[:,4]
        displacement_rms = csv_data[:,5]
        displacement_max = csv_data[:,6]

        # energy
        plt.subplot(6, 1, 1)
        plt.title('Energy')
        plt.ylim(*utils.calc_limit_for_plot(energies))
        plt.plot(xs, energies)
        # bare energy
        plt.subplot(6, 1, 2)
        plt.title('Energy Change')
        plt.ylim(*utils.calc_limit_for_plot(energy_changes))
        plt.plot(xs, energy_changes)
        # RMS Grad
        plt.subplot(6, 1, 3)
        plt.title('RMS Gradient')
        plt.ylim(*utils.calc_limit_for_plot(gradient_rms))
        plt.plot(xs, gradient_rms)
        # Max Grad
        plt.subplot(6, 1, 4)
        plt.title('Max Gradient')
        plt.ylim(*utils.calc_limit_for_plot(gradient_max))
        plt.plot(xs, gradient_max)
        # RMS Displacement
        plt.subplot(6, 1, 5)
        plt.title('RMS Displacement')
        plt.ylim(*utils.calc_limit_for_plot(displacement_rms))
        plt.plot(xs, displacement_rms)
        # Max Displacement
        plt.subplot(6, 1, 6)
        plt.title('Max Displacement')
        plt.ylim(*utils.calc_limit_for_plot(displacement_max))
        plt.plot(xs, displacement_max)

        plt.tight_layout()
        plt.show()

    def on_button_freq_view(self, event):
        if self.current_result_freq is None:
            self.logging('No available data.')
            return

        i = self.list_box_freq.GetSelection()
        if i < 0:
            self.logging('Please select frequency.')
            return

        try:
            freq_step = int(self.text_ctrl_freq_step.GetValue())
            assert freq_step > 0
        except:
            self.logging('Step should be positive integer.')
            return

        try:
            freq_shift = float(self.text_ctrl_freq_shift.GetValue())
            assert freq_shift > 0.0
        except:
            self.logging('Shift should be positive value.')
            return

        freq_dir = self.current_result_hess_file.parent / (self.current_result_hess_file.stem + '_freq')
        if not freq_dir.exists():
            freq_dir.mkdir()
        freq_file = freq_dir / 'freq{:0>5}.xyz'.format(i)
        self.current_result_freq.save_freq_xyz(file=freq_file, index=i, step=freq_step, max_shift=freq_shift)
        view.view_xyz_file(freq_file)

    def on_button_thermal_data_view(self, event):
        if self.current_result_thermal_data_file is None:
            self.logging('No thermal data file.')
            return

        title = self.current_result_thermal_data_file.name

        thermal_data = xtb.ThermalData(self.current_result_thermal_data_file)
        td_view_frame = ThermalDataViewFrame(parent=self.frame, title=title, thermal_data=thermal_data)
        td_view_frame.Show(True)

    def on_calc_end(self, event: CalcEndEvent):

        self.calculation_thread = None
        self.button_run.SetLabelText('Run')

        job_dir: Path = event.job_dir
        job_name: str = event.job_name
        job_result: opt_geometric.GeometricResult = event.job_result

        if job_result.status == 'converged':
            self.logging('Optimization successfully converged: {}'.format(job_name))
            msgbox = wx.MessageDialog(None,
                                      'Optimization successfully converged: {}\n'.format(job_name) +
                                      'Load result files?',
                                      'Converged',
                                      style=wx.YES_NO)
            load_check = msgbox.ShowModal()
            if load_check == wx.ID_YES:
                msgbox.Destroy()
                result_csv_file = job_dir / (job_name + '.csv')
                self.load_result_csv_file(result_csv_file)
            else:
                msgbox.Destroy()

        elif job_result.status == 'not converged':
            self.logging('Optimization NOT converged: {}'.format(job_name))
            msgbox = wx.MessageDialog(None,
                                      'Optimization NOT converged: {}\n'.format(job_name) +
                                      'Load result files?',
                                      'Not Converged',
                                      style=wx.YES_NO)
            load_check = msgbox.ShowModal()
            if load_check == wx.ID_YES:
                msgbox.Destroy()
                result_csv_file = job_dir / (job_name + '.csv')
                self.load_result_csv_file(result_csv_file)
            else:
                msgbox.Destroy()

        # fail in calculations with some errors or termination
        else:
            self.logging('Calculation failed: {}'.format(job_name))
            msgbox = wx.MessageDialog(None,
                                      'Calculation Failed: {}\n'.format(job_name) +
                                      'Try to load result files anyway?',
                                      'Calculation Failed',
                                      style=wx.YES_NO)
            load_check = msgbox.ShowModal()
            if load_check == wx.ID_YES:
                msgbox.Destroy()
                result_csv_file = job_dir / (job_name + '.csv')
                self.load_result_csv_file(result_csv_file)
            else:
                msgbox.Destroy()

            if job_result.error is not None:
                self.logging(job_result.error.args)
            else:
                self.logging('Unexpected Errors. Need to debug.')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = XTBOptTSApp(False)
    app.MainLoop()
