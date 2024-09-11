# Required PATHs settings
VIEWER_PATH = 'D:/programs/jmol/jmol.bat'
XTB_BIN = 'D:/programs/xtb-6.6.1/bin/xtb.exe'
XTB_PARAM_DIR = 'D:/programs/xtb-6.6.1/share/xtb'
XTB_OTHER_LIB_DIR = None  # in case other library directory (MinGW etc.) has  to be in path to run xtbscan.

# Customization of plotting
PLOT_SIZE = (8, 6)

# Constants (not necessary to edit)
FLOAT = float
ANG_TO_BOHR = 1.8897259886

XTB_LOG_FILE = 'xtblog.txt'
XTB_INPUT_FILE = 'xtbinp.txt'
XTB_ENERGY_FILE = 'energy'
XTB_GRADIENT_FILE = 'gradient'
XTB_HESSIAN_FILE = 'hessian'

XYZ_FORMAT = '{:<2s}  {:>12.8f}  {:>12.8f}  {:>12.8f}\n'

INIT_XYZ_FILE = 'init.xyz'
OUTPUT_TRAJECTORY_FILE_SUFFIX = '_traj.xyz'
OUTPUT_CSV_FILE_SUFFIX = '.csv'
OUTPUT_FINAL_XYZ_FILE_SUFFIX = '_final.xyz'
OUTPUT_HESS_FILE_SUFFIX = '_hess.txt'
OUTPUT_THERMAL_FILE_SUFFIX = '_thermo.txt'

STOP_FILE_SUFFIX = '_stopmessage.dat'

XTB_METHOD_LIST = [
    'gfn1',
    'gfn2',
    'gfn0',
    'gfnff'
]

XTB_SOLVENT_LIST = [
    'None',
    'Acetone',
    'Acetonitrile',
    'Aniline',
    'Benzaldehyde',
    'Benzene',
    'CH2Cl2',
    'CHCl3',
    'CS2',
    'Dioxane',
    'DMF',
    'DMSO',
    'Ether',
    'Ethylacetate',
    'Furane',
    'Hexadecane',
    'Hexane',
    'Methanol',
    'Nitromethane',
    'Octanol',
    'Phenol',
    'Toluene',
    'THF',
    'Water'
]

ATOM_LIST = ['bq', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
             'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
             'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
             'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
             'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
             'Mt', 'Ds', 'Rg', 'Cn']

CONVERGENCE_CRITERIA_LIST = [
    'Loose',
    'Normal',
    'Tight',
    'VeryTight'
]
