import pandas as pd

import numpy as np
from ase.visualize import view

from ase.io import read, write

atoms = read('POSCAR')
df = pd.read_table('ACF.dat', delim_whitespace=True, header=0, comment = '-')
charges = df['CHARGE'][1:-4].to_numpy()
n_si = len([a for a in atoms if a.symbol == 'Si'])
n_o = len([a for a in atoms if a.symbol == 'O'])
n_al = len([a for a in atoms if a.symbol == 'Al'])

ocharges = np.array([4]*n_si+[3]*n_al+[6]*n_o)
dcharges = charges-ocharges

atoms.set_initial_magnetic_moments(np.round(dcharges, 2))
write('with_charges.traj', atoms)
