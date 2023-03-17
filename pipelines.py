from ase.io import read, write
from ase.calculators.vasp import Vasp 
import shutil
import os
import subprocess
import pandas as pd
import numpy as np


def get_base_calc():
    base_calc = Vasp(gga = 'PE',  # Change this according to the functional used
                     lreal='Auto',
                     lplane = True,
                     lwave = False,
                     lcharg = False, 
                     npar = 4, 
                     prec = 'Normal',
                     encut = 400, 
                     ediff = 1e-6, 
                     algo = 'VeryFast', 
                     ismear = -5, 
                     gamma = True,
                     command = os.getenv('VASP_COMMAND'))
    
    return base_calc


def geo_opt(atoms, mode = 'vasp'):
    write('CONTCAR', atoms)
    calc = get_base_calc()
    calc.set(ibrion = 2,
             ediffg = -1e-2,
             nsw = 200,
             nelm = 200)

    # add other settings as needed
    opt_levels = {'kpts': {1: [3, 3, 3],
                           2: [5, 5, 5],
                           3: [7, 7, 7]}} 
    for level in [1, 2, 3]: 
        calc.set(kpts = opt_levels['kpts'][level])
        atoms_tmp = read('CONTCAR')
        atoms_tmp.calc = calc
        atoms_tmp.get_potential_energy()
        calc.reset()
        atoms_tmp = read('OUTCAR', index=-1)
        shutil.copyfile('CONTCAR', f'opt{level}.vasp')
        shutil.copyfile('vasprun.xml', f'opt{level}.xml')
        shutil.copyfile('OUTCAR', f'opt{level}.OUTCAR')

    return atoms_tmp


def freq(atoms, mode = 'vasp'):
    calc = get_base_calc()
    calc.set(ibrion = 5,
             potim = 0.015,
             nsw = 500, # as many dofs as needed
             kpts = [1, 7, 5]) # todo: kpts depends strongly on the structure, and should be supplied in atoms 
    atoms.calc = calc
    atoms.get_potential_energy()
    # todo: parse OUTCAR frequencies and modes

def bader(atoms):
    
    calc = get_base_calc()
    calc.set(ibrion = -1,
             nsw = 0,
             lorbit = 12,
             lcharg = True,
             laechg = True)
    
    atoms.calc = calc
    atoms.get_potential_energy()
    
    # add charges
    chgsum = os.getenv('VTST_SCRIPTS')+'/chgsum.pl'
    proc = subprocess.run([chgsum, 'AECCAR0', 'AECCAR2'], capture_output=True)
    
    # run bader
    bader = os.getenv('VTST_BADER')
    proc = subprocess.run([bader], capture_output=True)
    
    df = pd.read_table('ACF.dat', delim_whitespace=True, header=0, comment = '-')
    charges = df['CHARGE'][1:-4].to_numpy()
    n_si = len([a for a in atoms if a.symbol == 'Si'])
    n_o = len([a for a in atoms if a.symbol == 'O'])
    n_al = len([a for a in atoms if a.symbol == 'Al'])
    
    ocharges = np.array([4]*n_si+[3]*n_al+[6]*n_o)
    dcharges = -charges + ocharges
    atoms.set_initial_charges(np.round(dcharges, 2))
    
    return atoms


def cohp(atoms, bonds, lobsterin_template = None):
    calc = get_base_calc()
    calc.set(ibrion = -1,
             nsw = 0,
             isym = -1)

    n_si = len([a for a in atoms if a.symbol == 'Si'])
    n_o = len([a for a in atoms if a.symbol == 'O'])
    n_al = len([a for a in atoms if a.symbol == 'Al'])

    nelect = n_si*4 + n_o*6+n_al*3
    calc.set(nbands = nelect + 20) # giving 20 empty bands. may require more

    
    atoms.calc = calc
    atoms.get_potential_energy()

    lobsterin = 'lobsterin'
    if lobsterin_template:
        shutil.copyfile(lobsterin_template, f'./{lobsterin}')
    else:
        # check lobsterin is in the current folder
        assert os.path.exist(f'{lobsterin}'), f'{lobsterin} not found, aborting'
    with open(f'{lobsterin}', 'w+') as fhandle:
        for b in bonds:
            fhandle.write(f'cohpbetween atom {b[0]+1} and atom {b[1]+1}\n')
            
    # todo: launch lobster and retrieve results to ase
    # lobster in file needs to be modified so not launching it like so

    
if __name__ == '__main__':
    before = read('before.vasp')
    opted = geo_opt(before)
    write('opted.traj', opted)
