# Yantao Xia, 10/27/2022, xyttyxy@ucla.edu

from ase.optimize.bfgs import BFGS
import numpy as np
import pandas as pd

class auto_thermochemistry:
    # settings
    phonon_kpts = []
    phonon_supercell = []
    free_energy_thermo_info = [] # and the constants used there
    # syntax for file I/O
    # allowed file extensions
    # default_temperature and pressure

    def __init__(self, ncores=1):
        self.phonon_kpts = (40, 40, 40)
        self.phonon_supercell = (5, 5, 5)

        self.optimized_file_suffix = 'opt'
        self.allowed_file_extensions = ['traj', 'vasp', 'cif']
        self.pbc_cell = [10, 11, 12]
        self._geo_nl = 'nonlinear'
        self._geo_li = 'linear'
        self._geo_na = 'NA'
        self._phase_gas = 'gas'
        self._phase_crystal = 'crystal'
        self._symmetry_na = -1
        self._spin_na = -1
        self._spacegroup_na = 'NA'
        self.ncores = ncores

        self.thermo_info = pd.DataFrame(
            {'name': ['H2O', 'CO2', 'HCOOH', 'CuO_bulk', 'oxide_relax'],
             'phase': [self._phase_gas, self._phase_gas, self._phase_gas, self._phase_crystal, self._phase_crystal],
             'geometry': [self._geo_nl, self._geo_li, self._geo_nl, self._geo_na, self._geo_na],
             'symmetrynumber': [2, 2, 1, self._symmetry_na, self._symmetry_na],
             'spin': [0, 0, 0, -1, -1],
             'spacegroup': ['C2v', 'Dinfh', 'Cs', self._spacegroup_na, self._spacegroup_na],
             'formulaunits': [1, 1, 1, 4, 1],
             'calculate_vib': [True, True, True, True, False]},) # gas is always 1

    def build_lammps_calculator(self, name):
        from ase.calculators.lammpslib import LAMMPSlib
        lmp_cmds = ['pair_style reax/c NULL',
                    'pair_coeff * * ffield.reax Cu C O H',
                    'fix 1 all qeq/reax 1 0.0 10.0 1e-6 reax/c',
                    'neighbor        2 bin',
                    'neigh_modify    every 10 delay 0 check no']

        lammps_header = ['units real',
                         'atom_style charge',
                         'atom_modify map yes']

        # todo: change this to use only relevant element types
        # todo: read mass from ase.data

        calc = LAMMPSlib(lmpcmds = lmp_cmds + [f'dump 1 all custom 1 dump.{name} id type x y z vx vy vz fx fy fz'],
                         atom_types = {'Cu': 1, 'C': 2, 'O': 3, 'H': 4},
                         atom_type_masses = {'Cu': 63.546, 'C': 12.011, 'O': 15.599, 'H': 1.00784},
                         log_file = f'log-{name}.lammps',
                         keep_alive=True,
                         lammps_header = lammps_header,
                         directory = name)
        return calc

    def build_vasp_calculator(self, name, kpts=(1,1,1)):
        from ase.calculators.vasp import Vasp
        if self.read_table(name, 'phase') == self._phase_gas:
            calc = Vasp(command=f'mpirun -np {self.ncores:d} /u/home/x/xyttyxy/selfcompiled-programs/bin/vasp_gam_intel20_comp221101_withsol',
                        prec='Accurate',
                        istart=1,
                        algo='VeryFast',
                        ispin=2,
                        nelm=100,
                        ibrion=-1,
                        nsw=-1,
                        ediff=1E-6,
                        ediffg=-2E-2,
                        gga='PE',
                        encut=400,
                        lreal=False,
                        ismear=0,
                        sigma=0.01,
                        kpts=kpts,
                        gamma=True,
                        lwave=True, # load wavecar to make freq scf more efficient
                        lcharg=False)

        elif self.read_table(name, 'phase') == self._phase_crystal:
            calc = Vasp(command='/u/home/x/xyttyxy/selfcompiled-programs/bin/vasp_std_5.4.4',
                        prec='Accurate',
                        istart=0,
                        algo='Fast',
                        ispin=2,
                        nelm=100,
                        ibrion=-1,
                        ediff=1E-8,
                        ediffg=-1E-3,
                        gga='PE',
                        nsw=-1,
                        isif=0,
                        encut=400,
                        lreal='Auto',
                        ismear=-5,# insulator
                        kpts=kpts,
                        gamma=True,
                        lwave=False,
                        lcharg=False)

        return calc

    def get_vib_data(self, atoms, name, calculator):
        if calculator == 'reax':
            calc = self.build_lammps_calculator(name)
        elif calculator == 'nnp':
            RuntimeError('NNP not implemented. ')
            from ase.calculators.pynnp import PyNNP
            calc = PyNNP()
        elif calculator == 'dft':
            calc = self.build_vasp_calculator(name)

        atoms.set_pbc(True)
        atoms.calc = calc

        opt_fname = f'{name}-opt.traj'
        try:
            atoms = read(opt_fname)
            assert np.amax(atoms.get_forces()) < 0.02
            atoms.calc = calc # traj has dummy SinglePointCalculator,
            # causing force not PropertyNotImplementedError
        except (FileNotFoundError, AssertionError):
            dyn = BFGS(atoms, logfile=f'{name}-opt.log', trajectory=opt_fname)
            dyn.run(fmax=0.02)

        from ase.vibrations import Vibrations
        vib = Vibrations(atoms, name = name)
        vib.run()
        vib_data = vib.get_vibrations(read_cache=False)
        return vib_data

    def get_phonon_data(self, atoms, name, calculator):
        if calculator == 'reax':
            calc_opt = self.build_lammps_calculator(name)
            calc_phonon = self.build_lammps_calculator(name)
        elif calculator == 'nnp':
            RuntimeError('NNP not implemented. ')
            from ase.calculators.pynnp import PyNNP
            calc_opt = PyNNP()
            calc_phonon = PyNNP()
        elif calculator == 'dft':
            calc_opt = self.build_vasp_calculator(name, kpts=(10,10,10))
            calc_phonon = self.build_vasp_calculator(name)
        else:
            RuntimeError(f'Calculator not recognized: {calculator}')

        from ase.constraints import UnitCellFilter
        from ase.io import Trajectory
        from ase.phonons import Phonons

        atoms.set_pbc(True)

        atoms.calc = calc_opt
        ucf = UnitCellFilter(atoms)
        dyn = BFGS(ucf, logfile=f'{name}-opt.log')
        traj = Trajectory(f'{name}-opt.traj', 'w', atoms)
        dyn.attach(traj)
        dyn.run(fmax=0.02)

        # supercell should be specified by thermo_info
        ph = Phonons(atoms, calc_phonon, supercell=(1, 1, 1))
        ph.run()
        ph.read(acoustic=True)
        omega_e, dos_e = ph.dos(kpts=(40, 40, 40), npts=3000, delta=5e-4)

        # from phonons.py: def dos(): return omega_e, dos_e
        # from thermochemistry.py: omega_e = self.phonon_energies, dos_e = self.phonon_dos
        # so, 1st return is energy, 2nd is DOS

        return omega_e, dos_e

    def read_table(self, name, prop):
        return self.thermo_info[self.thermo_info['name'] == name][prop].item()

    def get_free_energies(self, atoms, name, options):
        phase = self.read_table(name, 'phase')
        if phase == self._phase_gas:
            # check if frequency is available when using cache
            fname = f'{name}-freq.pickle'
            if options['cache'] and os.path.isfile(fname):
                with open(fname, 'wb+') as file:
                    try:
                        vib_data = pickle.load(file)
                    except EOFError:
                        vib_data = self.get_vib_data(atoms, name, options['calculator'])
                        pickle.dump(vib_data, file)
            else:
                with open(fname, 'wb') as file:
                    vib_data = self.get_vib_data(atoms, name, options['calculator'])
                    pickle.dump(vib_data, file)

            # at this point, atoms need to be in the optimized geometry
            atoms_opt = read(f'{name}-opt.traj', index=-1)
            # todo: warn if atoms_opt very different from atoms
            from ase.thermochemistry import IdealGasThermo
            thermo = IdealGasThermo(vib_energies = vib_data.get_energies(),
                                    potentialenergy = atoms_opt.get_potential_energy(),
                                    atoms=atoms_opt,
                                    geometry=self.read_table(name, 'geometry'),
                                    symmetrynumber = self.read_table(name, 'symmetrynumber'),
                                    spin=self.read_table(name, 'spin'))
            G = thermo.get_gibbs_energy(temperature = options['temperature'],
                                        pressure = options['pressure'],
                                        verbose = options['debug'])
            retval = G
        elif phase == self._phase_crystal:
            if self.read_table(name, 'calculate_vib'):
                fname_dos = f'{name}-phonon_dos.pickle'
                fname_en = f'{name}-phonon_en.pickle'
                if options['cache'] and os.path.isfile(fname_dos) and os.path.isfile(fname_en):
                    with open(fname_en, 'rb') as file:
                        phonon_en = pickle.load(file)
                    with open(fname_dos, 'rb') as file:
                        phonon_dos = pickle.load(file)
                else:
                    # omega_e, dos_e
                    phonon_en, phonon_dos = self.get_phonon_data(atoms, name, options['calculator'])
                    with open(fname_en, 'wb') as file:
                        pickle.dump(phonon_en, file)
                    with open(fname_dos, 'wb') as file:
                        pickle.dump(phonon_dos, file)

                atoms_opt = read(f'{name}-opt.traj', index=-1)
                formula_units = self.read_table(name, 'formulaunits')
                from ase.thermochemistry import CrystalThermo
                thermo = CrystalThermo(phonon_DOS = phonon_dos,
                                       phonon_energies = phonon_en,
                                       formula_units = formula_units,
                                       potentialenergy = atoms_opt.get_potential_energy())
                A = thermo.get_helmholtz_energy(temperature = options['temperature'],
                                                verbose = options['debug'])
                retval = A * formula_units
            else:
                atoms_opt = read(f'{name}-opt.traj', index=-1)
                retval = atoms_opt.get_potential_energy()
        else:
            RuntimeError(f'phase is unknown: {phase}')
            
        return retval
    
    def get_chemical_potential(self, element_name, species_names, species_pressures, options):
        species = []
        for sp_name in species_names:
            species.append(get_structure(sp_name))

        unique_symbols = list(set([sym for sp in species for sym in sp.get_chemical_symbols()]))
        if len(unique_symbols) != len(species):
            msg = f'''number of elements {len(unique_symbols)} != \
            number of molecules{len(sp)}'''
            raise RuntimeError(msg)

        # build linear system
        free_energies = np.zeros((len(species),1))
        coeffs = np.zeros((len(species), len(unique_symbols))) # should be a square anyways
        for idx, (name, atoms, pressure) in enumerate(zip(species_names, species, species_pressures)):
            options['pressure'] = pressure
            free_energy = self.get_free_energies(atoms,
                                                 name,
                                                 options)

            free_energies[idx] = free_energy

            for jdx in range(len(unique_symbols)):
                coeffs[idx][jdx] = len([at for at in atoms if at.symbol == unique_symbols[jdx]])

        mus = np.linalg.solve(coeffs, free_energies)
        return mus[np.array(unique_symbols) == element_name].item()


if __name__ == '__main__':
    import argparse
    from ase.io import read
    import pickle
    import os
    auto_thermo = auto_thermochemistry()
    # read molecules in
    species = ['CO2', 'H2O', 'HCOOH', 'CuO_bulk'] # species recognized
    elements = ['H', 'C', 'O', 'Cu'] # elements recognized

    description_msg = """A wrapper around ase.thermochemistry, \
    ase.vibrations, and ase.phonon for thermochemistry calculations"""

    parser = argparse.ArgumentParser(description = description_msg)

    parser_helpmsg = """energy engine to do relxation, must be \
    one of reax, nnp, dft"""
    parser.add_argument('-c', '--calc',
                        default='reax',
                        help=parser_helpmsg)

    parser.add_argument('-T', '--temperature', type=float, default=298.0) # 25C
    parser.add_argument('--cache', type=bool, default=True,
                        action=argparse.BooleanOptionalAction,
                        help='attempt to read existing frequencies')
    parser.add_argument('-n', '--ncores', type=int, default=1,
                        help='number of cores to use for parallel calculations')
    parser.add_argument('--debug', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='turn debugging mode on/off')

    subparser_helpmsg = """program operates in either species \
    (free energy) or element (chemical potential) mode"""
    subparsers = parser.add_subparsers(help=subparser_helpmsg, dest='commands')
    parser_species = subparsers.add_parser('species')
    parser_species.add_argument('-P', '--pressure', type=float, default=1e5)


    def check_species_name(name):
        assert name in auto_thermo.thermo_info['name'].values, 'Species name not defined'
        from ase.collections import g2
        import os
        msg = f'(un-optimized) structure cannot be found for {name}'

        _file_found = False
        for ext in auto_thermo.allowed_file_extensions:
            if os.path.isfile(f'{name}.{ext}'):
                _file_found = True

        assert name in g2.names or _file_found, msg
        return name

    parser_species.add_argument('-n', '--name',
                                type=check_species_name, required=True)

    parser_element = subparsers.add_parser('element')

    parser_element.add_argument('-n', '--name', required=True)
    parser_element.add_argument('-s', '--species',
                                type=check_species_name, nargs='+',
                                required=True)
    parser_element.add_argument('-P', '--pressures', nargs='+',
                                type=float, required=True)

    args = parser.parse_args()
    auto_thermo.ncores = args.ncores
    if args.debug:
        import sys
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

    def get_structure(name):
        atoms = None

        for ext in auto_thermo.allowed_file_extensions:
            fname = f'{name}.{ext}'
            if os.path.isfile(fname):
                atoms = read(fname)
                break
        else:
            from ase.build import molecule
            atoms = molecule(name, cell=auto_thermo.pbc_cell)

        return atoms

    if args.commands == 'species':
        name = args.name
        # at this point name is verified to exist
        atoms = get_structure(name)
        assert atoms, 'atoms not initialized correctly'
        options = {'calculator': args.calc,
                   'temperature': args.temperature,
                   'pressure': args.pressure,
                   'cache': args.cache,
                   'debug': args.debug}
        # calculate free energies
        free_energy = auto_thermo.get_free_energies(atoms,
                                                    name,
                                                    options)
        print(f'{name}, {args.temperature}, {args.pressure}, {free_energy}')
    elif args.commands == 'element':
        name = args.name

        # need to specify reference points for the chemical potential
        options = {'calculator': args.calc,
                   'temperature': args.temperature,
                   'cache': args.cache,
                   'debug': args.debug}

        mu = auto_thermo.get_chemical_potential(name,
                                                args.species,
                                                args.pressures,
                                                options)
        print(f'{name}, {args.temperature}, {mu:.4f}')
