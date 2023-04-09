#!/usr/bin/env python


from ase.io import read, write
from ase.calculators.vasp import Vasp

import shutil
import os
import subprocess
import numpy as np


def set_vasp_key(calc, key, value):
    from ase.calculators.vasp.create_input import float_keys, exp_keys, string_keys, int_keys, bool_keys, list_int_keys, list_bool_keys, list_float_keys, special_keys, dict_keys

    if key in float_keys:
        calc.float_params[key] = value
    elif key in exp_keys:
        calc.exp_params[key] = value
    elif key in string_keys:
        calc.string_params[key] = value
    elif key in int_keys:
        calc.int_params[key] = value
    elif key in bool_keys:
        calc.bool_params[key] = value
    elif key in list_int_keys:
        calc.list_int_params[key] = value
    elif key in list_bool_keys:
        calc.list_bool_params[key] = value
    elif key in list_float_keys:
        calc.list_float_params[key] = value
    elif key in list_float_keys:
        calc.list_float_params[key] = value
    elif key in special_keys:
        calc.special_params[key] = value
    elif key in dict_keys:
        calc.dict_params[key] = value
        
    # some keys need special treatment
    # including kpts, gamma, xc
    if key in calc.input_params.keys():
        calc.input_params[key] = value
        
    
def get_base_calc():
    base_calc = Vasp(
        gga="PE",  # Change this according to the functional used
        lreal="Auto",
        lplane=True,
        lwave=False,
        lcharg=False,
        ncore=4,
        prec="Normal",
        encut=400,
        ediff=1e-6,
        algo="VeryFast",
        ismear=-5,  # warning: need to change if molecule/metal
        gamma=True,
        command=os.getenv("VASP_COMMAND"),
    )

    return base_calc


def cell_opt(atoms, npoints=5, eps=0.04):
    from ase.eos import calculate_eos

    calc = get_base_calc()

    calc.set(ibrion=-1, nsw=0, kpts=atoms.info["kpts"])
    atoms.calc = calc

    eos = calculate_eos(atoms, npoints=npoints, eps=eps, trajectory="eos.traj")

    v, e, B = eos.fit()
    eos.plot(filename="eos.png")
    opt_factor = v / atoms.get_volume()
    atoms.cell = atoms.cell * opt_factor
    write("opted_cell.vasp", atoms)
    return atoms


def axis_opt(atoms, axis, npoints=5, eps=0.04):
    """relax one vector of the cell"""
    kpts = atoms.info["kpts"]
    ens = np.zeros(npoints)
    vols = np.zeros(npoints)

    # by defualt, shrink/expand axis by 4%
    factors = np.linspace(1 - eps, 1 + eps, npoints)

    for ifactor, factor in np.ndenumerate(factors):
        atoms_tmp = atoms.copy()
        atoms_tmp.cell[axis] = atoms.cell[axis] * factor
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, kpts=kpts, directory=f"{factor:.2f}")
        atoms_tmp.calc = calc
        ens[ifactor] = atoms_tmp.get_potential_energy()
        vols[ifactor] = atoms_tmp.get_volume()

    from ase.eos import EquationOfState as EOS

    eos = EOS(volumes=vols, energies=ens, eos="sj")
    v0, e0, B = eos.fit()
    opt_factor = v0 / atoms.get_volume()
    atoms.cell[axis] = atoms.cell[axis] * opt_factor
    write("opted_axis.vasp", atoms)
    return atoms


def geo_opt(atoms, mode="vasp", opt_levels=None, fmax=0.02):
    calc = get_base_calc()
    if not opt_levels:
        # for bulks.
        # other systems: pass in argument
        opt_levels = {
            1: {"kpts": [3, 3, 3]},
            2: {"kpts": [5, 5, 5]},
            3: {"kpts": [7, 7, 7]},
        }

    levels = opt_levels.keys()
    if mode == 'vasp':
        write("CONTCAR", atoms)
        for level in levels:
            level_settings = opt_levels[level]
            # default settings when using built-in optimizer
            set_vasp_key(calc, 'ibrion', 2)
            set_vasp_key(calc, 'ediffg', -1e-2)
            set_vasp_key(calc, 'nsw', 200)
            set_vasp_key(calc, 'nelm', 200)
            # user-supplied overrides
            for key in level_settings.keys():
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp = read("CONTCAR")
            atoms_tmp.calc = calc
            atoms_tmp.get_potential_energy()
            calc.reset()
            atoms_tmp = read("OUTCAR", index=-1)
            shutil.copyfile("CONTCAR", f"opt{level}.vasp")
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
    elif mode == 'ase':
        atoms_tmp = atoms.copy()
        from ase.optimize import BFGS
        # this atoms_tmp is updated when optimizer runs
        for level in levels:
            # default settings when using ase optimizer
            set_vasp_key(calc, 'ibrion', -1)
            set_vasp_key(calc, 'nsw', 0)
            # user-supplied overrides
            level_settings = opt_levels[level]
            for key in level_settings.keys():
                if key in ['nsw', 'ibrion', 'ediffg']:
                    continue
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp.calc = calc
            opt = BFGS(atoms_tmp,
                       trajectory = f"opt{level}.traj",
                       logfile = f"opt{level}.log")
            opt.run(fmax=fmax)
            calc.reset()
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
            
    return atoms_tmp


def freq(atoms, mode="vasp"):
    from ase.constraints import FixAtoms

    calc = get_base_calc()
    if "kpts" in atoms.info.keys():
        kpts = atoms.info["kpts"]
    else:
        kpts = [1, 7, 5]
    calc.set(kpts=kpts)

    if mode == "vasp":
        # avoid this on large structures
        # ncore/npar unusable, leads to kpoint errors
        # isym must be switched off, leading to large memory usage
        calc.set(
            ibrion=5,
            potim=0.015,
            nsw=500,  # as many dofs as needed
            ncore=None,  # avoids error of 'changing kpoints'
            npar=None,
            isym=0,
        )  # turn off symmetry

        atoms.calc = calc
        atoms.get_potential_energy()
        # todo: parse OUTCAR frequencies and modes
    elif mode == "ase":
        # this should be used.
        from ase.vibrations import Vibrations

        calc.set(lwave=True, isym=-1)  # according to michael
        atoms.calc = calc
        constr = atoms.constraints
        constr = [c for c in constr if isinstance(c, FixAtoms)]
        vib_index = [a.index for a in atoms if a.index not in constr[0].index]
        vib = Vibrations(atoms, indices=vib_index)
        vib.run()  # this will save json files
        vib.summary()


def bader(atoms):
    def run_vasp(atoms):
        calc = get_base_calc()

        if "kpts" in atoms.info.keys():
            kpts = atoms.info["kpts"]
        else:
            kpts = [1, 7, 5]

        calc.set(
            ibrion=-1, nsw=0, lorbit=12, lcharg=True, laechg=True, kpts=kpts
        )

        atoms.calc = calc
        atoms.get_potential_energy()
        assert os.path.exists("AECCAR0"), "chgsum.pl: AECCAR0 not found"
        assert os.path.exists("AECCAR2"), "chgsum.pl: AECCAR2 not found"

    def run_bader():
        # add charges
        chgsum = os.getenv("VTST_SCRIPTS") + "/chgsum.pl"
        assert os.path.exists(chgsum), "chgsum not found"
        subprocess.run([chgsum, "AECCAR0", "AECCAR2"], capture_output=True)
        assert os.path.exists("CHGCAR_sum"), "chgsum.pl: CHGCAR_sum not found"

        # run bader
        bader = os.getenv("VTST_BADER")
        assert os.path.exists(bader), "bader not found"
        subprocess.run(
            [bader, "CHGCAR", "-ref", "CHGCAR_sum"], capture_output=True
        )
        assert os.path.exists("ACF.dat"), "bader: ACF.dat not found"

    def read_bader(atoms):
        import pandas as pd

        latoms = len(atoms)
        df = pd.read_table(
            "ACF.dat",
            delim_whitespace=True,
            header=0,
            skiprows=[1, latoms + 2, latoms + 3, latoms + 4, latoms + 5],
        )
        charges = df["CHARGE"].to_numpy()
        n_si = len([a for a in atoms if a.symbol == "Si"])
        n_o = len([a for a in atoms if a.symbol == "O"])
        n_al = len([a for a in atoms if a.symbol == "Al"])

        ocharges = np.array([4] * n_si + [3] * n_al + [6] * n_o)
        dcharges = -charges + ocharges
        atoms.set_initial_charges(np.round(dcharges, 2))

        return atoms

    run_vasp(atoms)

    run_bader()

    atoms_with_charge = read_bader(atoms)
    return atoms_with_charge


class COHP:
    def __init__(self, atoms, bonds, lobsterin_template=None):
        self.atoms = atoms
        self.bonds = bonds

        if lobsterin_template:
            with open(lobsterin_template) as fhandle:
                template = fhandle.readlines()
        else:
            template = [
                "COHPstartEnergy  -22\n",
                "COHPendEnergy     18\n",
                "basisSet          pbeVaspFit2015\n",
                "includeOrbitals   sp\n",
            ]

        self.lobsterin_template = template

    def run_vasp(self):
        atoms = self.atoms
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, isym=-1, prec="Accurate")

        n_si = len([a for a in atoms if a.symbol == "Si"])
        n_o = len([a for a in atoms if a.symbol == "O"])
        n_al = len([a for a in atoms if a.symbol == "Al"])

        nelect = n_si * 4 + n_o * 6 + n_al * 3
        calc.set(nbands=nelect + 20)  # giving 20 empty bands. may require more

        atoms.calc = calc
        atoms.get_potential_energy()

    def write_lobsterin(self):
        lobsterin = "lobsterin"

        with open(f"{lobsterin}", "w+") as fhandle:
            for line in self.lobsterin_template:
                fhandle.write(line)
            for b in self.bonds:
                fhandle.write(f"cohpbetween atom {b[0]+1} and atom {b[1]+1}\n")

    def run_lobster(self):
        lobster = os.getenv("LOBSTER")

        lobster_env = os.environ.copy()
        # typically we avoid using OpenMP, this is an exception
        lobster_env["OMP_NUM_THREADS"] = os.getenv("NSLOTS")

        subprocess.run([lobster], capture_output=True, env=lobster_env)

    def plot(self, cohp_xlim, cohp_ylim, icohp_xlim, icohp_ylim):
        # modded from https://zhuanlan.zhihu.com/p/470592188
        # lots of magic numbers, keep until it breaks down

        def read_COHP(fn):
            raw = open(fn).readlines()
            raw = [line for line in raw if "No" not in line][3:]
            raw = [[eval(i) for i in line.split()] for line in raw]
            return np.array(raw)

        import matplotlib.pyplot as plt

        data_cohp = read_COHP("./COHPCAR.lobster")
        symbols = self.atoms.get_chemical_symbols()
        labels_cohp = [
            f"{symbols[b[0]]}[{b[0]}]-{symbols[b[1]]}[{b[1]}]"
            for b in self.bonds
        ]
        icohp_ef = [
            eval(line.split()[-1])
            for line in open("./ICOHPLIST.lobster").readlines()[1:]
        ]

        data_len = (data_cohp.shape[1] - 3) // 2
        assert (
            len(labels_cohp) == data_len
        ), "Inconsistent bonds definition and COHPCAR.lobster"
        for i in range(data_len):
            fig, ax1 = plt.subplots(figsize=[2.4, 4.8])
            ax1.plot(
                -data_cohp[:, i * 2 + 3],
                data_cohp[:, 0],
                color="k",
                label=labels_cohp[i],
            )
            ax1.fill_betweenx(
                data_cohp[:, 0],
                -data_cohp[:, i * 2 + 3],
                0,
                where=-data_cohp[:, i * 2 + 3] >= 0,
                facecolor="green",
                alpha=0.2,
            )
            ax1.fill_betweenx(
                data_cohp[:, 0],
                -data_cohp[:, i * 2 + 3],
                0,
                where=-data_cohp[:, i * 2 + 3] <= 0,
                facecolor="red",
                alpha=0.2,
            )

            ax1.set_ylim(cohp_ylim)
            ax1.set_xlim(cohp_xlim)
            ax1.set_xlabel("-COHP (eV)", color="k", fontsize="large")
            ax1.set_ylabel("$E-E_\mathrm{F}$ (eV)", fontsize="large")
            ax1.tick_params(axis="x", colors="k")
            # ICOHP
            ax2 = ax1.twiny()
            ax2.plot(-data_cohp[:, i * 2 + 4], data_cohp[:, 0], color="grey")
            ax2.set_ylim(icohp_ylim)  # [-10, 6]
            ax2.set_xlim(icohp_xlim)  # [-0.01, 1.5]
            ax2.set_xlabel("-ICOHP (eV)", color="grey", fontsize="large")
            ax2.xaxis.tick_top()
            ax2.xaxis.set_label_position("top")
            ax2.tick_params(axis="x", colors="grey")

            # legends
            ax1.axvline(0, color="k", linestyle=":", alpha=0.5)
            ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
            labelx = max(icohp_xlim) - 0.05
            labely = max(icohp_ylim) - 0.5

            ax2.annotate(
                labels_cohp[i],
                xy=(labelx, labely),
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="w", alpha=0.5),
            )
            ax2.annotate(
                f"{-icohp_ef[i]:.3f}",
                xy=(labelx, -0.05),
                ha="right",
                va="top",
                color="grey",
            )
            fig.savefig(
                f"cohp-{i+1}.png",
                dpi=500,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()


class NEB:
    def __init__(self, initial, final):
        self.initial = initial
        self.final = final
        self.images = None
        self.kpts = initial.info["kpts"]
        self.fmin = 1e-2

    def interpolate(self, method="linear", nimage=8):
        if method == "linear":
            images = [self.initial]
            images += [self.initial.copy() for i in range(nimage - 2)]
            images += [self.final]

            from ase.neb import NEB

            neb = NEB(images)
            neb.interpolate()
            self.images = images

        elif method == "optnpath":
            types = list(set(self.initial.get_chemical_symbols()))
            template = {
                "nat": len(self.initial),  # Number of atoms
                "ngeomi": 2,  # Number of initial geometries
                "ngeomf": nimage,  # Number of geometries along the path
                "OptReac": False,  # Don't optimize the reactants ?
                "OptProd": False,  # Don't optimize the products
                "PathOnly": True,  # stop after generating the first path
                "AtTypes": types,  # Type of the atoms
                "coord": "mixed",
                "maxcyc": 10,  # Launch 10 iterations
                "IReparam": 1,  # re-distribution of points along the path every 1 iteration
                "SMax": 0.1,  # Max displacement will be 0.1 a.u.
                "ISpline": 5,  # Start using spline interpolation at iteration
                "prog": "VaSP",
            }  # optnpath refuse to work w/o prog tag
            path_fname = "tmp_neb.path"
            with open(path_fname, "w") as fhandle:
                fhandle.write("&path\n")
                for k in template.keys():
                    val = template[k]
                    if isinstance(val, bool):
                        val = str(val)[0]
                    elif isinstance(val, list):
                        val = " ".join(['"' + str(line) + '"' for line in val])
                    elif isinstance(val, str):
                        val = "'" + val + "'"

                    fhandle.write(f"  {k:s}={val},\n")
                fhandle.write("/\n")

            from ase.constraints import FixAtoms

            # a hack around the bug in optnpath:
            # if selective dynamics not used optnpath
            # will repeat 'Cartesian' in the output POSCARs
            self.initial.set_constraint(FixAtoms([]))
            self.final.set_constraint(FixAtoms([]))

            # another hack around optnpath not recognizing
            # POSCAR format with atom counts in VASP5
            is_fname = "tmp_init.vasp"
            fs_fname = "tmp_final.vasp"
            from ase.io import write

            write(is_fname, self.initial, vasp5=False, label="IS")
            write(fs_fname, self.final, vasp5=False, label="FS")

            os.system(f"cat {is_fname} {fs_fname} >> {path_fname}")
            os.remove(is_fname)
            os.remove(fs_fname)
            optnpath = os.getenv("OPTNPATH")

            subprocess.run([optnpath, path_fname], capture_output=True)
            os.remove(path_fname)
            os.remove("Path_cart.Ini")
            os.remove("Path.Ini")
            images = []
            for iimage in range(nimage):
                poscar_fname = f"POSCAR_{iimage:02d}"
                images.append(read(poscar_fname))
                os.remove(poscar_fname)

            self.images = images

    def write_input(self, backend):
        self.backend = backend
        if backend == "ase":
            for image in self.images[1:-2]:
                calc = get_base_calc()
                calc.set(ibrion=-1, nsw=0, kpts=self.kpts)
                image.calc = calc
            print("no input needs to be written for ase backend")

        elif backend == "vtst":
            calc = get_base_calc()

            calc.set(
                ibrion=3, images=len(self.images), lclimb=True, ncore=4, kpar=2
            )

            calc.write_input(self.initial)

            os.remove("POSCAR")
            for iimage in range(len(self.images)):
                workdir = f"{iimage:02d}"
                if not os.path.exists(workdir):
                    os.mkdir(workdir)
                write(f"{workdir}/POSCAR", self.images[iimage])

    def run(self):
        if self.backend == "ase":
            from ase.neb import NEB
            from ase.optimize import BFGS

            neb = NEB(self.images)
            optimizer = BFGS(neb, trajectory="I2F.traj")
            # todo: print out warnings about number of cpus and serial execution
            optimizer.run(fmax=self.fmin)

        elif self.backend == "vtst":
            command = os.getenv("VASP_COMMAND")
            # todo: check number of cpus makes sense
            subprocess.run(command, capture_output=True, shell=True)

    def monitor(self):
        # read the OUTCARs and get their energies.

        runs = []

        # inefficient: reads long long OUTCARs twice
        for iimage in range(1, len(self.images) - 1):
            with open(f"{iimage:02d}/OUTCAR", "r") as fhandle:
                lines = fhandle.readlines()
                run = 0
                for line in lines:
                    if "  without" in line:
                        run += 1
                runs.append(run)

        runs = min(runs)
        nimages = len(self.images)
        energies = np.zeros((runs, nimages))
        for iimage in range(1, len(self.images) - 1):
            run = 0
            with open(f"{iimage:02d}/OUTCAR", "r") as fhandle:
                lines = fhandle.readlines()
                for line in lines:
                    if "  without" in line:
                        energies[run][iimage] = float(line.split()[-1])
                        run += 1
                        if run >= runs:
                            break
        energies[:, 0] = self.initial.get_potential_energy()
        energies[:, -1] = self.final.get_potential_energy()

        import matplotlib.pyplot as plt

        for ien, en in enumerate(energies):
            plt.plot(en, label=str(ien))

        plt.legend()
        plt.savefig("neb_progress.png")


class Dimer:
    def __init__(self, atoms):
        calc = get_base_calc()
        calc.set(
            ibrion=3,
            ediffg=-2e-2,
            ediff=1e-8,
            nsw=500,
            ichain=2,
            potim=0,
            iopt=2,
            kpar=4,
            kpts=atoms.info["kpts"],
        )
        atoms.calc = calc
        self.atoms = atoms

    def run(self):
        self.atoms.get_potential_energy()
        
class gb_sliding:
    def distance(self, atoms, index, neighbor_position):
        dist = ((atoms[index].x - neighbor_position[0])**2 + (atoms[index].y - neighbor_position[1])**2 + (atoms[index].z - neighbor_position[2])**2)**(1/2)
        return dist
    def slide(self, atoms, disp, tolerance): # disp is displacement of the grain per step, tolerance is the minimum distance allowed between 2 atoms.
        disp = disp
        theta = np.arctan(atoms.get_cell()[1][1]/atoms.get_cell()[2][2])
        for atom in atoms:
            if atom.x > 13.8:
                atom.y = atom.y + disp*np.sin(theta)
                atom.z = atom.z + disp*np.cos(theta)
        for atom in atoms:
            if atom.y < 0:
                atom.y = atom.y + 7.73277061
            if atom.y > 7.73277061:
                atom.y = atom.y - 7.73277061
            if atom.z < 0:
                atom.z = atom.z + 13.39355158
            if atom.z > 13.39355158:
                atom.z = atom.z - 13.39355158
        nat_cut = natural_cutoffs(atoms, mult=0.95)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(atoms)
        c = 0
        close_atoms = 0
        for atom in atoms:
            d = np.array([])
            indices, offsets = nl.get_neighbors(c)
            for i, offset in zip(indices, offsets):
                pos = atoms.positions[i] + offset @ atoms.get_cell()    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
                d = np.append(d, self.distance(atoms, c, pos))
            for i in d:
                if i < tolerance:
                    close_atoms = close_atoms + 1
            c = c + 1
        if close_atoms == 0:           
            return atoms
        else:
            self.slide(atoms, disp)
    def analysis(self):
        E = np.array([])
        for i in range(80):
            os.chdir(f"./{i}/")
            atoms = read("OUTCAR")
            E = np.append(E, atoms.get_potential_energy())
            os.chdir("./../")
        return E

if __name__ == "__main__":
    pass
    # for testing, remove before committing
    # do not pollute diffs by commiting
