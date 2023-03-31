from ase.calculators.vasp import Vasp
import numpy as np
import sys
import os
from ase.atoms import Atoms


""" script for Si grain boundary post-calculation analysis
"""


class calculation:
    def __init__(self, path):
        self.atoms = Vasp(restart=True, directory=path).get_atoms()
        self.energy = self.atoms.get_potential_energy()


class reference(calculation):
    """reference calculations

    Parameters:
    ----------
    path: VASP working folder

    formula_units: number of formula units in POSCAR.
    If None, this will be determined based on the number of atoms

    Usage:
    ----------
    1. for a elemental bulk calculation,if the chemical potential
    is desired, leave formula_unit blank

    2. for a bulk binary/ternary alloy, if supercell is desired,
    pass in a fraction to formula_units

    3. for gb/surface that should be used as reference as a whole,
    pass formula_units = 1
    """

    def __init__(self, path, formula_units=None, supercell=None):
        calculation.__init__(self, path)

        if formula_units:
            self.formula_units = formula_units
        else:
            self.formula_units = len(self.atoms)

        if supercell:
            self.atoms = self.atoms.repeat(supercell)
            self.energy *= np.prod(np.array(supercell))
            self.formula_units *= np.prod(np.array(supercell))

    @property
    def ref_en(self):
        """Reference energy per formula unit."""
        return self.energy / self.formula_units


class gb_calc(calculation):
    """general calculation with any combination of elements present

    Parameters:
    ----------
    path: VASP working folder

    elm_refs: elemental references, dictionary containing ref_calc objects.

    gb_ref: grain boundary reference, ref_calc object.
    """

    def __init__(self, path, elm_refs, **kwargs):
        calculation.__init__(self, path)
        self.refs = elm_refs

        count = lambda atoms, sym: len([a for a in atoms if a.symbol == sym])
        ref_sum = lambda refs, counts, syms: sum(
            [refs[sym].ref_en * counts[sym] for sym in syms]
        )
        # if we have a grain boundary reference,
        # the elemental references are only used for the remaining atoms
        symbols = ["Si", "Al", "O"]

        if "gb_ref" in kwargs.keys():
            gb_ref = kwargs["gb_ref"]
            self.gb_ref = gb_ref
            # count remaining atoms
            at_counts = {}
            for sym in symbols:
                at_counts[sym] = count(self.atoms, sym) - count(gb_ref.atoms, sym)

            self.enref = ref_sum(self.refs, at_counts, symbols) + gb_ref.energy
        else:
            for sym in symbols:
                at_counts[sym] = count(self.atoms, ref_sum)
            self.enref = ref_sum(self.refs, at_counts, symbols)

        if "normalization" in kwargs.keys():
            normalization = kwargs["normalization"]
            if isinstance(normalization, int):
                self.normalization = normalization
            elif isinstance(normalization, str):
                self.normalization = count(self.atoms, normalization)
        else:
            self.normalization = count(self.atoms, "Al")

        assert self.normalization != 0, "error: normalization is 0"

    @property
    def endiff(self):
        """Energy of insertion/substitution w.r.t. supplied grain boundary

        Note: not normalized!
        """
        return self.energy - self.enref

    @property
    def endiff_norm(self):
        """endiff Energy normalized by formula_units when available
        or by number of Al in system"""

        return self.endiff / self.normalization

    @property
    def area(self):
        cell = self.atoms.cell
        return np.linalg.norm(np.cross(cell[1], cell[2]))

    @property
    def en_interface(self):
        """interface energy, eV/A^2"""
        return (self.endiff) / self.area / 2


class calc_series:
    """For a series of calculations with same number of atoms but in difference positions

    note: depending on whether gb_ref is passed, calc_series do different things:
    - when no gb_ref supplied, we have a convergence test
    - when yes gb_ref supplied, we are comparing different geometries of the same composition

    """

    def __init__(self, series_name, calc_names, dirs, elm_refs, **kwargs):
        self.srs_name = series_name
        self.calcs = {}
        for n, d in zip(calc_names, dirs):
            self.calcs[n] = gb_calc(d, elm_refs, **kwargs)

    def __str__(self, prop="etotal", wrt_param="name"):
        """
        prop: convergence of what property
        wrt: with respect to what parameter
        bench: which calculation considered to be the benchmark
        """

        if prop == "en_interface":
            properties = [calc.en_interface for key, calc in self.calcs.items()]
        elif prop == "etotal":
            properties = [calc.energy for key, calc in self.calcs.items()]
        elif prop == 'endiff':
            properties = [calc.endiff for key, calc in self.calcs.items()]

        if wrt_param == "name":
            parameters = self.calcs.keys()
        ret_str = [f"{wrt_param:20s}{prop:>10s}\n"]
        for prop, param in zip(properties, parameters):
            ret_str.append(f"{param:20s}{prop:>10.8f}\n")
        return ''.join(ret_str)

    def check_conv(self, prop="en_interface", wrt_param="name"):
        for line in self.__str__(prop, wrt_param):
            print(line)

    def print_all(self, prop="etotal"):
        print(self.__str__(prop, "name"))

    @property
    def min_en_calc(self):
        min_en = sys.float_info.max
        k_min = ""
        for k, c in self.calcs.items():
            if c.en_per_al < min_en:
                min_en = c.en_per_al
                c_min = c
                k_min = k

        return [c_min, k_min]
