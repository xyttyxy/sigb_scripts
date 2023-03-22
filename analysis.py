from ase.calculators.vasp import Vasp
import numpy as np
import sys
import os
from ase.atoms import Atoms


""" script for Si grain boundary post-calculation analysis
"""


class ref_calc:
    """Single element reference calculations

    Also serves as the base class for other calculations

    Parameters:
    ----------
    path: VASP working folder

    formula_units: number of formula units in POSCAR.
    If None, this will be determined based on the number of atoms

    """

    def __init__(self, path, formula_units=None):
        self.atoms = Vasp(restart=True, directory=path).get_atoms()
        self.energy = self.atoms.get_potential_energy()

        if formula_units:
            self.formula_units = formula_units
        else:
            self.formula_units = len(self.atoms)

    @property
    def ref_en(self):
        return self.energy / self.formula_units


class gb_conv(ref_calc):
    """Single Si-only GB calculation

    Note: this is intended for convergence w.r.t. to
    cell size. Check if this makes sense when used for
    calculation parameter convergence e.g. ENCUT and KPTS
    """

    def __init__(self, path, ref, formula_units=None):
        ref_calc.__init__(self, path, formula_units)
        self.ref = ref

    @property
    def area(self):
        cell = self.atoms.cell
        return np.linalg.norm(np.cross(cell[1], cell[2]))

    @property
    def en_interface(self):
        """interface energy, eV/A^2"""
        return (
            (self.energy - self.ref.ref_en * self.formula_units)
            / self.area
            / 2
        )


class real_calc(ref_calc):
    """Used for storing a 'real' calculation.

    Parameters:
    ----------
    path: VASP working folder

    elm_refs: elemental references, dictionary containing ref_calc objects.

    gb_ref: grain boundary reference, ref_calc object.
    """

    def __init__(self, path, elm_refs, gb_ref=None, formula_units=None):
        ref_calc.__init__(self, path, formula_units)
        self.refs = elm_refs

        self.at_counts = {}

        if gb_ref:
            self.gb_at_counts = {}
            self.gb_ref = gb_ref

        for k in ["Si", "Al", "O"]:
            self.at_counts[k] = len([a for a in self.atoms if a.symbol == k])
            if gb_ref:
                self.gb_at_counts[k] = len(
                    [a for a in self.gb_ref.atoms if a.symbol == k]
                )

        if formula_units:
            self.user_normalized = True
        else:
            self.user_normalized = False

    @property
    def endiff(self):
        """Energy of insertion/substitution w.r.t. supplied grain boundary

        Note: not normalized!
        """
        elms = ["Si", "Al", "O"]
        if hasattr(self, "gb_ref"):
            dn = {}
            for k in elms:
                dn[k] = self.at_counts[k] - self.gb_at_counts[k]

            endiff = self.energy - (
                sum([self.refs[k].ref_en * dn[k] for k in elms])
                + self.gb_ref.energy
            )
        else:
            endiff = self.energy - sum(
                [self.refs[k].ref_en * self.at_counts[k] for k in elms]
            )

        return endiff

    @property
    def en_per_al(self):
        """endiff Energy normalized by formula_units when available
        or by number of Al in system"""
        if self.user_normalized:
            return self.endiff / self.formula_units
        else:
            return self.endiff / self.at_counts["Al"]


class calc_series:
    """For a series of calculations with same number of atoms but in difference positions

    note: depending on whether gb_ref is passed, calc_series do different things:
    - when no gb_ref supplied, we have a convergence test
    - when yes gb_ref supplied, we are comparing different geometries of the same composition

    """

    def __init__(self, srs_name, calc_names, dirs, elm_refs, gb_ref=None):
        self.srs_name = srs_name
        self.calcs = {}
        for n, d in zip(calc_names, dirs):
            if gb_ref:
                self.conv = False
                self.calcs[n] = real_calc(d, elm_refs, gb_ref)
            else:
                self.conv = True
                self.calcs[n] = gb_conv(d, elm_refs["Si"])

    def check_conv(self):
        if not self.conv:
            raise Exception("Convergence is not defined on real_calc series")
        for k, c in self.calcs.items():
            print(
                f"conv test name = {k:10s}, interface energy = {c.en_interface:.2e} eV/A^2"
            )

    def print_all(self):
        for k, c in self.calcs.items():
            print(
                f"conv test name = {k:10s}, interstitial energy = {c.en_per_al:.2e} eV / Al"
            )

    @property
    def min_en_calc(self):
        if self.conv:
            raise Exception(
                "Wrong! Do not check minimum energy structure on convergence tests"
            )

        min_en = sys.float_info.max
        k_min = ""
        for k, c in self.calcs.items():
            if c.en_per_al < min_en:
                min_en = c.en_per_al
                c_min = c
                k_min = k

        return [c_min, k_min]


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore", message="Magnetic moment data not written in OUTCAR"
    )
    # do not assume any folder structure except for everything being under root
    root_dir = "/u/scratch/b/bsreehar/SCALAR"
    os.chdir(root_dir)

    elm_refs = {
        "Al": ref_calc("Primitive/Sigma3_GB_Primitive/Al_bulk/"),
        "Si": ref_calc("Si_bulk_Conventional/Lattice_optimization/"),
        "O": ref_calc("Sigma3_GB_Conventional/O2/geo-opt/", 2),
    }

    mode = "AlO"
    if mode == "Sigma3_layers":
        # convergence w.r.t. to number of layers for the Sigma 3 111 GB
        calcs_base = "Sigma3_GB_Conventional/"
        layers = range(3, 7)
        calc_dirs = [f"{calcs_base}/{layer:d}_layers" for layer in layers]
        calc_dirs[1] += "/geo-opt"
        sigma3_layer_conv = calc_series(
            srs_name="Sigma3_layers_convergence",
            calc_names=["3", "4", "5", "6"],
            dirs=calc_dirs,
            elm_refs=elm_refs,
        )
        sigma3_layer_conv.check_conv()

        # 4 layer is sufficient.
    elif mode == "single_calc":
        # tests for individual calculations
        calc_dirs = [
            f"Sigma3_GB_Conventional/Interstice_Al/",
            f"Sigma3_GB_Conventional/Substitution_Al/1_Al/4_layers/geo-opt",
        ]
        # grain boundary reference structure: the pristine gb
        gb_ref = ref_calc(f"Sigma3_GB_Conventional/4_layers/geo-opt/", 1)
        for d in calc_dirs:
            c = real_calc(d, elm_refs, gb_ref)
            print(d, c.en_per_al)

    elif mode == "1Al":
        gb_ref = gb_conv(
            "Sigma3_GB_Conventional/4_layers/geo-opt/", elm_refs["Si"], 1
        )
        base_dir = "Sigma3_GB_Conventional/Substitution_Al/2_Al/"
        calc_dirs = ["1", "2", "3"]
        calc_dirs = [f"{base_dir}/{d}" for d in calc_dirs]
        sub_2al = calc_series(
            srs_name="substitutional_2Al",
            calc_names=["1", "2", "3"],
            dirs=calc_dirs,
            elm_refs=elm_refs,
            gb_ref=gb_ref,
        )
        ck_min = sub_2al.min_en_calc
        print(ck_min[0].en_per_al)
    else:
        gb_ref = gb_conv(
            "Sigma3_GB_Conventional/4_layers/geo-opt/", elm_refs["Si"], 1
        )
        # gb_ref = gb_conv(f'Sigma3_GB_Conventional/Interstice_O/geo-opt4/', elm_refs['Si'], 1)
        base_dir = "Sigma3_GB_Conventional/Substitution_Al_Interstice_O/"
        calc_dirs = [
            "1",
            "1_Al_2_O_1",
            "1_Al_2_O_2",
            "1_Al_3_O_1",
            "1_Al_3_O_2",
            "1_Al_4_O_1",
            "1_Al_4_O_2",
            "2",
        ]
        calc_dirs = [f"{base_dir}/{d}" for d in calc_dirs]

        calc_names = [
            "O close",
            "2O close",
            "2O 1O far",
            "3O close",
            "3O 1O far",
            "4O close",
            "4O 1O far",
            "O far",
        ]

        sub_2al = calc_series(
            srs_name="substitutional_2Al",
            dirs=calc_dirs,
            calc_names=calc_names,
            elm_refs=elm_refs,
            gb_ref=gb_ref,
        )

        # can append to calcs directly if need finer control
        # bulk-based calculations
        base_dir = "Si_bulk_Conventional/"
        calc_dirs = [
            "Interstice_O",
            "Interstice_Al",
            "Substitution_O",
            "Substitution_Al",
            "Substitution_Al_Interstice_O",
        ]
        calc_dirs = [f"{base_dir}/{d}" for d in calc_dirs]
        calc_names = [c.split("/")[-1] for c in calc_dirs]

        for n, d in zip(calc_names, calc_dirs):
            sub_2al.calcs[n] = real_calc(d, elm_refs, formula_units=1)

        sub_2al.print_all()
