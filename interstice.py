#!/usr/bin/env python


import numpy as np
import random
from ase.io import read
from ase.neighborlist import NeighborList
from ase.atoms import Atoms
import sys


def calc_o_position(atoms, chosen_pos, neigh_pos):
    """gives a position of oxygen between 2 Si

    This is not PBC aware! Take care of PBC outside.
    O site on the bisecting plane of the bond with random polar angle.
    """
    # normal vector
    p = neigh_pos - chosen_pos  # (a,b,c)
    # vectors in plane
    u = np.array([p[1], -p[0], 0])  # can check: normal to p
    v = np.cross(p, u)  # normal to p and u; u, v is basis

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    twopi = np.pi * 2
    theta = random.random() * twopi
    d = 0.7860
    site = chosen_pos + p / 2 + (np.cos(theta) * u + np.sin(theta) * v) * d

    return site


def nudge_si_position(atoms, ichosen, chosen_pos, ineigh, neigh_pos):
    # the Si atoms involved need to be moved a bit along the Si-O vector
    for inudge, pos_nudge in zip([ichosen, ineigh], [chosen_pos, neigh_pos]):
        # vector points from O to Si
        pos_O = atoms.positions[-1]

        vnudge = pos_nudge - pos_O
        vnudge /= np.linalg.norm(vnudge)
        dnudge = 1.635 - 1.495  # amount to change Si-O bond length by

        atoms.positions[inudge] += vnudge * dnudge


def get_sites(atoms):
    """Add one oxygen to the interstitial site between each Si-Si bond"""
    rcut = 2.5
    cutoffs = [rcut / 2] * len(atoms)
    atoms_Si = Atoms(
        [at for at in atoms if at.symbol == "Si"], cell=atoms.cell, pbc=True
    )

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)

    nl.update(atoms_Si)

    # matrix = nl.get_connectivity_matrix(sparse=False)
    atoms_ret = []
    dict_offset = {}
    for at in atoms:
        chosen = at.index
        indices, offsets = nl.get_neighbors(chosen)
        for ineigh, offset in zip(indices, offsets):
            key = str(sorted([chosen, ineigh]))
            if key in dict_offset.keys() and np.all(
                np.equal(offset, np.array([0, 0, 0]))
            ):
                print(f"key {key:s} already seen")
                # already seen this before and both in original image
                continue
            else:
                dict_offset[key] = offset

            neigh_pos = atoms.positions[ineigh] + np.dot(
                offset, atoms.get_cell()
            )
            site = calc_o_position(atoms, at.position, neigh_pos)
            atoms_tmp = atoms.copy()#  + Atoms("O", positions=[site])
            atoms_tmp.info['O_site'] = site
            atoms_tmp.info['chosen'] = chosen
            atoms_tmp.info['chosen_pos'] = chosen_pos
            atoms_tmp.info['ineigh'] = ineigh
            atoms_tmp.info['neigh_pos'] = neigh_pos
            
            # nudge_si_position(
            #     atoms_tmp, chosen, at.position, ineigh, neigh_pos
            # )
            atoms_ret.append(atoms_tmp)

    return atoms_ret


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    else:
        fname = "./bulk_2x2x2.vasp"
    conventional = read(fname)
    random.seed(1)
    from ase.visualize import view

    atoms = get_sites(conventional)
    view(atoms)
