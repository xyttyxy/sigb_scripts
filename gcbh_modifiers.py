#!/usr/bin/env python
from interstice import get_sites, calc_o_position, nudge_si_position
import ase
import random
from ase.neighborlist import NeighborList
from pygcga.utilities import NoReasonableStructureFound
import numpy as np


def insert_to_interstice(atoms_in, z_min=10, z_max=15):
    """Insert oxygen to silicon interstitial sites"""
    atoms_only_Si = ase.atoms.Atoms(
        [at for at in atoms_in if at.symbol == "Si"], cell=atoms_in.cell, pbc=True
    )

    atoms_candidate = []

    atoms_lst = get_sites(atoms_only_Si)

    for atoms in atoms_lst:
        O_site = atoms.info["O_site"]
        chosen = atoms.info["chosen"]
        ineigh = atoms.info["ineigh"]
        chosen_pos = atoms.info["chosen_pos"]
        neigh_pos = atoms.info["neigh_pos"]

        atoms_tmp = atoms_in.copy() + ase.atoms.Atoms("O", positions=[O_site])

        nudge_si_position(atoms_tmp, chosen, chosen_pos, ineigh, neigh_pos)

        # no other atoms within rcut of O_site

        atoms_candidate.append(atoms_tmp)
    
    atoms_candidate2 = [
        atoms
        for atoms in atoms_candidate
        if atoms[-1].position[2] > z_min and atoms[-1].position[2] < z_max
    ]
    
    # remove the short-distance atoms
    atoms_candidate3 = []
    for atoms in atoms_candidate2:
        distances = atoms.get_distances(len(atoms) - 1, range(len(atoms) - 1))
        if not np.any(distances < 1):
            atoms_candidate3.append(atoms)

    if len(atoms_candidate3) > 0:
        return random.choice(atoms_candidate3)
    else:
        raise NoReasonableStructureFound(
            "Could not find O insertion position far enough away from other atoms"
        )


def rotate_O_around(atoms_in):
    """Rotate O atom around Si-Si bonds"""
    
    o_indices = [at.index for at in atoms_in if at.symbol == "O"]
    o_index = random.choice(o_indices)
    rcut = 1.8
    cutoffs = [rcut / 2] * len(atoms_in)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms_in)
    neigh_indices, neigh_offsets = nl.get_neighbors(o_index)

    # check the neighbor list
    if len(neigh_indices) < 2:
        raise NoReasonableStructureFound(f"O {o_index} does not have enough neighbors")
    else:
        # check at least 2 are Si or Al
        picked_indices = []
        picked_offsets = []
        for index, offset in zip(neigh_indices, neigh_offsets):
            if atoms_in[index].symbol == 'Si' or atoms_in[index].symbol == 'Al':
                picked_indices.append(index)
                picked_offsets.append(offset)
        if len(picked_indices) >= 2:
            neigh_indices = picked_indices
            neigh_offsets = picked_offsets
        else:
            raise NoReasonableStructureFound("O is not unambigously bonded to 2 atoms")
        # select 2
        neigh_indices, neigh_offsets = zip(*random.sample(list(zip(neigh_indices, neigh_offsets)), 2))

    positions = []
    for neigh_index, neigh_offset in zip(neigh_indices, neigh_offsets):
        positions.append(
            atoms_in.positions[neigh_index] + np.dot(neigh_offset, atoms_in.get_cell())
        )

    max_attempts = 10
    for i_attempt in range(max_attempts):
        site = calc_o_position(atoms_in, positions[0], positions[1])
        # delete the original O atom
        atoms_out = atoms_in.copy()
        del atoms_out[o_index]
        atoms_out += ase.atoms.Atoms("O", positions=[site])
        distances = atoms_out.get_all_distances(mic=True)[-1][:-1]
        if not np.any(distances < 1):
            return atoms_out
    
    # do not need to nudge Si atoms since already far apart?
    raise NoReasonableStructureFound(
        "Could not find O insertion position far enough away from other atoms"
    )


def random_perturbation(
    atoms_in,
    displacement_indices=None,
    exclude_elements=None,
    max_displacement=1,
    min_displacement=0.5,
    max_attempts=10,
        z_min = None, z_max=None,
):
    """Randomly displaces atoms

    :param atoms_in: atoms to be perturbed
    :param displacement_indices: atom index to be displaced. default: all atoms
    """
    if displacement_indices:
        selected_indices = displacement_indices
    else:
        
        element_mask = np.ones(len(atoms_in),dtype=bool)
        if exclude_elements:
            for element in exclude_elements:
                element_mask = element_mask & np.array(atoms_in.get_chemical_symbols()) == element

        position_mask = np.ones(len(atoms_in), dtype=bool)
        if z_max and z_min:
            pos_z = atoms_in.get_positions()[:, 2]
            position_mask = (pos_z < z_max) & (pos_z > z_min)

        selected_indices = np.argwhere(element_mask & position_mask)[:,0]

    for i_attempt in range(max_attempts):
        atoms_out = atoms_in.copy()
        for index in selected_indices:
            displacement_vector = (
                np.random.rand(3) * (max_displacement - min_displacement)
                + min_displacement
            )
            atoms_out[index].position += displacement_vector

        # check distances
        distances = atoms_out.get_all_distances()
        if not np.any(np.logical_and(distances > 0.0, distances < 1.4)):
            return atoms_out

    raise NoReasonableStructureFound("All random perturbations failed")


def mutate_SiAl(atoms_in, prob, z_min, z_max):
    """Randomly mutate one Si to Al and Al to Si

    :param prob: 0-1 probability for mutating Si to Al or vice versa
    """

    coin = random.random()
    if coin > prob:
        Al_indices = np.argwhere(np.array(atoms_in.get_chemical_symbols()) == 'Al')[:,0]
        chosen = random.choice(Al_indices)
        atoms_out = atoms_in.copy()
        atoms_out.symbols[chosen] = "Si"
    else:
        element_mask = np.array(atoms_in.get_chemical_symbols()) == 'Si'
        # only consider Al near GB
        pos_z = atoms_in.get_positions()[:, 2]
        position_mask = (pos_z < z_max) & (pos_z > z_min)
        Si_indices = np.argwhere(element_mask & position_mask)[:,0]
        chosen = random.choice(Si_indices)
        atoms_out = atoms_in.copy()
        atoms_out.symbols[chosen] = "Al"
    return atoms_out


def delete_O(atoms_in):
    """Randomly delete one oxygen atom
    """
    O_indices = np.argwhere(np.array(atoms_in.get_chemical_symbols()) == 'O')[:,0]

    chosen = random.choice(O_indices)
    atoms_out = atoms_in.copy()
    del atoms_out[chosen]
    return atoms_out


if __name__ == "__main__":
    from ase.visualize import view

    atoms_in = ase.io.read("1Al4O1.poscar")
    cell = atoms_in.cell
    if np.linalg.norm(cell[0]) > np.linalg.norm(cell[2]):
        # cell needs rotation
        atoms_in.rotate(90, 'y', rotate_cell=True)
        atoms_in.rotate(90, 'z', rotate_cell=True)
        tx = np.linalg.norm(cell[2])
        tz = np.linalg.norm(cell[0])
        atoms_in.translate([tx,0,tz])
        ase.io.write("1Al4O1_rotated.poscar", atoms_in)
        
    n_attempts = 100
    atomss_view = []
    for i_attempt in range(n_attempts):
        # atoms_reted = random_perturbation(atoms_in, z_min = 10, z_max = 15)
        # atoms_reted = rotate_O_around(atoms_in)
        try:
            atoms_reted = insert_to_interstice(atoms_in, z_min=10,z_max=15)
        except NoReasonableStructureFound as ex:
            print(f'failed, continuing, {ex}')
            continue
        
        atomss_view.append(atoms_reted)
    view(atomss_view)
    
