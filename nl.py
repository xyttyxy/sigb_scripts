from ase.neighborlist import NeighborList, PrimitiveNeighborList

# Si-Si bond = 2.368
# Si = 1.183

# Si-O 1.603
# O = 0.42
# O-Al 1.736
# Al = 1.316
# Al does not like to bond to Si but actual distance around 2.45 in single subbed.
# from this we have 2.49, so ok

from ase.io import read
def_cutoffs = {'Si': 1.183, 'O': 0.42, 'Al': 1.316}
atoms = read('POSCAR')

mult = 1.2
cutoffs = [def_cutoffs[a.symbol]*mult for a in atoms]

Al_index = [a.index for a in atoms if a.symbol == 'Al'][0]

# checking distances make sense
distances = atoms.get_all_distances(mic=True)

for i, d in enumerate(distances[Al_index]):
    if d < cutoffs[i] + cutoffs[Al_index] and i != Al_index:
        print(f'index = {i:3d}, element = {atoms[i].symbol:3s}, actual distance = {d:4.2f}, defined distance {cutoffs[Al_index]+cutoffs[i]:4.2f}')
    

# idk wtf is wrong with this class, did not bother with it
# nl = NeighborList(cutoffs, bothways=True)
# nl = PrimitiveNeighborList(cutoffs, bothways=True)
# nl.update(atoms)
# indices, offsets = nl.get_neighbors(Al_index)
# excluding itself
# indices = [i for i in indices if i != Al_index]

