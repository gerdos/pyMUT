from pymut import *
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser

parser = PDBParser()
io = PDBIO()

structure = parser.get_structure("my_structure", "5e0m.pdb")

rotamer_lib = load_rotamers("{}/rotamers.lib".format(DATA_DIR))  # Preload the rotamer library to increase speed

# TEST 1: Introducing a mutation to a simple PDB
mutate(structure, chain='A', mutate_to='TYR', res_num=44, mutation_type='best', rotamer_lib=rotamer_lib, verbose='debug')
# io.set_structure(structure)
# io.save("test1.pdb")
#
# # TEST 2: Create an alanine scan for the ligand (chain C) in 5e0m
# for pos in range(468, 477):
#     mutate(structure, chain='C', mutate_to='ALA', res_num=pos, rotamer_lib=rotamer_lib)
#     io.set_structure(structure)
#     io.save("test2.pdb")


