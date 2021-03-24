from pymut import *

pdb_obj = PDB('5e0m.pdb')
rotamer_lib = load_rotamers("{}/rotamers.lib".format(DATA_DIR))  # Preload the rotamer library to increase speed

# TEST 1: Introducing a mutation to a simple PDB
test1_pdb = pdb_obj.copy()
test1_pdb.mutate(chain='A', mutate_to='TYR', res_num=44, mutation_type='first', rotamer_lib=rotamer_lib)
test1_pdb.dump('test.pdb')

# TEST 2: Create an alanine scan for the ligand (chain C) in 5e0m
for pos in pdb_obj.parse()['C'].keys():
    test2_pdb = pdb_obj.copy()
    test2_pdb.mutate(chain='C', mutate_to='ALA', res_num=pos, rotamer_lib=rotamer_lib)
    test2_pdb.dump('ALA_SCAN_{}.pdb'.format(pos))

# TEST 3: Generate all rotamers for a given position
test3_pdb = pdb_obj.copy()
for n, rot in enumerate(test3_pdb.generate_all_rotamers('A', 'TYR', 44, rotamer_lib=rotamer_lib)):
    test3_pdb.dump('44_TYR_{}_5e0m.pdb'.format(n))


