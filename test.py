from pymut import *

pdb_obj = PDB('5e0m.pdb')
pdb_obj.mutate(chain='A', mutate_to='TYR', res_num=44, mutation_type='first')
pdb_obj.dump('test.pdb')
