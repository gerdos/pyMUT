import argparse
from pymut import mutate
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser

parser = argparse.ArgumentParser(description='Mutate a residue in a PDB file')
parser.add_argument("pdb", help="PDB file")
parser.add_argument("chain", help="Chain identifier of the residue")
parser.add_argument("residue_number", type=int, help="Number of residue to mutate")
parser.add_argument("mutate_to", help="Three letter code of the residue you want to mutate to")
parser.add_argument('-o', "--out_file", required=False, help="Output file name")
parser.add_argument('-m', '--mutation_type',
                    type=str,
                    choices=['first', 'random', 'best', 'bestother'],
                    default='first', required=False,
                    help="How to chose the rotamer.'first': Chose the rotamer with the highest probability. 'random': Chose a random rotamer with weighted probabilities. 'best': Chose the best rotamer based on Van der Waals energies. 'bestother': Same as the 'best' option, but ignore the parent chain")
parser.add_argument('-r', '--rotamer_library',
                    type=str,
                    choices=['all', '0.1', '0.05'],
                    default='all', required=False,
                    help="Which rotamer library to use. 'all' uses all otamers with 0.1 or 0.05 only uses rotamers with higher probability than given")

args = parser.parse_args()

parser = PDBParser(QUIET=1)
structure = parser.get_structure("my_structure", args.pdb)
all_atoms = list(structure.get_atoms())
mutate(structure, args.chain.upper(), args.residue_number, args.mutate_to.upper(), mutation_type=args.mutation_type)

io = PDBIO()
io.set_structure(structure)
fname = args.out_file
if not fname:
    fname = f'{args.pdb.split(".")[0]}_{args.chain}_{args.residue_number}_{args.mutate_to.lower()}'
io.save(fname)

