import pymut
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser, PPBuilder
import numpy as np


def mutate(pdb_obj, chain, res_num, mutate_to, rotamer_lib=None, mutation_type="best"):
    _residue = list(pdb_obj[0][chain].get_residues())[res_num]
    # print(_residue)
    _residue_atoms = list(_residue.get_atoms())
    for atom in _residue_atoms:
        if atom.name not in ['C', 'N', 'CA', 'O']:
            residue = atom.parent
            residue.detach_child(atom.id)
    polypeptide = PPBuilder().build_peptides(pdb_obj[0][chain])
    phi, psi = polypeptide[0].get_phi_psi_list()[res_num]
    phi, psi = round(np.rad2deg(phi), -1), round(np.rad2deg(psi), -1)
    # print(phi, psi)
    # print(_residue['N'].coord)
    sample_residue = pymut.RESIDUE_STRUCTURES[mutate_to]
    starting_points = np.mat([sample_residue["N"], sample_residue["CA"], sample_residue["C"]])
    end_points = np.mat([_residue["N"].coord, _residue["CA"].coord, _residue["C"].coord])
    R, t = pymut.rigid_transform_3D(starting_points, end_points)
    for atom, coords in sample_residue.items():
        sample_residue[atom] = np.squeeze(np.asarray(np.dot(R, coords) + t.T))
    # print(pymut.vector_distance(sample_residue['N'], _residue["N"].coord))
    # print(f"Structure has {len(list(structure.get_atoms()))} atoms")

    # if not rotamer_lib:
    #     rotamer_lib = pymut.load_rotamers()
    # selected_rotamer = sorted(rotamer_lib[mutate_to][phi][psi], key=lambda x: x['prob'], reverse=True)[0]
    selected_rotamer = {'prob': 0.22261, 'CHI1': -66.4, 'CHI2': -178.6, 'CHI3': -179.7, 'CHI4': 179.3}
    # Introduce the rotamer
    for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
        if mutate_to not in pymut.CHI_ANGLES[angle]:
            continue
        dihedral_start = pymut.dihedral_from_vectors(*[sample_residue[x] for x in pymut.CHI_ANGLES[angle][mutate_to]['ref_plane']])
        rotation_angle = dihedral_start - np.deg2rad(selected_rotamer[angle])
        axis = pymut.CHI_ANGLES[angle][mutate_to]['axis']
        # print(angle)
        for atom in pymut.RESIDUE_ORDER[mutate_to][pymut.RESIDUE_ORDER[mutate_to].index(axis[1])+1:]:
            sample_residue[atom] = np.dot(pymut.rotation_matrix(sample_residue[axis[0]] - sample_residue[axis[1]], rotation_angle),
                        sample_residue[atom] - sample_residue[axis[1]]) + sample_residue[axis[1]]
    for atom, coord in sample_residue.items():
        if atom not in ['C', 'N', 'CA', 'O']:
            new_atom = Atom(
                name=atom,
                fullname="{}{}".format(" "*(4-len(atom)), atom),  # for writing the structure, should be 4-char long
                coord=np.asarray(coord),
                bfactor=1.0,
                altloc=" ",
                occupancy=1.0,
                serial_number=9999  # does not matter much, only for writing the struct.
            )
            _residue.add(new_atom)
    io = PDBIO()
    io.set_structure(structure)
    io.save("test.pdb")
    return


parser = PDBParser(QUIET=1)
structure = parser.get_structure("1ctf", "test/5e0m.pdb")
all_atoms = list(structure.get_atoms())
print(f"Structure has {len(all_atoms)} atoms")
print(structure[0]['C'])
mutate(structure, 'A', 1, 'LYS')