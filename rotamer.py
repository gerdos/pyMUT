from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser, Polypeptide
from Bio import SVDSuperimposer
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

VW_RADII = {
    "ALA": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0
    },
    "CYS": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "SG": 1.8
    },
    "ASP": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "OD1": 1.5,
        "OD2": 1.5
    },
    "GLU": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD": 1.7,
        "OE1": 1.5,
        "OE2": 1.5
    },
    "PHE": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "CD1": 1.9,
        "CD2": 1.9,
        "CE1": 1.9,
        "CE2": 1.9,
        "CZ": 1.9
    },
    "GLY": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4
    },
    "HIS": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "ND1": 1.7,
        "CD2": 1.9,
        "CE1": 1.9,
        "NE2": 1.7
    },
    "ILE": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG1": 2.0,
        "CG2": 2.0,
        "CD1": 2.0
    },
    "LYS": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD": 2.0,
        "CE": 2.0,
        "NZ": 2.0
    },
    "LEU": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD1": 2.0,
        "CD2": 2.0
    },
    "MET": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "SD": 1.8,
        "CE": 2.0
    },
    "ASN": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "OD1": 1.6,
        "ND2": 1.6
    },
    "PRO": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD": 2.0
    },
    "GLN": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD": 1.7,
        "OE1": 1.6,
        "NE2": 1.6
    },
    "ARG": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 2.0,
        "CD": 2.0,
        "NE": 1.7,
        "CZ": 2.0,
        "NH1": 2.0,
        "NH2": 2.0
    },
    "SER": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "OG": 1.6
    },
    "THR": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "OG1": 1.6,
        "CG2": 2.0
    },
    "VAL": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG1": 2.0,
        "CG2": 2.0
    },
    "TRP": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "CD1": 1.9,
        "CD2": 1.7,
        "NE1": 1.7,
        "CE2": 1.7,
        "CE3": 1.9,
        "CZ2": 1.9,
        "CZ3": 1.9,
        "CH2": 1.9
    },
    "TYR": {
        "N": 1.7,
        "CA": 2.0,
        "C": 1.7,
        "O": 1.4,
        "CB": 2.0,
        "CG": 1.7,
        "CD1": 1.9,
        "CD2": 1.9,
        "CE1": 1.9,
        "CE2": 1.9,
        "CZ": 1.7,
        "OH": 1.6
    }
}


CHI_ANGLES = {"CHI1": {'CYS': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'SG']},
                       'ASP': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'SER': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'OG']},
                       'GLN': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'LYS': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'ILE': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG1']},
                       'PRO': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'THR': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'OG1']},
                       'PHE': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'ASN': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'HIS': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'LEU': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'ARG': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'TRP': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'VAL': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG1']},
                       'GLU': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'TYR': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']},
                       'MET': {'axis': ['CA', 'CB'], 'ref_plane': ['N', 'CA', 'CB', 'CG']}},
              "CHI2": {
                  'ASP': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'OD1']},
                  'GLN': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD']},
                  'LYS': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD']},
                  'ILE': {'axis': ['CB', 'CG1'], 'ref_plane': ['CA', 'CB', 'CG1', 'CD1']},
                  'PRO': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD']},
                  'PHE': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD1']},
                  'ASN': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'OD1']},
                  'HIS': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'ND1']},
                  'LEU': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD1']},
                  'ARG': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD']},
                  'TRP': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD1']},
                  'GLU': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD']},
                  'TYR': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'CD1']},
                  'MET': {'axis': ['CB', 'CG'], 'ref_plane': ['CA', 'CB', 'CG', 'SD']},
              },
              "CHI3": {
                  'ARG': {'axis': ['CG', 'CD'], 'ref_plane': ['CB', 'CG', 'CD', 'NE']},
                  'GLN': {'axis': ['CG', 'CD'], 'ref_plane': ['CB', 'CG', 'CD', 'OE1']},
                  'GLU': {'axis': ['CG', 'CD'], 'ref_plane': ['CB', 'CG', 'CD', 'OE1']},
                  'LYS': {'axis': ['CG', 'CD'], 'ref_plane': ['CB', 'CG', 'CD', 'CE']},
                  'MET': {'axis': ['CG', 'SD'], 'ref_plane': ['CB', 'CG', 'SD', 'CE']},
              },
              "CHI4": {
                  'ARG': {'axis': ['CD', 'NE'], 'ref_plane': ['CG', 'CD', 'NE', 'CZ']},
                  'LYS': {'axis': ['CG', 'CE'], 'ref_plane': ['CG', 'CD', 'CE', 'NZ']},
              }
              }

RESIDUE_ORDER = {'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
                 'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
                 'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
                 'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE2', 'OE1'],
                 'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
                 'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
                 'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
                 'THR': ['N', 'CA', 'C', 'O', 'CB', 'CG2', 'OG1'],
                 'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                 'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND2', 'OD1'],
                 'GLY': ['N', 'CA', 'C', 'O'],
                 'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD2', 'ND1', 'CE1', 'NE2'],
                 'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
                 'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
                 'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
                 'ALA': ['N', 'CA', 'C', 'O', 'CB'],
                 'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
                 'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
                 'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                 'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE']}


def load_rotamers(rotamer_loc="{}/rotamers.lib".format(DATA_DIR)):
    _dunbrack = {}
    with open(rotamer_loc) as fn:
        for line in fn:
            if line.startswith("#"):
                continue
            if not line.split()[0] in _dunbrack:
                _dunbrack[line.split()[0]] = {}
            if not int(line.split()[1]) in _dunbrack[line.split()[0]]:
                _dunbrack[line.split()[0]][int(line.split()[1])] = {}
            if not int(line.split()[2]) in _dunbrack[line.split()[0]][int(line.split()[1])]:
                _dunbrack[line.split()[0]][int(line.split()[1])][int(line.split()[2])] = []
            _dunbrack[line.split()[0]][int(line.split()[1])][int(line.split()[2])].append({
                'prob': float(line.split()[8]),
                'CHI1': float(line.split()[9]),
                'CHI2': float(line.split()[10]),
                'CHI3': float(line.split()[11]),
                'CHI4': float(line.split()[12])
            })
    return _dunbrack


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    from scipy.linalg import expm, norm
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def dihedral_from_vectors(v1, v2, v3, v4):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0 * (v2 - v1)
    b1 = v3 - v2
    b2 = v4 - v3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


def select_best_rotemer_based_on_clashes(pdb_object, chain, res_num, mutate_to, sample_residue, rotamers):
    best_rotamer = None
    lowest_energy = float('inf')
    for rotamer in rotamers:
        vdw_energy = 0
        # Introduce the rotamer
        for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
            if mutate_to not in CHI_ANGLES[angle]:
                continue
            dihedral_start = dihedral_from_vectors(
                *[sample_residue[x] for x in CHI_ANGLES[angle][mutate_to]['ref_plane']])
            rotation_angle = dihedral_start - np.deg2rad(rotamer[angle])
            axis = CHI_ANGLES[angle][mutate_to]['axis']
            # print(angle)
            for atom in RESIDUE_ORDER[mutate_to][RESIDUE_ORDER[mutate_to].index(axis[1]) + 1:]:
                sample_residue[atom] = np.dot(
                    rotation_matrix(sample_residue[axis[0]] - sample_residue[axis[1]], rotation_angle),
                    sample_residue[atom] - sample_residue[axis[1]]) + sample_residue[axis[1]]

        for rotamer_atom, rotamer_vector in sample_residue.items():
            for residues in list(pdb_object[0][chain].get_residues()):
                for residue_atoms in list(residues.get_atoms()):
                    if residues.get_id()[1] == res_num:  # Skip itself
                        continue
                    # print(residues.get_id()[1], residue_atoms.get_id())
                    # print(residues.get_resname(), residue_atoms.coord, rotamer_atom, rotamer_vector)
                    dist = distance(residue_atoms.coord, rotamer_vector)
                    if dist > 6:
                        continue
                    try:
                        vdw_radi = VW_RADII[residues.get_resname()][residue_atoms.get_id()] + VW_RADII[mutate_to][rotamer_atom]
                    except KeyError:
                        continue
                    # print(residues.get_id()[1], residue_atoms.get_id(), rotamer_atom, dist, ((vdw_radi / dist) ** 12 - (vdw_radi / dist) ** 6))
                    vdw_energy += ((vdw_radi / dist) ** 12 - (vdw_radi / dist) ** 6)
        # print(rotamer, vdw_energy)
        # print('________________________')
        if vdw_energy < lowest_energy:
            lowest_energy = vdw_energy
            best_rotamer = rotamer
    return best_rotamer


def mutate(pdb_obj, chain, res_num, mutate_to, rotamer_lib=None, mutation_type="best"):
    _residue = [x for x in pdb_obj[0][chain].get_residues() if x.get_id()[1] == res_num][0]
    # print(_residue)
    _residue_atoms = list(_residue.get_atoms())
    for atom in _residue_atoms:
        if atom.name not in ['C', 'N', 'CA', 'O']:
            residue = atom.parent
            residue.detach_child(atom.id)
    polypeptide = Polypeptide.Polypeptide(pdb_obj[0][chain])
    phi, psi = polypeptide.get_phi_psi_list()[res_num]
    phi, psi = round(np.rad2deg(phi), -1), round(np.rad2deg(psi), -1)
    # print(phi, psi)
    # print(_residue['N'].coord)
    sample_residue = {}

    with open('{}/{}.pdb'.format(DATA_DIR, mutate_to.upper())) as fn:
        for line in fn:
            sample_residue[line[12:16].strip()] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    starting_points = np.mat([sample_residue["N"], sample_residue["CA"], sample_residue["C"]])
    end_points = np.mat([_residue["N"].coord, _residue["CA"].coord, _residue["C"].coord])

    sup = SVDSuperimposer.SVDSuperimposer()
    sup.set(end_points, starting_points)
    sup.run()
    rot, tran = sup.get_rotran()

    for atom, coords in sample_residue.items():
        sample_residue[atom] = np.squeeze(np.asarray(np.dot(coords, rot) + tran))
    # print(pymut.vector_distance(sample_residue['N'], _residue["N"].coord))
    # print(f"Structure has {len(list(structure.get_atoms()))} atoms")
    if mutate_to not in ["ALA", "GLY"]:
        if not rotamer_lib:
            rotamer_lib = load_rotamers()

        if mutation_type == 'first':
            selected_rotamer = sorted(rotamer_lib[mutate_to][phi][psi], key=lambda x: x['prob'], reverse=True)[0]
        elif mutation_type == 'random':
            p = np.array([x['prob'] for x in rotamer_lib[mutate_to][phi][psi]])
            p /= p.sum()
            selected_rotamer = np.random.choice(rotamer_lib[mutate_to][phi][psi], p=p)
        elif mutation_type == 'best':
            selected_rotamer = select_best_rotemer_based_on_clashes(pdb_obj, chain, res_num, mutate_to, sample_residue, rotamer_lib[mutate_to][phi][psi])

        # Introduce the rotamer
        for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
            if mutate_to not in CHI_ANGLES[angle]:
                continue
            dihedral_start = dihedral_from_vectors(*[sample_residue[x] for x in CHI_ANGLES[angle][mutate_to]['ref_plane']])
            rotation_angle = dihedral_start - np.deg2rad(selected_rotamer[angle])
            axis = CHI_ANGLES[angle][mutate_to]['axis']
            # print(angle)
            for atom in RESIDUE_ORDER[mutate_to][RESIDUE_ORDER[mutate_to].index(axis[1]) + 1:]:
                sample_residue[atom] = np.dot(
                    rotation_matrix(sample_residue[axis[0]] - sample_residue[axis[1]], rotation_angle),
                    sample_residue[atom] - sample_residue[axis[1]]) + sample_residue[axis[1]]
    for atom, coord in sample_residue.items():
        if atom not in ['C', 'N', 'CA', 'O']:
            new_atom = Atom(
                name=atom,
                element=atom[0],
                fullname="{}{}".format(" " * (4 - len(atom)), atom),  # for writing the structure, should be 4-char long
                coord=np.asarray(coord),
                bfactor=1.0,
                altloc=" ",
                occupancy=1.0,
                serial_number=9999  # does not matter much, only for writing the struct.
            )
            _residue.add(new_atom)
    _residue.resname = mutate_to
    return


parser = PDBParser(QUIET=1)
structure = parser.get_structure("1ctf", "test/5e0m.pdb")
all_atoms = list(structure.get_atoms())
mutate(structure, 'A', 4, 'LYS', mutation_type='random')

io = PDBIO()
io.set_structure(structure)
io.save("test.pdb")
