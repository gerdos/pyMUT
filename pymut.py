from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import PDBParser, Polypeptide, NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio import SVDSuperimposer
import numpy as np
import os
import logging
from constants import *

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M')

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def load_rotamers(rotamer_loc="{}/rotamers.lib".format(DATA_DIR)):
    """
    Load the Dunbrack rotamer library
    """
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

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


def read_sample_residue(residue_name):
    sample_residue = {}
    with open('{}/{}.pdb'.format(DATA_DIR, residue_name.upper())) as fn:
        for line in fn:
            sample_residue[line[12:16].strip()] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return sample_residue


def is_backbone(atom):
    return atom.get_id() in ['C', 'N', 'CA', 'O']


def select_best_rotamer_based_on_clashes(pdb_object, chain, res_num, mutate_to, sample_residue, rotamers,
                                         skip_own_chain=False):
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
            if vdw_energy > lowest_energy:  # Skip pointless rotamers
                break
            atoms = unfold_entities(pdb_object[0], "A")
            ns = NeighborSearch(atoms)
            close_atoms = ns.search(rotamer_vector, 5)  # 5 Angstrom radii
            for close_atom in close_atoms:
                close_residue = close_atom.get_parent()
                chain_if_close_atom = close_residue.get_parent()
                if close_atom.get_parent().get_id()[
                    1] == res_num and chain_if_close_atom.get_id() == chain:  # Skip itself
                    # print("skipping_own_atom")
                    continue

                if skip_own_chain and chain_if_close_atom.get_id() == chain:
                    # print("Skipping own chain")
                    continue
                if abs(close_residue.get_id()[1] - res_num) == 1 and is_backbone(close_atom):
                    continue
                dist = distance(close_atom.coord, rotamer_vector)
                if dist > 6:
                    continue
                try:
                    vdw_radi = VW_RADII[close_atom.get_parent().get_resname()][close_atom.get_id()] + \
                               VW_RADII[mutate_to][
                                   rotamer_atom]
                except KeyError:
                    continue
                attractive_force = (vdw_radi / dist) ** 6
                vdw_energy += (attractive_force ** 2 - attractive_force)
        if vdw_energy < lowest_energy:
            lowest_energy = vdw_energy
            best_rotamer = rotamer
    return best_rotamer


def mutate(pdb_obj, chain, res_num, mutate_to, rotamer_lib=None, mutation_type="best", verbose='info'):
    level = logging.getLevelName(verbose.upper())
    logging.getLogger().setLevel(level)
    Polypeptide.Polypeptide(
        pdb_obj[0][chain]).get_phi_psi_list()  # This generates the xtra attribute for PHI and PSI angles for residues
    try:
        _residue = pdb_obj[0][chain][res_num]
    except KeyError:
        raise KeyError(f"Residue {res_num} not found in chain {chain}!")
    for atom in list(_residue.get_atoms()):  # Create a copy to remove from
        if not is_backbone(atom):
            atom.parent.detach_child(atom.id)
    phi, psi = [round(np.rad2deg(getattr(_residue.xtra, x, 0)), -1) for x in ['PHI', 'PSI']]

    sample_residue = read_sample_residue(mutate_to)
    starting_points = np.mat([sample_residue["N"], sample_residue["CA"], sample_residue["C"]])
    end_points = np.mat([_residue["N"].coord, _residue["CA"].coord, _residue["C"].coord])

    sup = SVDSuperimposer.SVDSuperimposer()
    sup.set(end_points, starting_points)
    sup.run()
    rot, tran = sup.get_rotran()

    for atom, coords in sample_residue.items():
        sample_residue[atom] = np.squeeze(np.asarray(np.dot(coords, rot) + tran))

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
            selected_rotamer = select_best_rotamer_based_on_clashes(pdb_obj, chain, res_num, mutate_to,
                                                                    sample_residue, rotamer_lib[mutate_to][phi][psi])
        elif mutation_type == "bestother":
            selected_rotamer = select_best_rotamer_based_on_clashes(pdb_obj, chain, res_num, mutate_to,
                                                                    sample_residue, rotamer_lib[mutate_to][phi][psi],
                                                                    skip_own_chain=True)
        else:
            raise ValueError(
                f"Unknown mutation type {mutation_type}. Possible choices are 'first', 'random', 'best', 'bestother'")

        # Introduce the rotamer
        for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
            if mutate_to not in CHI_ANGLES[angle]:
                continue
            dihedral_start = dihedral_from_vectors(
                *[sample_residue[x] for x in CHI_ANGLES[angle][mutate_to]['ref_plane']])
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