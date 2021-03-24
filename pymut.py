import numpy as np
import os
import sys
import copy

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class PDB:
    def __init__(self, input_data):
        self.pdb_dct = {}
        self.hetatom_data = ""
        if not os.path.isfile(input_data):
            raise FileNotFoundError()
        file_handler = open(input_data)
        for line in file_handler:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM"):
                if line[16] not in [" ", "A"]:
                    continue
                if line[21] not in self.pdb_dct:
                    self.pdb_dct[line[21]] = {}
                if int(line[22:26].strip()) not in self.pdb_dct[line[21]]:
                    self.pdb_dct[line[21]][int(line[22:26].strip())] = {
                        'RES': line[17:20],
                        "CHAIN": line[21],
                        'ORDER': []
                    }
                self.pdb_dct[line[21]][int(line[22:26].strip())][line[12:16].strip()] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                self.pdb_dct[line[21]][int(line[22:26].strip())]['ORDER'].append(line[12:16].strip())
            elif line.startswith("HETATM"):
                self.hetatom_data += line
        # Check the PDB, remove partial residues
        for chain in self.pdb_dct.keys():
            remove = set()
            for res_num in self.pdb_dct[chain].keys():
                if len(self.pdb_dct[chain][res_num]["ORDER"]) < 4:
                    remove.add(res_num)
                for i in ['C', 'N', 'CA', 'O']:
                    if i not in self.pdb_dct[chain][res_num]:
                        remove.add(res_num)
            for i in remove:
                sys.stderr.write("Partially complete residue residue in chain {} at position {}".format(chain, i))
                self.pdb_dct[chain].pop(i)

        file_handler.close()

    def parse(self):
        return self.pdb_dct

    def copy(self):
        return copy.deepcopy(self)

    def phi_psi(self, chain, res_num):
        if res_num - 1 not in self.pdb_dct[chain]:
            return None, dihedral_from_vectors(self.pdb_dct[chain][res_num]['N'],
                                               self.pdb_dct[chain][res_num]['CA'],
                                               self.pdb_dct[chain][res_num]['C'],
                                               self.pdb_dct[chain][res_num + 1]['N'])
        elif res_num + 1 not in self.pdb_dct[chain]:
            return dihedral_from_vectors(self.pdb_dct[chain][res_num - 1]['C'], self.pdb_dct[chain][res_num]['N'],
                                         self.pdb_dct[chain][res_num]['CA'],
                                         self.pdb_dct[chain][res_num]['C']), None
        else:
            return dihedral_from_vectors(self.pdb_dct[chain][res_num - 1]['C'], self.pdb_dct[chain][res_num]['N'],
                                         self.pdb_dct[chain][res_num]['CA'],
                                         self.pdb_dct[chain][res_num]['C']), dihedral_from_vectors(
                self.pdb_dct[chain][res_num]['N'], self.pdb_dct[chain][res_num]['CA'],
                self.pdb_dct[chain][res_num]['C'], self.pdb_dct[chain][res_num + 1]['N'])

    def introduce_rotamer(self, pdb_object, axis, chi):
        for i in pdb_object['ORDER'][pdb_object['ORDER'].index(axis[1]) + 1:]:
            if i == "H":
                continue
            yield i, np.dot(rotation_matrix(pdb_object[axis[0]] - pdb_object[axis[1]], chi),
                            pdb_object[i] - pdb_object[axis[1]]) + pdb_object[axis[1]]

    def mutate(self, chain, mutate_to, res_num, rotamer_lib=None, mutation_type="best"):
        """
        Mutates amino acid X at position index to a given residue with the respective highest probability rotamer
        :param res_num: Int, position
        :param mutate_to: Residue 3 letter AA code
        :param chain: Chain identifier for the residue
        :param rotamer_lib: Preloaded rotamer library to speed things up
        :param mutation_type: How to introduce the rotamer.
            random: A random rotamer will be used based on the amino acid frequency of the Eukariota proteome
            first: The rotamer with the highest probability will be used
            best: The rotamer with the least clashes will be used (in case of a tie 'first' will be used amongst the winners)
        :return: Mutates the object
        """
        if not rotamer_lib:
            rotamer_lib = load_rotamers()
        phi, psi = [round(np.rad2deg(x), -1) if x else 0 for x in self.phi_psi(chain, res_num)]
        # print(phi, psi)
        coords = self.pdb_dct[chain][res_num]

        # Do the rigid body transformation
        sample_residue = RESIDUE_STRUCTUES[mutate_to]
        starting_points = np.mat([sample_residue["N"], sample_residue["CA"], sample_residue["C"]])
        end_points = np.mat([coords["N"], coords["CA"], coords["C"]])
        R, t = rigid_transform_3D(starting_points, end_points)
        original_cb = coords['CB']
        break_atom = coords['ORDER'][
            max([idx for idx, atom in enumerate(coords['ORDER']) if atom in ['C', 'N', "CA", "O"]])]
        for i in coords['ORDER'][coords['ORDER'].index(break_atom) + 1:]:
            coords.pop(i)
        break_atom = "CB"
        if mutate_to != "GLY":
            # Do the rigid body transformation
            for i in RESIDUE_ORDER[mutate_to][RESIDUE_ORDER[mutate_to].index(break_atom):]:
                coords[i] = np.squeeze(np.asarray(np.dot(R, sample_residue[i]) + t.T))
        if vector_distance(original_cb, coords['CB']) > 0.5:
            sys.stderr.write(
                'RMSD of the transofrmed CB atom is more than 0.5A, which indicates some problem with the rigid body transformation')
        coords['ORDER'] = RESIDUE_ORDER[mutate_to]
        coords['RES'] = mutate_to
        if mutate_to in ["ALA", "GLY"]:
            self.pdb_dct[chain][res_num] = coords
            return
        if mutation_type == 'first':
            selected_rotamer = sorted(rotamer_lib[mutate_to][phi][psi], key=lambda x: x['prob'], reverse=True)[0]
        elif mutation_type == 'random':
            p = np.array([x['prob'] for x in rotamer_lib[mutate_to][phi][psi]])
            p /= p.sum()
            selected_rotamer = np.random.choice(rotamer_lib[mutate_to][phi][psi], p=p)
        elif mutation_type == 'best':
            selected_rotamer = self.select_best_rotemer_based_on_clashes(chain, coords,
                                                                         rotamer_lib[mutate_to][phi][psi])
        else:
            raise ValueError("Mutation type '{}' not valid".format(mutation_type))
        # Introduce the rotamer
        for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
            if mutate_to not in CHI_ANGLES[angle]:
                continue

            dihedral_start = dihedral_from_vectors(
                *[coords[x] for x in CHI_ANGLES[angle][mutate_to]['ref_plane']])
            rotation_angle = dihedral_start - np.deg2rad(selected_rotamer[angle])
            for i in self.introduce_rotamer(coords, CHI_ANGLES[angle][mutate_to]['axis'],
                                            rotation_angle):
                # print(i[0], i[1])
                coords[i[0]] = i[1]

        self.pdb_dct[chain][res_num] = coords

    def generate_all_rotamers(self, chain, mutate_to, res_num, rotamer_lib=None):
        """
        Mutates amino acid X at position index to a given residue wigth the respective highest probability rotamer
        :param res_num: Int, position
        :param mutate_to: Residue 3 letter AA code
        :param chain: Chain identifier for the residue
        :param rotamer_lib: Preloaded rotamer library to speed things up
        :param mutation_type: How to introduce the rotamer.
            random: A random rotamer will be used based on the amino acid frequency of the Eukariota proteome
            first: The rotamer with the highest probability will be used
            best: The rotamer with the least clashes will be used (in case of a tie 'first' will be used amongst the winners)
        :return: Mutates the object
        """
        if not rotamer_lib:
            rotamer_lib = load_rotamers()
        phi, psi = [round(np.rad2deg(x), -1) if x else 0 for x in self.phi_psi(chain, res_num)]
        # print(phi, psi)
        coords = self.pdb_dct[chain][res_num]
        # Do the rigid body transformation
        sample_residue = RESIDUE_STRUCTUES[mutate_to]
        starting_points = np.mat([sample_residue["N"], sample_residue["CA"], sample_residue["C"]])
        end_points = np.mat([coords["N"], coords["CA"], coords["C"]])
        R, t = rigid_transform_3D(starting_points, end_points)
        break_atom = coords['ORDER'][
            max([idx for idx, atom in enumerate(coords['ORDER']) if atom in ['C', 'N', "CA", "O"]])]
        for i in coords['ORDER'][coords['ORDER'].index(break_atom) + 1:]:
            coords.pop(i)

        break_atom = "CB"
        if mutate_to != "GLY":
            # Do the rigid body transformation
            for i in RESIDUE_ORDER[mutate_to][RESIDUE_ORDER[mutate_to].index(break_atom):]:
                coords[i] = np.squeeze(np.asarray(np.dot(R, sample_residue[i]) + t.T))

        coords['ORDER'] = RESIDUE_ORDER[mutate_to]
        coords['RES'] = mutate_to
        if mutate_to in ["ALA", "GLY"]:
            self.pdb_dct[chain][res_num] = coords
            yield {'prob': 1}
            return

        for selected_rotamer in rotamer_lib[mutate_to][phi][psi]:
            # Introduce the rotamer
            for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
                if mutate_to not in CHI_ANGLES[angle]:
                    continue

                dihedral_start = dihedral_from_vectors(
                    *[coords[x] for x in CHI_ANGLES[angle][mutate_to]['ref_plane']])
                rotation_angle = dihedral_start - np.deg2rad(selected_rotamer[angle])
                for i in self.introduce_rotamer(coords, CHI_ANGLES[angle][mutate_to]['axis'],
                                                rotation_angle):
                    coords[i[0]] = i[1]

            self.pdb_dct[chain][res_num] = coords
            yield self

    def select_best_rotemer_based_on_clashes(self, chain, coords, rotamers):
        skip = ['RES', 'CHAIN', 'ORDER']
        best_rotamer = None
        lowest_energy = float('inf')
        for rotamer in rotamers:
            vdw_energy = 0
            for angle in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
                if coords['RES'] not in CHI_ANGLES[angle]:
                    continue

                dihedral_start = dihedral_from_vectors(
                    *[coords[x] for x in CHI_ANGLES[angle][coords['RES']]['ref_plane']])
                rotation_angle = dihedral_start - np.deg2rad(rotamer[angle])
                for i in self.introduce_rotamer(coords, CHI_ANGLES[angle][coords['RES']]['axis'],
                                                rotation_angle):
                    coords[i[0]] = i[1]

            for rotamer_atom, rotamer_vector in coords.items():
                if rotamer_atom in skip:
                    continue
                for pdb_chain, residues in self.pdb_dct.items():
                    if pdb_chain == chain:
                        continue
                    for residue, atoms in residues.items():
                        for atom, vector in atoms.items():
                            if atom in skip:
                                continue
                            # print(atom, vector)
                            dist = self.distance(rotamer_vector, vector)
                            if dist > 6:
                                continue
                            try:
                                vdw_radi = VW_RADII[atoms['RES']][atom] + VW_RADII[coords['RES']][rotamer_atom]
                            except KeyError:
                                continue
                            vdw_energy += ((vdw_radi / dist) ** 12 - (vdw_radi / dist) ** 6)

            if vdw_energy < lowest_energy:
                lowest_energy = vdw_energy
                best_rotamer = rotamer
        return best_rotamer

    def distance(self, x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

    def dump(self, file_name):
        with open(file_name, "w") as fn:
            atom_idx = 1
            for i, j in sorted(self.pdb_dct.items()):
                for idx, res in sorted(j.items()):
                    for atom in res['ORDER']:
                        fn.write("{:<6}{:>5}  {:<4}{} {}{:>4}{:>12.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n".format(
                            'ATOM', atom_idx, atom, res['RES'], i, idx, res[atom][0], res[atom][1], res[atom][2], 1, 1,
                            atom[0]))
                        atom_idx += 1
            fn.write(self.hetatom_data)

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


def rigid_transform_3D(A, B):
    """
    Calculates the transformation that A --> B in the form of B = R*A+t where R is the rotation and t is the translation
    :param A: Set of points in thrre dimensions
    :param B: Set of points in three dimensions
    :return: R, t
    """
    assert len(A) == len(B)
    dim_a = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    a_centered = A - np.tile(centroid_A, (dim_a, 1))
    b_centered = B - np.tile(centroid_B, (dim_a, 1))

    # Get rotation from covariance
    cov_ab = np.transpose(a_centered) * b_centered
    U, S, Vt = np.linalg.svd(cov_ab)
    R = Vt.T * U.T

    # Reflexion
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T
    return R, t


def vector_distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


def vector_length(v):
    return np.sqrt(np.dot(v, v))


def vector_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (vector_length(v1) * vector_length(v2)))


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

CHI5 = {
    'ARG': {'axis': ['NE', 'CZ'], 'ref_plane': ['CD', 'NE', 'CZ', 'NH1']},
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

RESIDUE_STRUCTUES = {'CYS': {'N': np.array([-26.326, 16.318, -20.31]), 'CA': np.array([-26.19, 16.766, -21.675]),
                             'C': np.array([-24.778, 17.254, -21.943]), 'O': np.array([-24.219, 18.032, -21.166]),
                             'CB': np.array([-27.199, 17.891, -21.987]), 'SG': np.array([-27.292, 18.255, -23.71])},
                     'ASP': {'N': np.array([-11.652, 24.945, -23.047]), 'CA': np.array([-10.977, 26.196, -23.38]),
                             'C': np.array([-10.958, 26.406, -24.889]), 'O': np.array([-11.111, 27.526, -25.379]),
                             'CB': np.array([-9.543, 26.184, -22.826]), 'CG': np.array([-9.499, 25.968, -21.315]),
                             'OD1': np.array([-10.439, 26.409, -20.62]), 'OD2': np.array([-8.519, 25.366, -20.82])},
                     'SER': {'N': np.array([-25.432, 19.816, -9.835]), 'CA': np.array([-25.287, 20.019, -11.265]),
                             'C': np.array([-24.222, 21.068, -11.515]), 'O': np.array([-24.21, 22.104, -10.872]),
                             'CB': np.array([-26.617, 20.441, -11.89]), 'OG': np.array([-27.613, 19.458, -11.663])},
                     'GLN': {'N': np.array([-16.803, 23.582, -19.627]), 'CA': np.array([-16.613, 24.103, -20.962]),
                             'C': np.array([-15.135, 24.044, -21.333]), 'O': np.array([-14.298, 24.569, -20.615]),
                             'CB': np.array([-17.131, 25.538, -21.048]), 'CG': np.array([-16.904, 26.189, -22.39]),
                             'CD': np.array([-17.672, 25.497, -23.482]), 'NE2': np.array([-16.98, 25.122, -24.556]),
                             'OE1': np.array([-18.87, 25.273, -23.351])},
                     'LYS': {'N': np.array([-30.291, 9.809, -19.561]), 'CA': np.array([-30.42, 9.558, -18.123]),
                             'C': np.array([-29.462, 8.444, -17.749]), 'O': np.array([-29.216, 7.554, -18.566]),
                             'CB': np.array([-31.835, 9.14, -17.744]), 'CG': np.array([-32.887, 10.213, -17.779]),
                             'CD': np.array([-34.114, 9.684, -17.043]), 'CE': np.array([-35.173, 10.718, -16.814]),
                             'NZ': np.array([-36.3, 10.088, -16.079])},
                     'ILE': {'N': np.array([-24.723, 16.543, -29.892]), 'CA': np.array([-25.444, 16.623, -28.645]),
                             'C': np.array([-25.674, 15.218, -28.118]), 'O': np.array([-26.231, 14.358, -28.802]),
                             'CB': np.array([-26.778, 17.379, -28.82]), 'CG1': np.array([-26.505, 18.786, -29.352]),
                             'CG2': np.array([-27.537, 17.452, -27.496]), 'CD1': np.array([-27.739, 19.557, -29.731])},
                     'PRO': {'N': np.array([-36.343, 14.149, -13.933]), 'CA': np.array([-34.902, 13.943, -13.726]),
                             'C': np.array([-34.509, 12.462, -13.765]), 'O': np.array([-35.401, 11.629, -13.634]),
                             'CB': np.array([-34.658, 14.531, -12.336]), 'CG': np.array([-35.781, 15.472, -12.106]),
                             'CD': np.array([-36.956, 14.921, -12.84])},
                     'THR': {'N': np.array([-14.821, 23.412, -22.456]), 'CA': np.array([-13.449, 23.345, -22.924]),
                             'C': np.array([-12.94, 24.73, -23.292]), 'O': np.array([-13.695, 25.574, -23.784]),
                             'CB': np.array([-13.299, 22.428, -24.145]), 'CG2': np.array([-13.488, 20.985, -23.756]),
                             'OG1': np.array([-14.267, 22.779, -25.142])},
                     'PHE': {'N': np.array([-28.948, 12.065, -22.096]), 'CA': np.array([-29.289, 10.767, -21.543]),
                             'C': np.array([-29.581, 10.823, -20.052]), 'O': np.array([-29.196, 11.771, -19.371]),
                             'CB': np.array([-28.145, 9.773, -21.818]), 'CG': np.array([-26.907, 10.008, -20.986]),
                             'CD1': np.array([-25.985, 10.979, -21.338]), 'CD2': np.array([-26.662, 9.245, -19.866]),
                             'CE1': np.array([-24.858, 11.195, -20.579]), 'CE2': np.array([-25.519, 9.455, -19.098]),
                             'CZ': np.array([-24.624, 10.427, -19.458])},
                     'ASN': {'N': np.array([-16.126, 18.285, -30.794]), 'CA': np.array([-15.11, 17.243, -30.694]),
                             'C': np.array([-15.056, 16.583, -29.33]), 'O': np.array([-14.502, 17.138, -28.38]),
                             'CB': np.array([-13.725, 17.789, -31.001]), 'CG': np.array([-12.668, 16.705, -30.918]),
                             'ND2': np.array([-12.572, 15.896, -31.959]), 'OD1': np.array([-11.986, 16.568, -29.906])},
                     'GLY': {'N': np.array([-27.987, 5.028, -16.309]), 'CA': np.array([-28.389, 3.676, -15.973]),
                             'C': np.array([-27.596, 3.159, -14.789]), 'O': np.array([-27.395, 3.875, -13.804])},
                     'HIS': {'N': np.array([-33.674, 6.051, -22.758]), 'CA': np.array([-33.378, 7.431, -22.384]),
                             'C': np.array([-31.98, 7.83, -22.822]), 'O': np.array([-31.148, 8.217, -22.005]),
                             'CB': np.array([-33.533, 7.618, -20.876]), 'CG': np.array([-34.909, 7.309, -20.38]),
                             'CD2': np.array([-36.002, 8.099, -20.234]), 'ND1': np.array([-35.298, 6.044, -20.004]),
                             'CE1': np.array([-36.565, 6.067, -19.627]), 'NE2': np.array([-37.013, 7.301, -19.761])},
                     'LEU': {'N': np.array([-27.499, 13.371, -24.957]), 'CA': np.array([-28.489, 13.605, -23.928]),
                             'C': np.array([-28.994, 12.27, -23.41]), 'O': np.array([-29.405, 11.422, -24.191]),
                             'CB': np.array([-29.632, 14.458, -24.506]), 'CG': np.array([-30.873, 14.677, -23.655]),
                             'CD1': np.array([-30.546, 15.523, -22.45]), 'CD2': np.array([-31.952, 15.333, -24.527])},
                     'ARG': {'N': np.array([-25., 17.353, -7.23]), 'CA': np.array([-25.084, 18.721, -7.717]),
                             'C': np.array([-24.902, 18.766, -9.226]), 'O': np.array([-24.302, 17.875, -9.827]),
                             'CB': np.array([-24.031, 19.603, -7.035]), 'CG': np.array([-22.614, 19.199, -7.349]),
                             'CD': np.array([-21.587, 19.98, -6.545]), 'NE': np.array([-20.256, 19.904, -7.161]),
                             'CZ': np.array([-19.624, 20.917, -7.755]), 'NH1': np.array([-20.174, 22.125, -7.806]),
                             'NH2': np.array([-18.42, 20.727, -8.287])},
                     'TRP': {'N': np.array([-31.344, 14.013, -16.124]), 'CA': np.array([-30.675, 14.016, -17.428]),
                             'C': np.array([-29.308, 14.68, -17.398]), 'O': np.array([-29.009, 15.483, -16.515]),
                             'CB': np.array([-31.526, 14.725, -18.471]), 'CG': np.array([-32.818, 14.062, -18.718]),
                             'CD1': np.array([-33.951, 14.167, -17.961]), 'CD2': np.array([-33.129, 13.168, -19.788]),
                             'CE2': np.array([-34.476, 12.773, -19.624]), 'CE3': np.array([-32.407, 12.667, -20.877]),
                             'NE1': np.array([-34.951, 13.391, -18.497]), 'CZ2': np.array([-35.107, 11.899, -20.504]),
                             'CZ3': np.array([-33.044, 11.803, -21.758]), 'CH2': np.array([-34.379, 11.426, -21.562])},
                     'ALA': {'N': np.array([-22.643, 18.456, -32.033]), 'CA': np.array([-23.103, 17.168, -31.547]),
                             'C': np.array([-23.671, 17.301, -30.151]), 'O': np.array([-23.155, 18.043, -29.318]),
                             'CB': np.array([-21.968, 16.16, -31.557])},
                     'VAL': {'N': np.array([-23.024, 20.829, -34.622]), 'CA': np.array([-22.855, 20.57, -33.196]),
                             'C': np.array([-23.408, 19.209, -32.819]), 'O': np.array([-24.513, 18.851, -33.205]),
                             'CB': np.array([-23.54, 21.654, -32.355]), 'CG1': np.array([-23.185, 21.489, -30.888]),
                             'CG2': np.array([-23.148, 23.039, -32.848])},
                     'GLU': {'N': np.array([-29.901, 4.654, -18.94]), 'CA': np.array([-31.146, 3.945, -18.728]),
                             'C': np.array([-31.73, 3.598, -20.087]), 'O': np.array([-31.691, 4.413, -21.013]),
                             'CB': np.array([-32.113, 4.81, -17.918]), 'CG': np.array([-33.412, 4.132, -17.557]),
                             'CD': np.array([-34.356, 5.054, -16.805]), 'OE1': np.array([-33.928, 6.15, -16.383]),
                             'OE2': np.array([-35.534, 4.678, -16.637])},
                     'TYR': {'N': np.array([-26.64, 15.205, -34.085]), 'CA': np.array([-26.135, 16.476, -34.55]),
                             'C': np.array([-27.272, 17.429, -34.878]), 'O': np.array([-28.297, 17.023, -35.423]),
                             'CB': np.array([-25.289, 16.283, -35.8]), 'CG': np.array([-23.852, 15.942, -35.525]),
                             'CD1': np.array([-23.48, 14.65, -35.193]), 'CD2': np.array([-22.865, 16.912, -35.609]),
                             'CE1': np.array([-22.145, 14.329, -34.941]), 'CE2': np.array([-21.531, 16.605, -35.358]),
                             'CZ': np.array([-21.184, 15.311, -35.034]), 'OH': np.array([-19.859, 15.006, -34.794])},
                     'MET': {'N': np.array([-39.961, 11.67, -27.989]), 'CA': np.array([-38.933, 12.289, -27.17]),
                             'C': np.array([-37.738, 12.695, -28.037]), 'O': np.array([-37.124, 13.737, -27.807]),
                             'CB': np.array([-38.517, 11.331, -26.058]), 'CG': np.array([-37.717, 11.965, -24.939]),
                             'SD': np.array([-37.139, 10.736, -23.755]), 'CE': np.array([-38.682, 10.189, -23.033])}}

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
