import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def load_best_rotamer(rotamer_loc="{}/rotamers.lib".format(DATA_DIR)):
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
            if float(line.split()[8]) < 0.05:
                continue
            _dunbrack[line.split()[0]][int(line.split()[1])][int(line.split()[2])].append({
                'prob': float(line.split()[8]),
                'CHI1': float(line.split()[9]),
                'CHI2': float(line.split()[10]),
                'CHI3': float(line.split()[11]),
                'CHI4': float(line.split()[12])
            })
    return _dunbrack

with open("data/better_5_rotamers", 'w') as fn:
    for i,j in load_best_rotamer().items():
        for k,l in j.items():
            for u,v in l.items():
                for q in v:
                    fn.write(str(i) + '\t' + str(k) + '\t' + str(u) + '\t' + str(q) + '\n')