# PyMUT

Python3 method to introduce mutations to proteins

### Usage:

Download the library, and place its location inside your $PYTHON_PATH environment variable

    from pymut import *
    pdb_obj = PDB('5e0m.pdb')
    pdb_obj.mutate(chain='A', mutate_to='TYR', res_num=44, mutation_type='first')
    pdb_obj.dump('test.pdb')

The resulting PDB file will contain the mutation. Please note, that resulting PDB file will only contain ATOM
coordinates.

Original residue             |  Mutated residue
:-------------------------:|:-------------------------:
![alt text](assets/original_residue.png)   |  ![alt text](assets/mutated_residue.png)

### Parameters

The `PDB` object requires a basic PDB file location

The `mutate` function takes the following arguments

Parameter             |  Type  | Default | Description | Required
:---------------:|:----------:|:--------:|:--------------:|:---:
chain  |  str | - | Chain identifier of the residue you want to mutate | Yes
resnum  |  int | - | Position of the residue you want to mutate in the PDB file | Yes
mutate_to  |  str | - | 3 letter IUPAC amino acid code (only the 20 standard residues are available) | Yes
rotamer_lib  |  object | None | Preloaded rotamer library returned from the `load_rotamers` function. If not given, it will be loaded separately for each PDB mutation call| No
mutation_type  |  str | 'first' | How to select the rotamer. 3 possible options: `first`: Select the most likely rotamer based on probability in the library. `random`: Select a rotamer randomly based on the probability in the library. `best`: Select the best rotamer based on VdW energy | No

### Dependencies

PyMUT only requires numpy. The current implementation was tested on version 1.19.3

To install numpy on a standard Linux system

    pip3 install numpy

### Before use

Please note, that the PDB file format is *messy*. This algorithm was not designed to work with all crazy PDB files, the PDB parser
inside is a basic one, designed to work with *standard* PDB files. The algorithm assumes continuous neighboring residues, which is required for the dihedral calculations.

N and C terminal residues should be omitted, as the PHI and PSI andles are not defined. The algorithm will assume 0 for the missing degree

### References, licenses

This project is licensed under the terms of the MIT license.

The rotamer library is based on the following article: 

A Smoothed Backbone-Dependent Rotamer Library for Proteins Derived from Adaptive Kernel Density Estimates and Regressions  *Maxim V. Shapovalov Roland L. Dunbrack Jr.*

The rotamer library is licensed by <a href='https://creativecommons.org/licenses/by/4.0/legalcode' target='_blank'>Creative Commons CC BY 4.0 license</a>, and can be freely used, modified and distributed. 





    