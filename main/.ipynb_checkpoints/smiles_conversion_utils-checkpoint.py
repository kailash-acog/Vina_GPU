import os
import re
import random
import string
from typing import Optional, List
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw 


try:
    from protonator import protonator
    has_protonator = True
except ImportError:
    protonator, has_protonator = None, False


""" Reads a file with SMILES between two indices """
def read_smi_file(filename: str, i_from: int, i_to: int) -> List[Chem.Mol]:
    mol_list = []
    with open(filename, 'r') as smiles_file:
        for i, line in enumerate(smiles_file):
            if i_from <= i < i_to:
                tokens = line.split()
                smiles = tokens[0]
                mol_list.append(Chem.MolFromSmiles(smiles))
    return mol_list


""" Converts an RDKit molecule (2D representation) to a 3D representation """
def get_structure(mol: Chem.Mol, num_conformations: int, index: int) -> Optional[Chem.Mol]:
    if has_protonator:
        mol = protonator(mol)

    mol = Chem.AddHs(mol)
    new_mol = Chem.Mol(mol)

    conformer_energies = []
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformations, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    conformer_energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, nonBondedThresh=100.0)

    if index == 0:
        i = conformer_energies.index(min(conformer_energies))
    elif index > 0:
        i = index - 1
    else:
        raise ValueError("index cannot be less than zero.")
    
    new_mol.AddConformer(mol.GetConformer(i))
    return new_mol

""" Converts RDKit molecules to structures """
def molecules_to_structure(population: List[Chem.Mol], num_conformations: int, index: int, num_cpus: int):
    with mp.Pool(num_cpus) as pool:
        args = [(p, num_conformations, index) for p in population]
        generated_molecules = pool.starmap(get_structure, args)

        names = [''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) for _ in generated_molecules]
        return generated_molecules, names


""" Saves an RDKit molecule to an SDF file """
def molecule_to_sdf(mol: Chem.Mol, output_filename: str, name: Optional[str] = None):
    if name is not None:
        mol.SetProp("_Name", name)
    writer = Chem.SDWriter(output_filename)
    writer.write(mol)
    writer.close()

