from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem, rdMolAlign

def calculate_rmsd(custom_pdb_file, reference_pharmacophores):

    # Convert pharmacophores to RDKit molecules
    custom_mol = Chem.MolFromPDBFile(custom_pdb_file)
    reference_mols = [Chem.MolFromPDBFile(pdb_file) for pdb_file in reference_pharmacophores]

    # Perform common substructure search among the custom molecule and reference molecules
    res = rdFMCS.FindMCS([custom_mol] + reference_mols)
    common_substructure = Chem.MolFromSmarts(res.smartsString)

    # Align molecules based on the common substructure
    rmsd_values = []
    for ref_mol in reference_mols:
        ref_match = ref_mol.GetSubstructMatch(common_substructure)
        custom_match = custom_mol.GetSubstructMatch(common_substructure)
        rmsd = rdMolAlign.AlignMol(custom_mol, ref_mol, atomMap=list(zip(custom_match, ref_match)))
        rmsd_values.append(rmsd)

    return rmsd_values

