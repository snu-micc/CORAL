import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

ATOM_LIST = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P",
    "H", "B", "Si", "Se",
    "Li", "Na", "K", "Mg", "Ca", "Zn",
]

PAULING_EN = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58,
    "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82, "Sr": 0.95,
    "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10,
    "I": 2.66,
}

HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRAL_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

NODE_DIM = 58
EDGE_DIM = 12
GLOBAL_DIM = 4


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        encoding[choices.index(value)] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def atom_features(atom, partial_charge):
    feats = (
        one_hot(atom.GetSymbol(), ATOM_LIST)
        + one_hot(atom.GetDegree(), list(range(7)))
        + one_hot(atom.GetTotalValence(), list(range(7)))
        + one_hot(atom.GetHybridization(), HYBRID_LIST)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [float(atom.GetFormalCharge())]
        + [float(partial_charge)]
        + one_hot(atom.GetChiralTag(), CHIRAL_LIST)
        + [1.0 if atom.IsInRingSize(s) else 0.0 for s in range(3, 9)]
        + [atom.GetMass() * 0.01]
        + [PAULING_EN.get(atom.GetSymbol(), 0.0)]
    )
    return np.array(feats, dtype=np.float32)


def bond_features(bond):
    feats = (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + one_hot(bond.GetStereo(), STEREO_TYPES)
    )
    return np.array(feats, dtype=np.float32)


def global_features(mol):
    feats = []
    for fn in (lambda m: np.log1p(Descriptors.MolWt(m)),
               lambda m: float(Descriptors.NumHDonors(m)),
               lambda m: float(Descriptors.NumHAcceptors(m)),
               lambda m: float(Descriptors.NumRotatableBonds(m))):
        try:
            feats.append(fn(mol))
        except Exception:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def smiles_to_graph(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    atoms = list(mol.GetAtoms())
    charges = [props.GetMMFFPartialCharge(a.GetIdx()) for a in atoms]
    if len(atoms) != len(charges) or not atoms:
        return None

    x = np.stack([atom_features(a, c) for a, c in zip(atoms, charges)])

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[u, v], [v, u]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_attr = np.stack(edge_attrs)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, EDGE_DIM), dtype=np.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "g": global_features(mol).reshape(1, -1),
    }
