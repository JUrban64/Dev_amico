from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, Selection
from scipy.spatial.distance import cdist
import tempfile
import sys 
import glob
import os
import json

# ============================================================
# Známé kofaktory a jejich funkční skupiny
# ============================================================
COFACTOR_FUNCTIONAL_GROUPS = {
    'NAD': {
        'adenine':       ['C8', 'N9', 'C4', 'C5', 'N7', 'C2', 'N1', 'C6', 'N6', 'N3'],
        'ribose_A':      ['C1B', "C2'", "C3'", "C4'", "O4'", "C5'", "O2'", "O3'"],
        'phosphate':     ['PA', 'O1A', 'O2A', 'O5B', 'O5D', 'PN', 'O1N', 'O2N'],
        'ribose_N':      ['C1D', 'C2D', 'C3D', 'C4D', 'O4D', 'C5D', 'O2D', 'O3D'],
        'nicotinamide':  ['C2N', 'C3N', 'C4N', 'C5N', 'C6N', 'N1N', 'C7N', 'N7N', 'O7N'],
    },
    'FAD': {
        'isoalloxazine': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C4A', 'N5',
                          'C5A', 'C6', 'C7', 'C7M', 'C8', 'C8M', 'C9', 'C9A',
                          'N10', 'C10'],
        'ribitol':       ["C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'"],
        'phosphate':     ['PA', 'O1A', 'O2A', 'O5B', 'O5A', 'PN', 'O1N', 'O2N'],
        'ribose':        ['C1B', 'C2B', 'O2B', 'C3B', 'O3B', 'C4B', 'O4B', 'C5B'],
        'adenine':       ['C2A', 'N1A', 'C6A', 'N6A', 'C5A', 'N7A', 'C8A', 'N9A', 'C4A', 'N3A'],
    },
    'ATP': {
        'adenine':       ['C8', 'N9', 'C4', 'C5', 'N7', 'C2', 'N1', 'C6', 'N6', 'N3'],
        'ribose':        ["C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'"],
        'alpha_P':       ['PA', 'O1A', 'O2A', 'O3A', 'O5B'],
        'beta_P':        ['PB', 'O1B', 'O2B', 'O3B'],
        'gamma_P':       ['PG', 'O1G', 'O2G', 'O3G'],
    },
    'acetyl-CoA': {
        'adenine':       ['C8', 'N9', 'C4', 'C5', 'N7', 'C2', 'N1', 'C6', 'N6', 'N3'],
        'ribose':        ["C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'"],
        'phosphate':     ['PA', 'O1A', 'O2A', 'P3A', 'O3A'],
        'pantothenate':  [],  # variable atom naming
        'cysteamine':    [],
    },
    'B12': {
        # B12 je komplexní, pro účely GNN extrakce zde necháváme prázdný slovník,
        # fallback na 'unknown' element embeddingy se o něj bezpečně postará.
    },
}

# Mapování prvku na index (pro one-hot encoding ligandových uzlů)
LIGAND_ELEMENTS = ['C', 'N', 'O', 'P', 'S', 'H', 'OTHER']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(LIGAND_ELEMENTS)}


class BindingSiteExtractor:
    """
    Extrahuje binding site residues + ligandové atomy z PDB struktury.
    Podporuje tvorbu protein-ligand interakčních grafů.
    """
    
    def __init__(self, distance_threshold=6.0):
        self.threshold = distance_threshold
        self.parser = PDBParser(QUIET=True)
        self.supported_cofactors = list(COFACTOR_FUNCTIONAL_GROUPS.keys())

    def _protein_id_from_path(self, pdb_file):
        """Create stable protein identifier from PDB filename."""
        return os.path.splitext(os.path.basename(pdb_file))[0]

    def _serialize_residue(self, residue):
        """Convert Bio.PDB residue object to JSON-serializable dict."""
        hetflag, resseq, icode = residue.get_id()
        return {
            'chain_id': residue.get_parent().id,
            'resname': residue.get_resname(),
            'hetflag': hetflag,
            'resseq': int(resseq),
            'icode': icode.strip() if isinstance(icode, str) else '',
        }

    def _serialize_binding_site_info(self, binding_site_info):
        """Convert extract_binding_site output to JSON-safe format."""
        serializable = dict(binding_site_info)
        serializable['binding_site_residues'] = [
            self._serialize_residue(r)
            for r in binding_site_info.get('binding_site_residues', [])
        ]
        serializable['contact_map'] = np.asarray(
            binding_site_info.get('contact_map', [])
        ).tolist()

        serialized_ligand_atoms = []
        for atom in binding_site_info.get('ligand_atoms', []):
            serialized_ligand_atoms.append({
                'name': atom['name'],
                'element': atom['element'],
                'coord': np.asarray(atom['coord']).tolist(),
                'functional_group': atom.get('functional_group', 'unknown'),
            })
        serializable['ligand_atoms'] = serialized_ligand_atoms

        serializable['ligand_bonds'] = [
            [int(i), int(j), float(d)]
            for i, j, d in binding_site_info.get('ligand_bonds', [])
        ]

        serializable['protein_ligand_contacts'] = [
            {
                'protein_idx': int(c['protein_idx']),
                'ligand_idx': int(c['ligand_idx']),
                'distance': float(c['distance']),
                'interaction_type': c['interaction_type'],
            }
            for c in binding_site_info.get('protein_ligand_contacts', [])
        ]

        serializable['binding_site_indices'] = [
            int(idx) for idx in binding_site_info.get('binding_site_indices', [])
        ]
        serializable['n_binding_site'] = int(binding_site_info.get('n_binding_site', 0))
        return serializable

    def save_binding_sites(self, binding_sites_by_protein, output_json):
        """Save binding site data keyed by protein_id to a JSON file."""
        serializable = {
            protein_id: self._serialize_binding_site_info(bs_info)
            for protein_id, bs_info in binding_sites_by_protein.items()
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def load_binding_sites(self, input_json):
        """Load binding site data previously saved by save_binding_sites."""
        with open(input_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _preprocess_pdb(self, pdb_file):
        """Fix PDB files with 4-char HETATM residue names (e.g. LIG1 from Boltz).
        
        PDB format allows only 3 characters for residue name (columns 18-20,
        1-indexed). Boltz-docked structures use 4-char names like 'LIG1' which
        shifts chain ID and resSeq by 1 position, causing BioPython PDBParser
        to crash with 'invalid literal for int()'.
        
        If no fix is needed, returns the original path (no temp file created).
        
        Returns:
            (path, needs_cleanup): path to (possibly fixed) PDB file and whether
                                   it's a temp file that should be deleted.
        """
        needs_fix = False
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if (line.startswith('HETATM') or line.startswith('ATOM  ')):
                if len(line) > 21 and line[20] != ' ':
                    needs_fix = True
                    break
        
        if not needs_fix:
            return pdb_file, False
        
        # Fix lines where residue name bleeds into column 20
        fixed_lines = []
        for line in lines:
            if ((line.startswith('HETATM') or line.startswith('ATOM  '))
                    and len(line) > 21 and line[20] != ' '):
                # Remove the extra character at position 20, shift rest left
                line = line[:20] + line[21:]
                # Pad to at least 80 chars to maintain PDB format
                if len(line) < 81:  # 80 chars + newline
                    line = line.rstrip('\n').ljust(80) + '\n'
            fixed_lines.append(line)
        
        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        tmp.writelines(fixed_lines)
        tmp.close()
        return tmp.name, True
    
    def _resolve_ligand_name(self, structure, ligand_name):
        """Resolve the actual residue name of the ligand in the PDB file.
        
        Boltz-docked structures often use generic names like 'LIG1' instead
        of the real cofactor name (e.g. 'NAD'). This method first checks for
        the expected name, then falls back to any HETATM residue that is not
        water (HOH/WAT) and has enough heavy atoms to be a cofactor.
        
        Args:
            structure: Bio.PDB structure
            ligand_name: expected residue name (e.g. 'NAD')
        
        Returns:
            resolved_name: str – the actual residue name found in the PDB
        """
        # 1. Try the expected name first
        if self._get_ligand_coords(structure, ligand_name) is not None:
            return ligand_name
        
        # 2. Collect all HETATM residue names (skip water and common ions)
        skip_names = {'HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO',
                      'ACE', 'NME', 'NH2', 'CL', 'NA', 'MG', 'ZN', 'CA',
                      'FE', 'MN', 'K', 'IOD', 'BR'}
        hetatm_candidates = {}  # name → atom_count
        
        for model_obj in structure:
            for chain in model_obj:
                for residue in chain:
                    het_flag = residue.get_id()[0]
                    if het_flag.startswith('H') or het_flag == 'W':
                        resname = residue.get_resname().strip()
                        if resname not in skip_names:
                            n_atoms = sum(1 for a in residue.get_atoms()
                                          if a.element.strip() != 'H')
                            hetatm_candidates[resname] = (
                                hetatm_candidates.get(resname, 0) + n_atoms
                            )
        
        if not hetatm_candidates:
            raise ValueError(f"Ligand {ligand_name} not found and no HETATM "
                             f"candidates in structure")
        
        # Pick the candidate with the most heavy atoms (most likely the cofactor)
        best_name = max(hetatm_candidates, key=hetatm_candidates.get)
        print(f"    ℹ Ligand '{ligand_name}' nenalezen, použit fallback: "
              f"'{best_name}' ({hetatm_candidates[best_name]} heavy atoms)")
        return best_name

    def _extract_label_from_path(self, pdb_file):
        """Získá jméno ground-truth kofaktoru z názvu složky v cestě k souboru."""
        path_parts = Path(pdb_file).parts
        folder_to_ligand = {
            'NAD': 'NAD',
            'FAD': 'FAD',
            'ATP': 'ATP',
            'COA': 'acetyl-CoA',
            'B12': 'B12'
        }
        for part in path_parts:
            if part in folder_to_ligand:
                return folder_to_ligand[part]
        return 'unknown'

    def extract_binding_site(self, pdb_file, ligand_name=None):
        """
        Args:
            pdb_file: path to PDB file
            ligand_name: (Ignorováno) Ponecháno pro zpětnou kompatibilitu
        
        Returns:
            binding_site_info: dict with binding site data
        """
        # 1. Zjistíme Ground Truth label z názvu složky
        expected_ligand = self._extract_label_from_path(pdb_file)
        # Preprocess PDB to fix 4-char HETATM residue names (Boltz LIG1 etc.)
        parsed_path, needs_cleanup = self._preprocess_pdb(pdb_file)
        try:
            structure = self.parser.get_structure('protein', parsed_path)
        finally:
            if needs_cleanup:
                import os
                os.unlink(parsed_path)
        
        # Resolve actual ligand name (we pass expected_ligand as a hint, but it falls back to largest HETATM if not found)
        actual_ligand_name = self._resolve_ligand_name(structure, expected_ligand)
        
        # Get protein chain (usually chain A)
        model = structure[0]
        protein_chain = None
        for chain in model:
            if len(list(chain.get_residues())) > 10:  # Skip small chains
                protein_chain = chain
                break
        
        if protein_chain is None:
            raise ValueError("No protein chain found")
        
        # Extract full sequence
        full_sequence = self._get_sequence(protein_chain)
        
        # Get ligand coordinates (using resolved name)
        ligand_coords = self._get_ligand_coords(structure, actual_ligand_name)
        
        if ligand_coords is None:
            raise ValueError(f"Ligand {actual_ligand_name} not found")
        
        # Find binding site residues
        binding_site_residues = []
        binding_site_indices = []

        aa_index = 0  # Vlastní počítadlo pro aminokyseliny
        for residue in protein_chain.get_residues():
            if not self._is_aa(residue):
                continue  # Přeskočí vodu/ligandy a nezvedne aa_index
                
            residue_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
            min_dist = cdist(residue_coords, ligand_coords).min()
            
            if min_dist <= self.threshold:
                binding_site_residues.append(residue)
                binding_site_indices.append(aa_index) # Ukládáme správný ESM index
                
            aa_index += 1
        
        # Extract local contact map and get ca_coords
        local_contact_map, ca_coords_list = self._compute_local_contact_map(
            binding_site_residues
        )
        
        # Get sequences
        bs_sequence = ''.join([
            self._three_to_one(res.get_resname()) 
            for res in binding_site_residues
        ])
        
        # Determine true label based PURELY on the folder name
        label = -1
        if expected_ligand in self.supported_cofactors:
            label = self.supported_cofactors.index(expected_ligand)
        else:
            print(f"Warning: Unknown cofactor from path: {expected_ligand} in {pdb_file}")
            
        return {
            'protein_id': self._protein_id_from_path(pdb_file),
            'label': label,
            'full_sequence': full_sequence,
            'binding_site_sequence': bs_sequence,
            'binding_site_indices': binding_site_indices,
            'binding_site_residues': binding_site_residues,
            'contact_map': local_contact_map,
            'protein_coords': ca_coords_list,
            'n_binding_site': len(binding_site_residues),
            'ligand_name': expected_ligand,  # Uložíme 100% jistý název ze složky
            'actual_ligand_name': actual_ligand_name,  # Pouze pro informaci (co našel fyzicky)
            'pdb_file': pdb_file,
            # Ligand info vynecháno (pouze protein)
            'ligand_atoms': [],
            'ligand_bonds': [],
            'protein_ligand_contacts': [],
        }
    
    def _get_ligand_coords(self, structure, ligand_name):
        """Extract ligand atom coordinates"""
        ligand_coords = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() == ligand_name:
                        for atom in residue.get_atoms():
                            ligand_coords.append(atom.get_coord())
        
        if len(ligand_coords) == 0:
            return None
        
        return np.array(ligand_coords)
    
    def _extract_ligand_atoms(self, structure, ligand_name):
        """
        Extrahuje detailní informace o atomech ligandu.
        
        Poznámka: U uměle dockovaných struktur (Boltz apod.) mohou být
        atom names nestandardní (např. C50 místo C8). V takovém případě
        bude functional_group = 'unknown', ale element type a souřadnice
        jsou korektní – to stačí pro výpočet protein-ligand kontaktů
        a rozlišení kvalitních vs špatných interakcí.
        
        Returns:
            ligand_atoms: list of dicts, each with:
                - 'name': atom name (e.g. 'C1', 'N1', 'PA')
                - 'element': element symbol (e.g. 'C', 'N', 'O', 'P')
                - 'coord': np.array [3] coordinates
                - 'functional_group': str or 'unknown' 
                    (e.g. 'adenine', 'nicotinamide', 'phosphate')
        """
        ligand_atoms = []
        
        # Funkční skupiny pro tento kofaktor
        fg_map = COFACTOR_FUNCTIONAL_GROUPS.get(ligand_name, {})
        # Inverzní mapa: atom_name → functional_group
        atom_to_fg = {}
        for fg_name, atom_names in fg_map.items():
            for aname in atom_names:
                atom_to_fg[aname] = fg_name
        
        for model_obj in structure:
            for chain in model_obj:
                for residue in chain:
                    if residue.get_resname() == ligand_name:
                        for atom in residue.get_atoms():
                            atom_name = atom.get_name().strip()
                            element = atom.element.strip() if atom.element else 'X'
                            
                            # Přiřaď funkční skupinu
                            fg = atom_to_fg.get(atom_name, 'unknown')
                            
                            ligand_atoms.append({
                                'name': atom_name,
                                'element': element,
                                'coord': atom.get_coord(),
                                'functional_group': fg,
                            })
        
        return ligand_atoms
    
    def _compute_protein_ligand_contacts(self, binding_site_residues, 
                                          ligand_coords, structure, 
                                          ligand_name, threshold=4.5):
        """
        Spočítá kontakty protein residue ↔ ligand atom.
        
        Args:
            binding_site_residues: list of Bio.PDB residues
            ligand_coords: [n_lig_atoms, 3] np array
            structure: Bio.PDB structure (pro ligand atom info)
            ligand_name: str
            threshold: distance threshold for P-L contact (Å)
        
        Returns:
            contacts: list of dicts:
                {
                    'protein_idx': int (index in binding_site_residues),
                    'ligand_idx': int (index in ligand_atoms list),
                    'distance': float,
                    'interaction_type': str ('hbond_candidate', 'hydrophobic',
                                            'ionic', 'other')
                }
        """
        if ligand_coords is None or len(binding_site_residues) == 0:
            return []
        
        # Extrahuj ligand atomy pro element info
        ligand_atoms = self._extract_ligand_atoms(structure, ligand_name)
        
        # Polar elements (pro odhad typu interakce)
        polar_elements = {'N', 'O', 'S'}
        charged_pos_atoms = {'NZ', 'NH1', 'NH2', 'NE'}  # Lys, Arg
        charged_neg_atoms = {'OD1', 'OD2', 'OE1', 'OE2'}  # Asp, Glu
        hydrophobic_elements = {'C'}
        
        contacts = []
        
        for prot_idx, residue in enumerate(binding_site_residues):
            for prot_atom in residue.get_atoms():
                prot_coord = prot_atom.get_coord()
                prot_element = prot_atom.element.strip() if prot_atom.element else 'X'
                prot_atom_name = prot_atom.get_name().strip()
                
                for lig_idx, lig_atom in enumerate(ligand_atoms):
                    dist = np.linalg.norm(prot_coord - lig_atom['coord'])
                    
                    if dist <= threshold:
                        # Klasifikuj typ interakce
                        lig_element = lig_atom['element']
                        
                        if (prot_element in polar_elements and 
                            lig_element in polar_elements and dist <= 3.5):
                            itype = 'hbond_candidate'
                        elif (prot_atom_name in charged_pos_atoms and 
                              lig_element in {'O', 'P'}):
                            itype = 'ionic'
                        elif (prot_atom_name in charged_neg_atoms and 
                              lig_element in {'N'}):
                            itype = 'ionic'
                        elif (prot_element in hydrophobic_elements and 
                              lig_element in hydrophobic_elements):
                            itype = 'hydrophobic'
                        else:
                            itype = 'other'
                        
                        contacts.append({
                            'protein_idx': prot_idx,
                            'ligand_idx': lig_idx,
                            'distance': float(dist),
                            'interaction_type': itype,
                        })
        
        # Deduplikuj: ponech jen nejkratší kontakt za pár (prot_idx, lig_idx)
        best_contacts = {}
        for c in contacts:
            key = (c['protein_idx'], c['ligand_idx'])
            if key not in best_contacts or c['distance'] < best_contacts[key]['distance']:
                best_contacts[key] = c
        
        return list(best_contacts.values())
    
    def _compute_ligand_bonds(self, ligand_atoms, bond_threshold=1.9):
        """
        Odhadne kovalentní vazby uvnitř ligandu na základě vzdáleností.
        
        Typické délky vazeb:
            C-C: 1.54, C=C: 1.34, C-N: 1.47, C=N: 1.29
            C-O: 1.43, C=O: 1.23, P-O: 1.61, C-S: 1.82
        
        Args:
            ligand_atoms: list of atom dicts (from _extract_ligand_atoms)
            bond_threshold: max distance for covalent bond (Å)
        
        Returns:
            bonds: list of (atom_idx_i, atom_idx_j, distance)
        """
        if len(ligand_atoms) == 0:
            return []
        
        coords = np.array([a['coord'] for a in ligand_atoms])
        dist_matrix = cdist(coords, coords)
        
        bonds = []
        # Minimální vzdálenost 0.5 Å (vyloučí self-loop a artefakty)
        for i in range(len(ligand_atoms)):
            for j in range(i + 1, len(ligand_atoms)):
                d = dist_matrix[i, j]
                if 0.5 < d <= bond_threshold:
                    bonds.append((i, j, float(d)))
        
        return bonds
    
    def _compute_local_contact_map(self, residues, threshold=8.0):
        """
        Compute contact map only within binding site residues
        """
        n = len(residues)
        contact_map = np.zeros((n, n))
        
        # Get CA coordinates
        ca_coords = []
        for res in residues:
            if 'CA' in res:
                ca_coords.append(res['CA'].get_coord())
            else:
                # Fallback to centroid if no CA
                coords = [atom.get_coord() for atom in res.get_atoms()]
                ca_coords.append(np.mean(coords, axis=0))
        
        ca_coords = np.array(ca_coords)
        
        # Compute pairwise distances
        dist_matrix = cdist(ca_coords, ca_coords)
        
        # Contact if distance < threshold
        contact_map = (dist_matrix < threshold).astype(float)
        
        return contact_map, ca_coords.tolist()

    def _is_aa(self, residue):
        """Check if residue is amino acid"""
        return residue.get_id()[0] == ' '
    
    def _get_sequence(self, chain):
        """Extract sequence from chain"""
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = []
        for residue in chain.get_residues():
            if self._is_aa(residue):
                resname = residue.get_resname()
                if resname in three_to_one:
                    sequence.append(three_to_one[resname])
        
        if len(sequence) == 0:
            print(f"    ⚠ Prázdná sekvence z chain {chain.get_id()} "
                  f"({sum(1 for _ in chain.get_residues())} residues celkem)")
        
        return ''.join(sequence)
    
    def _three_to_one(self, three_letter):
        """Convert 3-letter to 1-letter AA code"""
        conversion = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return conversion.get(three_letter, 'X')


# Použití
if __name__ == '__main__':
    import tqdm
    import glob
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Extract binding sites from PDB files.")
    parser.add_argument('--pdb-root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Binding_Sites'),
                        help='Root directory for PDB files.')
    parser.add_argument('--split-file', default=None,
                        help='Optional path to text file containing protein IDs to process (e.g. train.txt).')
    parser.add_argument('--output-json', default='binding_sites_by_protein.json',
                        help='Output JSON file path.')
    # Odstraněn argument --ligand-name, protože se teď zjišťuje ze složky
    args = parser.parse_args()

    extractor = BindingSiteExtractor(distance_threshold=6.0)

    pdb_files = glob.glob(os.path.join(args.pdb_root, '**', '*.pdb'), recursive=True)

    if not pdb_files:
        print(f"No PDB files found under: {args.pdb_root}")
        sys.exit(1)

    if args.split_file:
        print(f"Filtering PDB files using IDs from: {args.split_file}")
        with open(args.split_file, 'r', encoding='utf-8') as f:
            allowed_ids = set(line.strip() for line in f if line.strip())
            
        filtered_files = []
        for pdb_file in pdb_files:
            pid = os.path.splitext(os.path.basename(pdb_file))[0]
            if pid in allowed_ids:
                filtered_files.append(pdb_file)
        pdb_files = filtered_files
        print(f"Filtered down to {len(pdb_files)} matched PDB files")

    print(f"Found {len(pdb_files)} PDB files to process")

    #print(extractor.extract_binding_site(pdb_files[0], ligand_name='NAD'))

    binding_sites_by_protein = {}
    for pdb_file in tqdm.tqdm(pdb_files, desc="Processing PDB files"):
        try:
            bs_info = extractor.extract_binding_site(pdb_file)
            binding_sites_by_protein[bs_info['protein_id']] = bs_info
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    output_json = args.output_json
    extractor.save_binding_sites(binding_sites_by_protein, output_json)
    loaded_binding_sites = extractor.load_binding_sites(output_json)

    print(f"Successfully processed {len(binding_sites_by_protein)} structures")
    print(f"Saved binding sites to {output_json}")
    print(f"Loaded {len(loaded_binding_sites)} proteins from {output_json}")