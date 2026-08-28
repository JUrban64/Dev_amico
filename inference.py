import os
import argparse
import subprocess
import torch
from pathlib import Path
from glob import glob

from data_prep.esm2_feature_ex import ESMFeatureExtractor
from data_prep.build_p2rank_dataset import get_ca_coords_and_seq
from model import AttentionMIL_ESM
from Bio.PDB import PDBParser

def run_p2rank(pdb_path, prank_exec="p2rank_2.5.1/prank"):
    """Spustí P2Rank na jednom PDB souboru a vrátí cestu ke složce s výsledky."""
    pdb_path = Path(pdb_path)
    out_dir = pdb_path.with_suffix("").name + "_prank_output"
    out_dir = Path("./temp_inference") / out_dir
    
    cmd = [
        prank_exec, "predict",
        "-c", "alphafold",
        "-f", str(pdb_path.resolve()),
        "-o", str(out_dir.resolve()),
        "-visualizations", "0"
    ]
    
    print(f"Spouštím P2Rank na {pdb_path.name}...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out_dir

def process_predicted_pockets(out_dir):
    """Najde predikované kapsy, extrahuje sekvence a získá ESM embeddingy."""
    pocket_files = glob(os.path.join(out_dir, '*_pocket_*.pdb'))
    pocket_files.sort() # např. pocket_1, pocket_2...
    
    if not pocket_files:
        print("P2Rank nenašel žádné kapsy.")
        return None
        
    print(f"P2Rank našel {len(pocket_files)} kapes.")
    
    parser = PDBParser(QUIET=True)
    extractor = ESMFeatureExtractor()
    
    bag_features = []
    
    for pfile in pocket_files:
        structure = parser.get_structure('pocket', pfile)
        seq, _, _ = get_ca_coords_and_seq(structure)
        
        if len(seq) == 0:
            continue
            
        emb = extractor.extract_embeddings(seq) # [L, 1280]
        bag_features.append(torch.FloatTensor(emb))
        
    if not bag_features:
        return None
        
    # Spojení všech reziduí ze všech kapes do jednoho velkého tensoru [Total_L, 1280]
    bag_tensor = torch.cat(bag_features, dim=0)
    return bag_tensor

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 1. Definice tříd (kofaktorů) – toto pořadí odpovídá tréninku
    # V původním Binding_site_ex.py to bylo: ['NAD', 'FAD', 'ATP', 'acetyl-CoA', 'B12']
    # Upravte dle potřeby, pokud se to liší
    class_names = ['NAD', 'FAD', 'ATP', 'acetyl-CoA', 'B12']
    
    # 2. Načtení modelu
    print(f"Načítám model z {args.model_path}...")
    model = AttentionMIL_ESM(
        in_features=1280, 
        hidden_dim=256, 
        num_classes=len(class_names),
        num_heads=2
    ).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model nenalezen: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Predikce pro vstupní soubor
    pdb_path = Path(args.input)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Vstupní soubor nenalezen: {pdb_path}")
        
    # P2Rank
    out_dir = run_p2rank(pdb_path, args.prank)
    
    # ESM extrakce
    features = process_predicted_pockets(out_dir)
    
    if features is None:
        print("Inference selhala – nelze vyextrahovat vlastnosti.")
        return
        
    print(f"Extrahováno {features.shape[0]} reziduí (instancí) pro MIL síť.")
    
    # 4. Model Inference
    features = features.to(device).unsqueeze(0) # Přidat dummy batch dimenzi (model očekává unbatched, záleží na implementaci forward)
    # V AttentionMIL_ESM.forward je X tvaru [N, D] (jeden bag), takže bez unsqueeze
    features = features.squeeze(0) 
    
    with torch.no_grad():
        logits, A = model(features)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    # 5. Výsledky
    print("\n" + "="*40)
    print(f" VÝSLEDKY INFERENCE PRO {pdb_path.name}")
    print("="*40)
    
    probs_np = probs.cpu().numpy()
    results = []
    for i, p in enumerate(probs_np):
        results.append((class_names[i], p))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"TOP-1 Predikce: {results[0][0]} (Jistota: {results[0][1]*100:.2f} %)\n")
    print("Detailní pravděpodobnosti:")
    for name, p in results:
        print(f" - {name:>12}: {p*100:6.2f} %")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for AMICO")
    parser.add_argument('-i', '--input', required=True, help="Path to input PDB file")
    parser.add_argument('-m', '--model-path', default='best_esm_mil.pt', help="Path to trained model weights")
    parser.add_argument('--prank', default='data_prep/p2rank_2.5.1/prank', help="Path to P2Rank executable")
    
    args = parser.parse_args()
    main(args)
