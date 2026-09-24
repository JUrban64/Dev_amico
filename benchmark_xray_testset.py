#!/usr/bin/env python3
"""
=============================================================================
AMICO: Benchmark nejvýkonnějších modelů a Foldseeku na experimentálních 
       X-ray strukturách vazebných míst (Ground-Truth Positive Samples)
=============================================================================

Tento skript načte experimentální PDB struktury z adresáře Binding_Sites
(např. /storage/brno2/home/urbany/EquiPocket-MIL-/Binding_Sites/<COFACTOR>/positive/),
které prokazatelně vážou dané kofaktory (ATP, B12, acetyl-CoA, FAD, NAD),
a otestuje na nich nejlepší natrénované modely AMICO:
  1. Foldseek (1-Nearest Neighbor strukturní alignment)
  2. Cross-Attention MIL (cross_attention_mil_best.pt)
  3. Self-Attention MIL (self_attention_mil_best.pt)
  4. Ligand Cross-Attention MIL (ligand_cross_attention_mil_best.pt)
  5. Standard Attention MIL (best_esm_mil.pt - pokud existuje)

Skript automaticky:
  - Vyhledá všechny PDB struktury a P2Rank výstupy v positive/ složkách.
  - Využije předpočítané tenzory, nebo extrahuje ESM-2 embeddingy kapes a proteinů on-the-fly s cacheováním.
  - Spustí inferenci a spočítá Accuracy, Macro F1, Per-Class F1 a konfuzní matice.
  - Uloží detailní srovnávací tabulku v Markdownu, CSV i JSON formátu.
"""

import os
import sys
import glob
import json
import time
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Přidání kořenového adresáře do sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

FOLDER_TO_COFACTOR = {
    'ATP': 'ATP',
    'B12': 'B12',
    'COA': 'acetyl-CoA',
    'acetyl-CoA': 'acetyl-CoA',
    'acetyl_coa': 'acetyl-CoA',
    'FAD': 'FAD',
    'NAD': 'NAD'
}

# ============================================================================
# 1. POMOCNÉ FUNKCE PRO HLEDÁNÍ CEST A NORMALIZACI ID
# ============================================================================

def normalize_id(pid):
    """Normalizuje ID proteinu pro spolehlivé párování napříč formáty."""
    if not pid:
        return ""
    p = str(pid).strip()
    p = os.path.basename(p)
    p = p.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '').replace('_predictions', '')
    p = p.replace('_MERGED', '').replace('_merged', '').replace('clean_', '')
    return p.lower()


def find_default_binding_sites_dir():
    """Hledá složku Binding_Sites na clusteru i lokálně."""
    candidates = [
        "/storage/brno2/home/urbany/EquiPocket-MIL-/Binding_Sites",
        os.path.join(PROJECT_ROOT, "..", "EquiPocket-MIL-", "Binding_Sites"),
        os.path.join(PROJECT_ROOT, "Binding_Sites"),
        os.path.join(PROJECT_ROOT, "data_prep", "Binding_Sites"),
        os.path.join(PROJECT_ROOT, "..", "Binding_Sites")
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def find_default_train_structures_dir():
    """Hledá referenční/trénovací PDB struktury pro Foldseek databázi."""
    candidates = [
        os.path.join(PROJECT_ROOT, "structures"),
        os.path.join(PROJECT_ROOT, "data_prep", "structures"),
        "/storage/brno2/home/urbany/EquiPocket-MIL-/structures",
        os.path.join(PROJECT_ROOT, "..", "EquiPocket-MIL-", "structures"),
        os.path.join(PROJECT_ROOT, "..", "structures"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # Ověříme, zda obsahuje PDB soubory
            if glob.glob(os.path.join(c, "**", "*.pdb"), recursive=True):
                return os.path.abspath(c)
    return None


# ============================================================================
# 2. SBĚR TESTOVACÍCH X-RAY STRUKTUR (POSITIVE SAMPLES)
# ============================================================================

def collect_xray_test_pdbs(binding_sites_dir):
    """
    Prohledá podsložky v Binding_Sites (ATP/positive, NAD/positive, COA/positive, ...)
    a sestaví seznam testovacích PDB s ground-truth labely.
    """
    bs_path = Path(binding_sites_dir)
    if not bs_path.exists():
        raise FileNotFoundError(f"Složka s vazebnými místy nenalezena: {bs_path}")

    print(f"\n📂 Prohledávám X-ray struktury v: {bs_path.resolve()}")
    samples = []
    
    # Procházíme podsložky odpovídající kofaktorům
    for sub in bs_path.iterdir():
        if not sub.is_dir() or sub.name.startswith('.'):
            continue
            
        cofactor_key = sub.name.upper()
        # Vyhledání odpovídajícího názvu kofaktoru
        matched_cofactor = None
        for k, v in FOLDER_TO_COFACTOR.items():
            if k.upper() == cofactor_key:
                matched_cofactor = v
                break
                
        if not matched_cofactor:
            continue

        label_idx = TARGET_NAMES.index(matched_cofactor)
        positive_dir = sub / "positive"
        
        # Pokud neexistuje přímo 'positive', zkusíme prohledat přímo sub
        search_dirs = [positive_dir] if positive_dir.exists() else [sub]

        for sdir in search_dirs:
            pdbs = [p for p in sdir.glob("*.pdb") if "_pocket_" not in p.name and "prank_output" not in str(p)]
            for pdb_file in pdbs:
                base_stem = pdb_file.stem
                clean_stem = base_stem.replace('.pdb', '')
                
                # Zkusíme najít P2Rank výstupní složku
                prank_candidates = [
                    pdb_file.parent / f"{clean_stem}_prank_output",
                    pdb_file.parent / f"{clean_stem}.pdb_prank_output",
                    sub / f"{clean_stem}_prank_output",
                    sub / "positive" / f"{clean_stem}_prank_output"
                ]
                prank_dir = next((c for c in prank_candidates if c.exists() and c.is_dir()), None)
                
                samples.append({
                    'pdb_path': str(pdb_file.resolve()),
                    'protein_id': base_stem,
                    'norm_id': normalize_id(base_stem),
                    'cofactor': matched_cofactor,
                    'label': label_idx,
                    'prank_dir': str(prank_dir.resolve()) if prank_dir else None
                })

    print(f"✅ Celkem nalezeno {len(samples)} experimentálních X-ray struktur (positive samples):")
    counts = defaultdict(int)
    has_prank = defaultdict(int)
    for s in samples:
        counts[s['cofactor']] += 1
        if s['prank_dir']:
            has_prank[s['cofactor']] += 1
            
    for name in TARGET_NAMES:
        c = counts[name]
        hp = has_prank[name]
        print(f"   - {name:<12s}: {c:>4d} struktur (s P2Rank výstupy: {hp:>4d})")

    return samples


# ============================================================================
# 3. EXTRAKCE / NAČTENÍ FEATURE REPREZENTACÍ (ESM-2 POCKETS + PROTEIN)
# ============================================================================

def parse_full_sequence_from_pdb(pdb_path):
    """Extrahuje sekvenci aminokyselin ze všech řetězců PDB souboru."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    try:
        structure = parser.get_structure('protein', pdb_path)
        seq = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_id()[0] == ' ':
                        rname = res.get_resname()
                        seq.append(three_to_one.get(rname, 'X'))
        return ''.join(seq)
    except Exception as e:
        print(f"⚠️ Chyba při čtení sekvence z {pdb_path}: {e}")
        return None


def parse_pockets_from_prank_output(prank_dir, pdb_path):
    """
    Získá sekvence jednotlivých kapes z P2Rank složky.
    Prioritně využije _residues.csv, fallback na fyzické _pocket_*.pdb soubory.
    """
    import csv
    from Bio.PDB import PDBParser
    
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    pockets = []
    
    # 1. Zkusit najít _residues.csv
    res_csvs = glob.glob(os.path.join(prank_dir, "*_residues.csv"))
    if res_csvs and os.path.exists(pdb_path):
        res_csv = res_csvs[0]
        try:
            # Načteme PDB strukturu pro mapování chain/res_id na rezidua
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure('p', pdb_path)
            res_dict = {}
            for model in struct:
                for chain in model:
                    cid = chain.get_id().strip()
                    for r in chain:
                        if r.get_id()[0] == ' ':
                            rid = str(r.get_id()[1]).strip()
                            res_dict[(cid, rid)] = r.get_resname()
                            
            pocket_residues = defaultdict(list)
            with open(res_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    clean = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                    pnum = clean.get('pocket', '0')
                    if pnum != '0' and pnum != '':
                        c = clean.get('chain', '').strip()
                        rlabel = clean.get('residue_label', '').strip()
                        pocket_residues[pnum].append((c, rlabel))
                        
            # Seřazení kapes podle čísla
            for pnum in sorted(pocket_residues.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                seq = []
                for c, rlabel in pocket_residues[pnum]:
                    key = (c, rlabel)
                    if key in res_dict:
                        rname = res_dict[key]
                        seq.append(three_to_one.get(rname, 'X'))
                if seq:
                    pockets.append(''.join(seq))
        except Exception:
            pass

    # 2. Fallback: Hledání fyzických _pocket_*.pdb
    if not pockets:
        pocket_pdbs = sorted(glob.glob(os.path.join(prank_dir, "*_pocket_*.pdb")))
        parser = PDBParser(QUIET=True)
        for ppdb in pocket_pdbs:
            try:
                pstruct = parser.get_structure('pocket', ppdb)
                seq = []
                for model in pstruct:
                    for chain in model:
                        for r in chain:
                            if r.get_id()[0] == ' ':
                                seq.append(three_to_one.get(r.get_resname(), 'X'))
                if seq:
                    pockets.append(''.join(seq))
            except Exception:
                continue

    return pockets


def load_or_extract_features(samples, data_dir=None, cache_path="xray_extracted_features.pt", device=None):
    """
    Zajistí pocket_features [N_pockets, 1280] a full_protein_feature [1280] pro všechny vzorky.
    1. Zkusí načíst z cache (xray_extracted_features.pt).
    2. Zkusí dohledat v existujícím esm_dataset.pt a esm_full_proteins.pt.
    3. Chybějící extrahuje on-the-fly pomocí HuggingFace ESM-2 (facebook/esm2_t33_650M_UR50D).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_dict = {}

    # 1. Kontrola lokální cache
    if os.path.exists(cache_path):
        try:
            print(f"\n📦 Načítám předextrahované features z cache: {cache_path}")
            features_dict = torch.load(cache_path, map_location='cpu', weights_only=False)
            print(f"   Načteno {len(features_dict)} proteinů z cache.")
        except Exception as e:
            print(f"⚠️ Nelze načíst cache {cache_path}: {e}")
            features_dict = {}

    # 2. Kontrola předpočítaných datasetových souborů (esm_dataset.pt, esm_full_proteins.pt)
    search_dirs = [data_dir] if data_dir else []
    search_dirs.extend([
        os.path.join(PROJECT_ROOT, "data_prep"),
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, "..", "data_prep"),
        os.path.join(PROJECT_ROOT, "..")
    ])

    pockets_path, full_path = None, None
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        cand_pockets = os.path.join(d, "esm_dataset.pt")
        cand_full = os.path.join(d, "esm_full_proteins.pt")
        if os.path.exists(cand_pockets) and not pockets_path:
            pockets_path = cand_pockets
        if os.path.exists(cand_full) and not full_path:
            full_path = cand_full

    if pockets_path and full_path:
        print(f"\n🔍 Hledám chybějící proteiny v existujících datasetech:")
        print(f"   - Kapsy: {pockets_path}")
        print(f"   - Full proteiny: {full_path}")
        try:
            from dataset_cross_mil import load_cross_mil_data
            precomputed_bags = load_cross_mil_data(pockets_path, full_path, mode='pockets')
            norm_map = {normalize_id(b['protein_id']): b for b in precomputed_bags}

            matched = 0
            for s in samples:
                nid = s['norm_id']
                if nid not in features_dict and nid in norm_map:
                    bag = norm_map[nid]
                    features_dict[nid] = {
                        'pocket_features': bag['pocket_features'].cpu(),
                        'full_protein_feature': bag['full_protein_feature'].cpu()
                    }
                    matched += 1
            print(f"   Úspěšně namapováno {matched} proteinů z existujících tenzorů.")
        except Exception as e:
            print(f"⚠️ Chyba při načítání precomputed tenzorů: {e}")

    # 3. Zjištění, které vzorky stále chybí
    missing_samples = [s for s in samples if s['norm_id'] not in features_dict]

    if missing_samples:
        print(f"\n⚡ Je třeba extrahovat ESM-2 embeddingy pro {len(missing_samples)} struktur...")
        print("   Načítám ESM-2 model (facebook/esm2_t33_650M_UR50D)...")
        from transformers import AutoTokenizer, EsmModel
        
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        esm_model = EsmModel.from_pretrained(model_name).to(device)
        esm_model.eval()

        def get_esm_embedding(seq_str):
            if not seq_str:
                return None
            inputs = tokenizer(seq_str, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = esm_model(**inputs)
            emb = out.last_hidden_state[0, 1:-1, :] # [L, 1280]
            return emb.mean(dim=0).cpu() # [1280]

        extracted_new = 0
        for s in missing_samples:
            nid = s['norm_id']
            pdb_path = s['pdb_path']
            prank_dir = s['prank_dir']
            
            # A) Full protein embedding
            full_seq = parse_full_sequence_from_pdb(pdb_path)
            if not full_seq:
                continue
            full_emb = get_esm_embedding(full_seq)
            if full_emb is None:
                continue
                
            # B) Pocket embeddings
            pocket_seqs = parse_pockets_from_prank_output(prank_dir, pdb_path) if prank_dir else []
            pocket_embs = []
            for pseq in pocket_seqs:
                pemb = get_esm_embedding(pseq)
                if pemb is not None:
                    pocket_embs.append(pemb)
                    
            if not pocket_embs:
                # Fallback: pokud P2Rank nenašel kapsu, použijeme full protein embedding jako dummy kapsu
                pocket_tensor = full_emb.unsqueeze(0)
            else:
                pocket_tensor = torch.stack(pocket_embs) # [N, 1280]

            features_dict[nid] = {
                'pocket_features': pocket_tensor,
                'full_protein_feature': full_emb
            }
            extracted_new += 1
            if extracted_new % 20 == 0 or extracted_new == len(missing_samples):
                print(f"   Extrahováno {extracted_new}/{len(missing_samples)}...")

        # Uložit do cache
        try:
            torch.save(features_dict, cache_path)
            print(f"💾 Nově extrahované features uloženy do cache: {cache_path}")
        except Exception as e:
            print(f"⚠️ Nelze uložit cache: {e}")

    print(f"✅ Připraveny features pro {len(features_dict)}/{len(samples)} proteinů.")
    return features_dict


# ============================================================================
# 4. INFERENCE HLUBOKÝCH MODELŮ (PyTorch)
# ============================================================================

def run_neural_inference(model_key, checkpoint_filename, valid_samples, features_dict, models_dir, device):
    """Provede inferenci zvoleného modelu nad připravenými vzorky."""
    ckpt_path = os.path.join(models_dir, checkpoint_filename)
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint pro {model_key} nenalezen: {ckpt_path} -> Přeskakuji.")
        return None

    print(f"\n🧠 Spouštím inferenci modelu: {model_key} (Checkpoint: {checkpoint_filename})...")
    
    # 1. Inicializace architektury modelu
    if model_key == "cross_attention_mil":
        from model_cross_attention_mil import CrossAttentionMIL
        model = CrossAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
    elif model_key == "self_attention_mil":
        from model_self_attention_mil import SelfAttentionMIL
        model = SelfAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
    elif model_key == "ligand_cross_mil":
        from model_ligand_cross_attention_mil import LigandCrossAttentionMIL
        model = LigandCrossAttentionMIL(feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
    elif model_key == "standard_mil":
        from model import AttentionMIL_ESM
        model = AttentionMIL_ESM(in_features=1280, hidden_dim=256, num_classes=5, dropout=0.25, gated_attention=True)
    else:
        print(f"Neznámý model {model_key}")
        return None

    # Načtení vah
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Chyba při načítání vah z {ckpt_path}: {e}")
        return None

    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []
    preds_by_pid = {}

    with torch.no_grad():
        for s in valid_samples:
            nid = s['norm_id']
            feat_data = features_dict.get(nid)
            if not feat_data:
                continue

            p_feat = feat_data['pocket_features'].unsqueeze(0).to(device) # [1, N, 1280]
            full_feat = feat_data['full_protein_feature'].unsqueeze(0).to(device) # [1, 1280]
            mask = torch.zeros(1, p_feat.size(1), dtype=torch.bool, device=device) # [1, N]
            
            if model_key in ["cross_attention_mil", "self_attention_mil", "ligand_cross_mil"]:
                logits, _ = model(p_feat, mask, full_feat)
            elif model_key == "standard_mil":
                logits, _ = model(p_feat, mask)
            else:
                continue

            pred_class = torch.argmax(logits, dim=-1).item()
            true_class = s['label']
            
            y_true.append(true_class)
            y_pred.append(pred_class)
            preds_by_pid[s['protein_id']] = pred_class

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'preds_by_pid': preds_by_pid
    }


# ============================================================================
# 5. FOLDSEEK 1-NN STRUKTURNÍ BENCHMARK
# ============================================================================

def run_foldseek_xray_benchmark(test_samples, train_dir, threads=8):
    """
    Spustí Foldseek easy-search: X-ray testovací proteiny (query) vs. trénovací PDB (target).
    Pro každý query protein vybere Top-1 hit podle nejvyššího qtmscore / e-value.
    """
    if shutil.which("foldseek") is None:
        print("⚠️ Foldseek příkaz nebyl nalezen v PATH. Přeskakuji Foldseek benchmark.")
        return None

    if not train_dir or not os.path.exists(train_dir):
        print(f"⚠️ Trénovací adresář pro Foldseek nenalezen ({train_dir}). Přeskakuji.")
        return None

    print(f"\n🔍 Spouštím Foldseek 1-NN Benchmark...")
    print(f"   Query set (X-ray test): {len(test_samples)} struktur")
    print(f"   Target databáze (Train): {train_dir}")

    # Mapování target PDB -> ground truth label
    # Prohledáme všechny PDB soubory v train_dir a určíme jejich kofaktor podle podsložky
    train_pdbs = glob.glob(os.path.join(train_dir, "**", "*.pdb"), recursive=True)
    target_labels = {}
    for p in train_pdbs:
        tstem = os.path.basename(p).replace('.pdb', '')
        t_nid = normalize_id(tstem)
        
        # Zjištění kofaktoru z cesty
        p_parts = Path(p).parts
        cofactor = None
        for part in p_parts:
            part_up = part.upper()
            for k, v in FOLDER_TO_COFACTOR.items():
                if k.upper() == part_up:
                    cofactor = v
                    break
            if cofactor:
                break
                
        if cofactor and cofactor in TARGET_NAMES:
            target_labels[t_nid] = TARGET_NAMES.index(cofactor)

    if not target_labels:
        print("⚠️ Nepodařilo se detekovat labely kofaktorů v trénovací Foldseek databázi.")
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        query_dir = os.path.join(tmp_dir, "queries")
        os.makedirs(query_dir, exist_ok=True)
        
        # Vytvoření symlinků na query soubory
        for s in test_samples:
            src = s['pdb_path']
            dst = os.path.join(query_dir, f"{s['norm_id']}.pdb")
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        fs_work = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_work, exist_ok=True)

        cmd = [
            "foldseek", "easy-search",
            query_dir, train_dir, out_tsv, fs_work,
            "--format-output", "query,target,evalue,qtmscore,bits",
            "-e", "10.0",
            "--threads", str(threads)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"❌ Chyba při spuštění Foldseeku: {e}")
            return None

        if not os.path.exists(out_tsv):
            print("⚠️ Foldseek nevygeneroval výstupní TSV soubor.")
            return None

        # Načtení zarovnání
        try:
            df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "evalue", "qtmscore", "bits"])
            df['query'] = df['query'].str.replace('.pdb', '', regex=False).apply(normalize_id)
            df['target'] = df['target'].str.replace('.pdb', '', regex=False).apply(normalize_id)
            
            # Řazení podle query a qtmscore sestupně
            df = df.sort_values(by=['query', 'qtmscore'], ascending=[True, False])
            # Ponechat pouze Top-1 hit
            top1 = df.drop_duplicates(subset=['query'], keep='first')
            
            foldseek_preds = {}
            for _, row in top1.iterrows():
                q = row['query']
                t = row['target']
                if t in target_labels:
                    foldseek_preds[q] = target_labels[t]

            y_true, y_pred = [], []
            preds_by_pid = {}
            unaligned = 0

            for s in test_samples:
                nid = s['norm_id']
                y_true.append(s['label'])
                if nid in foldseek_preds:
                    pred_l = foldseek_preds[nid]
                    y_pred.append(pred_l)
                    preds_by_pid[s['protein_id']] = pred_l
                else:
                    # Nezarovnaný protein -> označíme jako miss (predikce -1)
                    y_pred.append(-1)
                    preds_by_pid[s['protein_id']] = -1
                    unaligned += 1

            if unaligned > 0:
                print(f"ℹ️ Foldseek nenalezl žádný hit pro {unaligned} testovacích proteinů (nastaveno na miss).")

            return {
                'y_true': y_true,
                'y_pred': y_pred,
                'preds_by_pid': preds_by_pid
            }
        except Exception as e:
            print(f"⚠️ Chyba při zpracování výstupu Foldseeku: {e}")
            return None


# ============================================================================
# 6. VYHODNOCENÍ A TVORBA VÝSTUPNÍCH REPORTŮ
# ============================================================================

def compute_metrics_and_summary(eval_results, test_samples, out_prefix):
    """Spočítá metriky, vytiskne konfuzní matice a uloží výsledky do Markdown / CSV."""
    summary_rows = []
    
    print("\n" + "=" * 105)
    print("                     VÝSLEDKY BENCHMARKU NA EXPERIMENTÁLNÍCH X-RAY STRUKTURÁCH                     ")
    print("=" * 105)
    print(f"{'Model':<26s} | {'Vzorků':<7s} | {'Accuracy':<10s} | {'Macro F1':<10s} | {'Weighted F1':<12s} | {'acetyl-CoA':<10s} | {'ATP':<8s} | {'B12':<8s} | {'FAD':<8s} | {'NAD':<8s}")
    print("-" * 120)

    for mkey, res in eval_results.items():
        if not res:
            continue
        y_t = res['y_true']
        y_p = res['y_pred']

        # Pro výpočet metrik nahradíme případné -1 hodnoty (nezarovnáno ve Foldseeku)
        valid_pairs = [(t, p) for t, p in zip(y_t, y_p) if p != -1]
        
        # Celková accuracy
        acc = accuracy_score(y_t, y_p)
        f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        f1_weighted = f1_score(y_t, y_p, average='weighted', zero_division=0)
        
        rep = classification_report(y_t, y_p, target_names=TARGET_NAMES, labels=list(range(5)), output_dict=True, zero_division=0)
        
        per_class = {}
        for cname in TARGET_NAMES:
            per_class[cname] = rep.get(cname, {}).get('f1-score', 0.0) * 100

        print(f"{mkey:<26s} | {len(y_t):<7d} | {acc * 100:6.2f} %  | {f1_macro * 100:6.2f} %  | {f1_weighted * 100:6.2f} %    | {per_class['acetyl-CoA']:6.1f} %   | {per_class['ATP']:6.1f} % | {per_class['B12']:6.1f} % | {per_class['FAD']:6.1f} % | {per_class['NAD']:6.1f} %")

        summary_rows.append({
            'Model': mkey,
            'Total_Samples': len(y_t),
            'Accuracy': round(acc * 100, 2),
            'Macro_F1': round(f1_macro * 100, 2),
            'Weighted_F1': round(f1_weighted * 100, 2),
            'F1_acetyl-CoA': round(per_class['acetyl-CoA'], 2),
            'F1_ATP': round(per_class['ATP'], 2),
            'F1_B12': round(per_class['B12'], 2),
            'F1_FAD': round(per_class['FAD'], 2),
            'F1_NAD': round(per_class['NAD'], 2)
        })

    print("=" * 120 + "\n")

    # Detailní konfuzní matice
    for mkey, res in eval_results.items():
        if not res:
            continue
        y_t = res['y_true']
        y_p = res['y_pred']
        cm = confusion_matrix(y_t, y_p, labels=list(range(5)))
        print(f"📊 Konfuzní matice: {mkey}")
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in TARGET_NAMES], columns=[f"Pred_{c}" for c in TARGET_NAMES])
        print(cm_df.to_string())
        print("-" * 60)

    # Uložení CSV souhrnu
    sum_df = pd.DataFrame(summary_rows)
    csv_out = f"{out_prefix}.csv"
    sum_df.to_csv(csv_out, index=False)
    print(f"\n💾 Souhrnná tabulka uložena do: {csv_out}")

    # Uložení Markdown reportu
    md_out = f"{out_prefix}.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# AMICO: Benchmark Modelů na Experimentálních X-ray Strukturách\n\n")
        f.write(f"Vygenerováno: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Dataset obsahuje experimentálně ověřené X-ray struktury vážící příslušné kofaktory (Ground-Truth positive vazebná místa).\n\n")
        f.write("### Celkové výsledky\n\n")
        f.write(sum_df.to_markdown(index=False))
        f.write("\n\n")

    print(f"📄 Markdown report uložen do: {md_out}")

    # Uložení predikcí po jednotlivých proteinech do detailního CSV
    detailed_rows = []
    for s in test_samples:
        pid = s['protein_id']
        row = {
            'protein_id': pid,
            'ground_truth': s['cofactor'],
            'true_label': s['label']
        }
        for mkey, res in eval_results.items():
            if res and 'preds_by_pid' in res:
                p_idx = res['preds_by_pid'].get(pid, -1)
                p_name = TARGET_NAMES[p_idx] if 0 <= p_idx < 5 else "N/A"
                row[f"{mkey}_pred"] = p_name
                row[f"{mkey}_correct"] = (p_idx == s['label'])
        detailed_rows.append(row)

    det_df = pd.DataFrame(detailed_rows)
    det_csv = f"{out_prefix}_detailed_predictions.csv"
    det_df.to_csv(det_csv, index=False)
    print(f"📋 Detailní predikce pro každý protein uloženy do: {det_csv}\n")


# ============================================================================
# 7. HLAVNÍ FUNKCE (MAIN)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Otestuje nejlepší AMICO modely a Foldseek na experimentálních X-ray strukturách.")
    parser.add_argument(
        "--binding-sites-dir", default=None,
        help="Cesta ke složce s X-ray vazebnými místy (default: auto-detekce EquiPocket-MIL-/Binding_Sites)."
    )
    parser.add_argument(
        "--models-dir", default=PROJECT_ROOT,
        help="Složka s váhami modelů (*_best.pt). Default: kořen projektu."
    )
    parser.add_argument(
        "--train-structures-dir", default=None,
        help="Složka s trénovacími strukturami pro Foldseek databázi."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Cesta k předpočítaným .pt datasetům (esm_dataset.pt, esm_full_proteins.pt)."
    )
    parser.add_argument(
        "--features-cache", default="xray_extracted_features.pt",
        help="Cesta k souboru s nacacheovanými extrahovanými features (default: xray_extracted_features.pt)."
    )
    parser.add_argument(
        "--out-prefix", default="xray_benchmark_results",
        help="Prefix pro výstupní CSV/MD soubory (default: xray_benchmark_results)."
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help="Výpočetní zařízení pro PyTorch modely (default: auto)."
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Počet vláken pro Foldseek (default: 8)."
    )
    parser.add_argument(
        "--skip-foldseek", action="store_true",
        help="Přeskočí Foldseek benchmark."
    )
    parser.add_argument(
        "--skip-neural", action="store_true",
        help="Přeskočí neuronové modely (spustí pouze Foldseek)."
    )

    args = parser.parse_args()

    # Výběr výpočetního zařízení
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("      AMICO: BENCHMARK NEJLEPŠÍCH MODELŮ NA X-RAY EXPERIMENTÁLNÍCH STRUKTURÁCH      ")
    print("=" * 80)
    print(f"Použité zařízení (PyTorch): {device}")

    # 1. Detekce adresářů
    bs_dir = args.binding_sites_dir or find_default_binding_sites_dir()
    if not bs_dir:
        print("❌ Chyba: Nepodařilo se nalézt složku Binding_Sites.")
        print("Zadejte cestu manuálně pomocí parametru: --binding-sites-dir /cesta/k/Binding_Sites")
        sys.exit(1)

    train_struct_dir = args.train_structures_dir or find_default_train_structures_dir()

    # 2. Sběr X-ray testovacích struktur
    test_samples = collect_xray_test_pdbs(bs_dir)
    if not test_samples:
        print(f"❌ Žádné PDB struktury v {bs_dir} nebyly nalezeny.")
        sys.exit(1)

    eval_results = {}

    # 3. Foldseek Benchmark
    if not args.skip_foldseek:
        fs_res = run_foldseek_xray_benchmark(test_samples, train_struct_dir, threads=args.threads)
        if fs_res:
            eval_results['foldseek_1nn'] = fs_res
    else:
        print("ℹ️ Foldseek benchmark přeskočen na základě parametru --skip-foldseek.")

    # 4. Neural Models Benchmark
    if not args.skip_neural:
        # A) Příprava / extrakce / načtení features
        features_dict = load_or_extract_features(
            samples=test_samples,
            data_dir=args.data_dir,
            cache_path=args.features_cache,
            device=device
        )

        valid_samples = [s for s in test_samples if s['norm_id'] in features_dict]
        if not valid_samples:
            print("❌ Pro nalezené proteiny se nepodařilo připravit features.")
            sys.exit(1)

        # B) Seznam nejlepších modelů k otestování
        neural_models_to_test = [
            ("cross_attention_mil", "cross_attention_mil_best.pt"),
            ("self_attention_mil", "self_attention_mil_best.pt"),
            ("ligand_cross_mil", "ligand_cross_attention_mil_best.pt"),
            ("standard_mil", "best_esm_mil.pt")
        ]

        for mkey, ckpt_name in neural_models_to_test:
            res = run_neural_inference(
                model_key=mkey,
                checkpoint_filename=ckpt_name,
                valid_samples=valid_samples,
                features_dict=features_dict,
                models_dir=args.models_dir,
                device=device
            )
            if res:
                eval_results[mkey] = res
    else:
        print("ℹ️ Neuronové modely přeskočeny na základě parametru --skip-neural.")

    # 5. Výpočet metrik a export výsledků
    if eval_results:
        compute_metrics_and_summary(eval_results, test_samples, args.out_prefix)
    else:
        print("⚠️ Žádný model nebyl úspěšně vyhodnocen.")


if __name__ == '__main__':
    main()
