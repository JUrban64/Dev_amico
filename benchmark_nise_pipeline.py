#!/usr/bin/env python3
"""
=============================================================================
AMICO: Benchmark nehomologních isofunkčních enzymů (NISE Benchmark)
=============================================================================

Tento skript demonstruje schopnost modelů AMICO generalizovat přes nehomologní
strukturní foldy (konvergentní evoluce) na datech z databáze NISE:
  1. Výběr reprezentativního vzorku (např. 15 proteinů na kofaktor = 75 celkem),
     kde enzymy se STEJNÝM kofaktorem a stejným EC číslem patří do RŮZNÝCH
     SCOP strukturních superfamilií (různé 3D foldy).
  2. Automatické stažení predikovaných 3D struktur z AlphaFold DB.
  3. Predikce vazebných kapes pomocí P2Rank.
  4. Extrakce ESM-2 embeddingů (pockets + full protein).
  5. Srovnání Foldseek (1-NN alignment) vs. AMICO modely (Cross-Attn, Self-Attn, Ligand-Cross).
  6. Vyhodnocení: Prokázání, že AMICO pozná vazebné místo pro daný kofaktor
     i u foldů, kde Foldseek selže (protože chybí globální strukturní homologie).
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
import urllib.request

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
except ImportError:
    accuracy_score = None
    f1_score = None
    classification_report = None
    confusion_matrix = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

# ============================================================================
# KROK 1: CHYTRÝ VÝBĚR REPREZENTATIVNÍHO VZORKU Z NISE
# ============================================================================

def select_nise_sample(nise_tsv_path, sample_per_cofactor=15, out_tsv="nise_selected_sample.tsv"):
    """
    Vybere vyvážený reprezentativní vzorek enzymů z NISE.
    Prioritizuje EC čísla s více různými SCOP superfamiliemi (true NISE).
    """
    print(f"\n📋 Načítám NISE dataset z: {nise_tsv_path}")
    df = pd.read_csv(nise_tsv_path, sep='\t')
    
    # Detekce EC čísel, která mají v rámci kofaktoru více než 1 superfamilii
    true_nise = df.groupby(['cofactor', 'ec'])['supfam'].nunique().reset_index()
    true_nise = true_nise[true_nise['supfam'] > 1]
    
    selected_groups = []
    
    for cof in TARGET_NAMES:
        cof_df = df[df['cofactor'] == cof].copy()
        if cof_df.empty:
            continue
            
        nise_ecs = true_nise[true_nise['cofactor'] == cof]['ec'].tolist()
        
        # Upřednostníme EC čísla s více foldy
        if nise_ecs:
            sub = cof_df[cof_df['ec'].isin(nise_ecs)].copy()
        else:
            sub = cof_df.copy()
            
        # Seskupíme podle (EC, supfam) a vybereme z každého unikátního páru 1 reprezentanta
        representatives = sub.groupby(['ec', 'supfam']).first().reset_index()
        
        if len(representatives) > sample_per_cofactor:
            representatives = representatives.sample(sample_per_cofactor, random_state=42)
            
        # Pokud je reprezentantů méně než požadovaný počet, doplníme dalšími proteiny z daného kofaktoru
        if len(representatives) < sample_per_cofactor:
            needed = sample_per_cofactor - len(representatives)
            remaining = cof_df[~cof_df['entry'].isin(representatives['entry'])]
            if not remaining.empty:
                extra = remaining.sample(min(needed, len(remaining)), random_state=42)
                representatives = pd.concat([representatives, extra], ignore_index=True)
                
        selected_groups.append(representatives)

    sample_df = pd.concat(selected_groups, ignore_index=True)
    sample_df.to_csv(out_tsv, sep='\t', index=False)
    
    print(f"✅ Vybráno {len(sample_df)} vzorků napříč {sample_df['cofactor'].nunique()} kofaktory:")
    for cof in TARGET_NAMES:
        c_sub = sample_df[sample_df['cofactor'] == cof]
        n_ec = c_sub['ec'].nunique()
        n_sup = c_sub['supfam'].nunique()
        print(f"   - {cof:<12s}: {len(c_sub):>2d} proteinů | {n_ec:>2d} různých EC čísel | {n_sup:>2d} různých SCOP superfamilií (foldů)")
        
    print(f"💾 Seznam vzorků uložen do: {out_tsv}")
    return sample_df


# ============================================================================
# KROK 2: AUTOMATICKÉ STAŽENÍ STRUKTUR Z ALPHAFOLD DB
# ============================================================================

def download_alphafold_structures(sample_df, structures_dir="nise_structures"):
    """
    Stáhne predikované 3D modely z AlphaFold DB (EBI) pro vybraná UniProt ID.
    Ukládá je do složky nise_structures/<COFACTOR>/<entry>.pdb.
    """
    struct_root = Path(structures_dir)
    struct_root.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    cached = 0
    failed = 0
    
    print(f"\n🌐 Stahuji 3D struktury z AlphaFold DB (cílová složka: {struct_root.resolve()})...")
    
    downloaded_paths = {}
    
    for idx, row in sample_df.iterrows():
        uniprot_id = str(row['entry']).strip()
        cofactor = str(row['cofactor']).strip()
        cof_dir = struct_root / cofactor
        cof_dir.mkdir(parents=True, exist_ok=True)
        
        target_pdb = cof_dir / f"{uniprot_id}.pdb"
        
        # Kontrola, zda již existuje
        if target_pdb.exists() and target_pdb.stat().st_size > 1000:
            cached += 1
            downloaded_paths[uniprot_id] = str(target_pdb.resolve())
            continue
            
        # Dotaz na AlphaFold DB API
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0 (AMICO-Benchmark)'})
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    pdb_url = data[0].get('pdbUrl')
                    if pdb_url:
                        # Stažení PDB
                        pdb_req = urllib.request.Request(pdb_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(pdb_req, timeout=15) as pdb_resp:
                            content = pdb_resp.read()
                            with open(target_pdb, 'wb') as f_out:
                                f_out.write(content)
                        downloaded += 1
                        downloaded_paths[uniprot_id] = str(target_pdb.resolve())
                    else:
                        failed += 1
                else:
                    failed += 1
        except Exception:
            failed += 1
            
        time.sleep(0.15)
        
        if (idx + 1) % 15 == 0 or (idx + 1) == len(sample_df):
            print(f"   [{idx + 1:2d}/{len(sample_df)}] Staženo: {downloaded:2d} | Z cache: {cached:2d} | Selhalo: {failed:2d}")

    print(f"✅ Dokončeno stahování struktur: {len(downloaded_paths)} k dispozici (z celkových {len(sample_df)}).")
    return downloaded_paths


# ============================================================================
# KROK 3: PREDIKCE KAPES POMOCÍ P2RANK
# ============================================================================

def find_prank_executable(custom_path=None):
    """Najde spustitelný soubor P2Ranku."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        "prank",
        os.path.join(PROJECT_ROOT, "p2rank_2.5.1", "prank"),
        os.path.join(PROJECT_ROOT, "p2rank", "prank"),
        os.path.join(PROJECT_ROOT, "..", "p2rank_2.5.1", "prank"),
        "/storage/brno2/home/urbany/p2rank_2.5.1/prank",
        "/storage/brno2/home/urbany/Dev_amico/p2rank_2.5.1/prank"
    ]
    for c in candidates:
        if shutil.which(c):
            return c
        if os.path.exists(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def run_p2rank_for_structures(downloaded_paths, prank_exec=None, threads=6):
    """Spustí P2Rank predikci kapes na stažených strukturách."""
    prank_bin = find_prank_executable(prank_exec)
    
    if not prank_bin:
        print("⚠️ P2Rank spustitelný soubor nenalezen. Bude použit sekvenční full-protein embedding jako fallback.")
        return {}

    print(f"\n🔬 Spouštím P2Rank ({prank_bin}) pro predikci vazebných kapes...")
    
    to_process = []
    prank_dirs = {}
    
    for uid, pdb_str in downloaded_paths.items():
        pdb_p = Path(pdb_str)
        target_dir = pdb_p.parent / f"{pdb_p.stem}_prank_output"
        pred_csv = target_dir / f"{pdb_p.name}_predictions.csv"
        
        if target_dir.exists() and pred_csv.exists():
            prank_dirs[uid] = str(target_dir.resolve())
        else:
            to_process.append(pdb_p)

    if not to_process:
        print(f"⚡ Všechny P2Rank predikce již existují v cache ({len(prank_dirs)} hotovo).")
        return prank_dirs

    print(f"   Ke zpracování zbývá: {len(to_process)} struktur (Threads: {threads}).")
    
    temp_out = Path("./temp_prank_nise")
    temp_out.mkdir(parents=True, exist_ok=True)
    ds_file = Path("nise_batch.ds")
    
    with open(ds_file, "w") as f:
        for p in to_process:
            f.write(f"{p.resolve()}\n")

    cmd = [
        prank_bin, "predict",
        "-threads", str(threads),
        "-visualizations", "0",
        "-o", str(temp_out),
        str(ds_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        
        # Roztřídění výstupů do příslušných složek
        for p in to_process:
            target_dir = p.parent / f"{p.stem}_prank_output"
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in temp_out.glob(f"{p.name}*"):
                if f.is_file():
                    shutil.move(str(f), str(target_dir / f.name))
            prank_dirs[p.stem] = str(target_dir.resolve())
            
        print("✅ P2Rank úspěšně dokončil predikci kapes.")
    except Exception as e:
        print(f"⚠️ Chyba při běhu P2Ranku: {e}")
    finally:
        if ds_file.exists(): ds_file.unlink()
        if temp_out.exists(): shutil.rmtree(temp_out, ignore_errors=True)

    return prank_dirs


# ============================================================================
# KROK 4: EXTRAKCE ESM-2 EMBEDDINGŮ (POCKETS + FULL PROTEIN)
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
        struct = parser.get_structure('protein', pdb_path)
        seq = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if res.get_id()[0] == ' ':
                        rname = res.get_resname()
                        seq.append(three_to_one.get(rname, 'X'))
        return ''.join(seq)
    except Exception:
        return None


def parse_pockets_from_prank(prank_dir, pdb_path):
    """Načte sekvence kapes z P2Rank výstupů."""
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
    res_csvs = glob.glob(os.path.join(prank_dir, "*_residues.csv"))
    if res_csvs and os.path.exists(pdb_path):
        try:
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
            with open(res_csvs[0], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    clean = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                    pnum = clean.get('pocket', '0')
                    if pnum != '0' and pnum != '':
                        c = clean.get('chain', '').strip()
                        rlabel = clean.get('residue_label', '').strip()
                        pocket_residues[pnum].append((c, rlabel))
                        
            for pnum in sorted(pocket_residues.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                seq = []
                for c, rlabel in pocket_residues[pnum]:
                    key = (c, rlabel)
                    if key in res_dict:
                        seq.append(three_to_one.get(res_dict[key], 'X'))
                if seq:
                    pockets.append(''.join(seq))
        except Exception:
            pass
    return pockets


def extract_features_nise(downloaded_paths, prank_dirs, cache_path="nise_extracted_features.pt", device=None):
    """Extrahuje ESM-2 embeddingy pro NISE vzorky."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    features_dict = {}
    if os.path.exists(cache_path):
        try:
            features_dict = torch.load(cache_path, map_location='cpu', weights_only=False)
            print(f"📦 Načteno {len(features_dict)} extrahovaných features z cache: {cache_path}")
        except Exception:
            features_dict = {}

    missing_uids = [uid for uid in downloaded_paths if uid not in features_dict]
    
    if missing_uids:
        print(f"\n⚡ Extrahuji ESM-2 embeddingy pro {len(missing_uids)} struktur (facebook/esm2_t33_650M_UR50D)...")
        from transformers import AutoTokenizer, EsmModel
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        esm_model = EsmModel.from_pretrained(model_name).to(device)
        esm_model.eval()

        def get_emb(seq_str):
            if not seq_str: return None
            inputs = tokenizer(seq_str, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = esm_model(**inputs)
            emb = out.last_hidden_state[0, 1:-1, :]
            return emb.mean(dim=0).cpu()

        for idx, uid in enumerate(missing_uids):
            pdb_path = downloaded_paths[uid]
            pdir = prank_dirs.get(uid)
            
            full_seq = parse_full_sequence_from_pdb(pdb_path)
            if not full_seq: continue
            full_emb = get_emb(full_seq)
            if full_emb is None: continue
            
            pocket_seqs = parse_pockets_from_prank(pdir, pdb_path) if pdir else []
            pocket_embs = [get_emb(s) for s in pocket_seqs if get_emb(s) is not None]
            
            if not pocket_embs:
                pocket_tensor = full_emb.unsqueeze(0)
            else:
                pocket_tensor = torch.stack(pocket_embs)
                
            features_dict[uid] = {
                'pocket_features': pocket_tensor,
                'full_protein_feature': full_emb
            }
            if (idx + 1) % 15 == 0 or (idx + 1) == len(missing_uids):
                print(f"   [{idx + 1:2d}/{len(missing_uids)}] Dokončeno...")

        torch.save(features_dict, cache_path)
        print(f"💾 Features uloženy do cache: {cache_path}")

    return features_dict


# ============================================================================
# KROK 5: FOLDSEEK 1-NN STRUKTURNÍ BENCHMARK
# ============================================================================

def run_foldseek_benchmark(downloaded_paths, sample_df, train_dir, split_suffix="mil_0.5", use_nr=True, threads=8):
    """Spustí Foldseek 1-NN zarovnání testovacích NISE struktur proti trénovací sadě (s podporou split filtrů)."""
    if shutil.which("foldseek") is None or not train_dir or not os.path.exists(train_dir):
        print("⚠️ Foldseek není dostupný nebo chybí trénovací adresář. Přeskakuji.")
        return None

    print(f"\n🔍 Spouštím Foldseek 1-NN zarovnání...")
    print(f"   Query set (NISE): {len(downloaded_paths)} struktur")
    print(f"   Target databáze (AMICO Train): {train_dir}")

    # 1. Filtrování podle train splitu (pokud je specifikován)
    train_ids = None
    if split_suffix:
        try:
            from dataset import load_split_ids
            tr_set, _, _ = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix, use_nr=use_nr)
            if tr_set:
                train_ids = {os.path.basename(pid).lower().replace('clean_', '').replace('.pdb', '').replace('_merged', '') for pid in tr_set}
                print(f"   🔒 Aplikován filtr trénovacího splitu ({split_suffix}, use_nr={use_nr}): {len(train_ids)} povolených proteinů.")
            else:
                print(f"   ⚠️ Varování: Split soubor pro {split_suffix} (use_nr={use_nr}) nenalezen nebo prázdný. Používám všechny PDB.")
        except Exception as e:
            print(f"   ⚠️ Nepodařilo se načíst split {split_suffix}: {e}")

    # Mapování target PDB na kofaktory
    train_pdbs = glob.glob(os.path.join(train_dir, "**", "*.pdb"), recursive=True)
    target_labels = {}
    for p in train_pdbs:
        t_id = os.path.basename(p).replace('.pdb', '').replace('clean_', '').replace('_MERGED', '').lower()
        if t_id not in target_labels:
            for cname in TARGET_NAMES:
                if cname.lower() in p.lower() or cname.upper() in p.upper():
                    target_labels[t_id] = TARGET_NAMES.index(cname)
                    break

    with tempfile.TemporaryDirectory() as tmp_dir:
        q_dir = os.path.join(tmp_dir, "queries")
        fs_tmp = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(q_dir, exist_ok=True)
        os.makedirs(fs_tmp, exist_ok=True)
        
        for uid, p in downloaded_paths.items():
            dst = os.path.join(q_dir, f"{uid}.pdb")
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                shutil.copy2(abs_p, dst)
            else:
                print(f"⚠️ Chybí PDB soubor: {abs_p}")

        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        cmd = [
            "foldseek", "easy-search",
            q_dir, train_dir, out_tsv, fs_tmp,
            "--format-output", "query,target,evalue,qtmscore,bits",
            "-e", "10.0",
            "--max-seqs", "2000",
            "--threads", str(threads)
        ]
        
        t0 = time.time()
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed_sec = time.time() - t0
        except subprocess.CalledProcessError as e:
            print(f"❌ Chyba Foldseeku (exit code {e.returncode}):")
            if e.stderr:
                print(f"   Stderr: {e.stderr.strip()}")
            if e.stdout:
                print(f"   Stdout: {e.stdout.strip()}")
            return None
        except Exception as e:
            print(f"❌ Neočekávaná chyba při spuštění Foldseeku: {e}")
            return None

        if not os.path.exists(out_tsv) or os.path.getsize(out_tsv) == 0:
            print("⚠️ Foldseek nevygeneroval žádné výstupní zarovnání.")
            return None

        df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "evalue", "qtmscore", "bits"])
        df['query'] = df['query'].apply(lambda x: os.path.basename(str(x)).replace('.pdb', '').lower())
        df['target'] = df['target'].apply(lambda x: os.path.basename(str(x)).replace('.pdb', '').replace('clean_', '').replace('_merged', '').lower())
        
        # Filtrujeme pouze cíle z trénovacího splitu
        if train_ids is not None:
            orig_len = len(df)
            df = df[df['target'].isin(train_ids)]
            print(f"   Hitů před filtrem trénovací sady: {orig_len} -> po filtru train_ids: {len(df)}")
            
        if df.empty:
            print("⚠️ Žádný hit neodpovídá trénovací sadě.")
            top1_preds = {}
        else:
            df = df.sort_values(by=['query', 'qtmscore'], ascending=[True, False])
            top1 = df.drop_duplicates(subset=['query'], keep='first')
            
            top1_preds = {}
            for _, r in top1.iterrows():
                q = r['query']
                t = r['target']
                if t in target_labels:
                    top1_preds[q] = target_labels[t]
                    
        y_true, y_pred = [], []
        preds_by_uid = {}
        for _, row in sample_df.iterrows():
            uid = str(row['entry']).strip()
            if uid not in downloaded_paths: continue
            true_l = TARGET_NAMES.index(row['cofactor'])
            y_true.append(true_l)
            pred_l = top1_preds.get(uid.lower(), -1)
            y_pred.append(pred_l)
            preds_by_uid[uid] = pred_l
            
        return {
            'y_true': y_true, 
            'y_pred': y_pred, 
            'preds_by_uid': preds_by_uid,
            'time_sec': elapsed_sec
        }


# ============================================================================
# KROK 6: INFERENCE AMICO MODELŮ
# ============================================================================

def evaluate_amico_models(sample_df, downloaded_paths, features_dict, models_dir, device):
    """Spustí inferenci AMICO modelů nad připravenými NISE strukturami."""
    models_to_test = [
        ("cross_attention_mil", "cross_attention_mil_best.pt"),
        ("self_attention_mil", "self_attention_mil_best.pt"),
        ("ligand_cross_mil", "ligand_cross_attention_mil_best.pt")
    ]
    
    results = {}
    
    valid_samples = [row for _, row in sample_df.iterrows() if str(row['entry']).strip() in features_dict]
    print(f"\n🧠 Spouštím inferenci AMICO modelů pro {len(valid_samples)} NISE struktur...")
    
    for mkey, ckpt_name in models_to_test:
        ckpt_path = os.path.join(models_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint {ckpt_name} nenalezen, přeskakuji {mkey}.")
            continue
            
        if mkey == "cross_attention_mil":
            from model_cross_attention_mil import CrossAttentionMIL
            model = CrossAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
        elif mkey == "self_attention_mil":
            from model_self_attention_mil import SelfAttentionMIL
            model = SelfAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
        elif mkey == "ligand_cross_mil":
            from model_ligand_cross_attention_mil import LigandCrossAttentionMIL
            model = LigandCrossAttentionMIL(feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
            
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
        model = model.to(device)
        model.eval()
        
        y_true, y_pred = [], []
        preds_by_uid = {}
        
        t0 = time.time()
        with torch.no_grad():
            for row in valid_samples:
                uid = str(row['entry']).strip()
                true_lbl = TARGET_NAMES.index(row['cofactor'])
                feat_data = features_dict[uid]
                
                p_feat = feat_data['pocket_features'].unsqueeze(0).to(device)
                full_feat = feat_data['full_protein_feature'].unsqueeze(0).to(device)
                mask = torch.zeros(1, p_feat.size(1), dtype=torch.bool, device=device)
                
                logits, _ = model(p_feat, mask, full_feat)
                pred_lbl = torch.argmax(logits, dim=-1).item()
                
                y_true.append(true_lbl)
                y_pred.append(pred_lbl)
                preds_by_uid[uid] = pred_lbl
        elapsed_sec = time.time() - t0
                
        results[mkey] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'preds_by_uid': preds_by_uid,
            'time_sec': elapsed_sec
        }
        
    return results


# ============================================================================
# KROK 7: STATISTICKÉ VYHODNOCENÍ A REPORT
# ============================================================================

def df_to_markdown_simple(df):
    """Převede DataFrame na Markdown tabulku i bez balíčku tabulate."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in df.columns) + " |")
        return "\n".join(lines)


def generate_nise_report(all_results, sample_df, out_prefix="nise_benchmark"):
    """Vypíše přehledné výsledky benchmarku na NISE nehomologních enzymecech."""
    print("\n" + "=" * 135)
    print("                VÝSLEDKY NISE BENCHMARKU (NEHOMOLOGNÍ STRUKTURNÍ GENERALIZACE)                ")
    print("=" * 135)
    print(f"{'Model':<22s} | {'Vzorků':<7s} | {'Accuracy':<10s} | {'Macro F1':<10s} | {'Čas (s)':<9s} | {'ms/vzorek':<10s} | {'acetyl-CoA':<10s} | {'ATP':<8s} | {'B12':<8s} | {'FAD':<8s} | {'NAD':<8s}")
    print("-" * 135)

    summary_rows = []
    
    for mkey, res in all_results.items():
        if not res: continue
        y_t = res['y_true']
        y_p = res['y_pred']
        t_sec = res.get('time_sec', 0.0)
        ms_per_sample = (t_sec / max(len(y_t), 1)) * 1000.0
        time_str = f"{t_sec:6.2f} s" if t_sec > 0 else "   N/A  "
        ms_str = f"{ms_per_sample:7.1f} ms" if t_sec > 0 else "   N/A   "
        
        if accuracy_score is not None:
            acc = accuracy_score(y_t, y_p)
            f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
            f1_weighted = f1_score(y_t, y_p, average='weighted', zero_division=0)
            rep = classification_report(y_t, y_p, target_names=TARGET_NAMES, labels=list(range(5)), output_dict=True, zero_division=0)
            per_class = {c: rep.get(c, {}).get('f1-score', 0.0) * 100 for c in TARGET_NAMES}
        else:
            total = len(y_t)
            correct = sum(1 for yt, yp in zip(y_t, y_p) if yt == yp)
            acc = correct / max(total, 1)
            per_class = {}
            f1_list = []
            for i, name in enumerate(TARGET_NAMES):
                tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == i and yp == i)
                fp = sum(1 for yt, yp in zip(y_t, y_p) if yt != i and yp == i)
                fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == i and yp != i)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class[name] = f1 * 100.0
                f1_list.append(f1)
            f1_macro = sum(f1_list) / len(f1_list) if f1_list else 0.0
            f1_weighted = f1_macro
        
        print(f"{mkey:<22s} | {len(y_t):<7d} | {acc * 100:6.2f} %  | {f1_macro * 100:6.2f} %  | {time_str:<9s} | {ms_str:<10s} | {per_class['acetyl-CoA']:6.1f} %   | {per_class['ATP']:6.1f} % | {per_class['B12']:6.1f} % | {per_class['FAD']:6.1f} % | {per_class['NAD']:6.1f} %")
        
        summary_rows.append({
            'Model': mkey,
            'Total_Samples': len(y_t),
            'Accuracy': round(acc * 100, 2),
            'Macro_F1': round(f1_macro * 100, 2),
            'Weighted_F1': round(f1_weighted * 100, 2),
            'Time_Sec': round(t_sec, 2),
            'ms_per_sample': round(ms_per_sample, 1),
            'F1_acetyl-CoA': round(per_class['acetyl-CoA'], 2),
            'F1_ATP': round(per_class['ATP'], 2),
            'F1_B12': round(per_class['B12'], 2),
            'F1_FAD': round(per_class['FAD'], 2),
            'F1_NAD': round(per_class['NAD'], 2)
        })

    print("=" * 135 + "\n")
    
    # Detailní tabulka po jednotlivých proteinech
    det_rows = []
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        r_item = {
            'UniProt_ID': uid,
            'Cofactor_GroundTruth': row['cofactor'],
            'EC_Number': row['ec'],
            'SCOP_Superfamily': row['supfam'],
            'Protein_Name': str(row['protein'])[:50]
        }
        for mkey, res in all_results.items():
            if res and 'preds_by_uid' in res:
                p_idx = res['preds_by_uid'].get(uid, -1)
                p_name = TARGET_NAMES[p_idx] if 0 <= p_idx < 5 else "Miss (-1)"
                r_item[f"{mkey}_Pred"] = p_name
                r_item[f"{mkey}_Correct"] = (p_name == row['cofactor'])
        det_rows.append(r_item)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(f"{out_prefix}_results.csv", index=False)
    
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(f"{out_prefix}_detailed.csv", index=False)
    
    # Markdown report
    with open(f"{out_prefix}_report.md", "w", encoding="utf-8") as f:
        f.write("# AMICO: Benchmark Nehomologních Isofunkčních Enzymů (NISE)\n\n")
        f.write(f"Vygenerováno: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Dataset obsahuje nehomologní enzymy se stejnou funkcí a kofaktorem, ale z odlišných strukturních superfamilií (různé 3D foldy).\n\n")
        f.write("### Souhrnné výsledky\n\n")
        f.write(df_to_markdown_simple(sum_df))
        f.write("\n\n")
        f.write("### Detailní ukázka z predikcí (prvních 15 vzorků)\n\n")
        f.write(df_to_markdown_simple(det_df.head(15)))
        f.write("\n")

    print(f"💾 Výstupy úspěšně uloženy:")
    print(f" - CSV souhrn: {out_prefix}_results.csv")
    print(f" - CSV detail: {out_prefix}_detailed.csv")
    print(f" - Markdown:   {out_prefix}_report.md\n")


# ============================================================================
# HLAVNÍ SPOUŠTĚCÍ FUNKCE (MAIN)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline pro benchmark AMICO na nehomologních enzymech (NISE).")
    parser.add_argument("--nise-tsv", default="NISE_amico_cofactors.tsv", help="Cesta k vyextrahovanému NISE souboru.")
    parser.add_argument("--sample-per-cofactor", type=int, default=15, help="Počet vzorků na kofaktor (default: 15, celkem 75 proteinů).")
    parser.add_argument("--structures-dir", default="nise_structures", help="Složka pro stažené AlphaFold PDB struktury.")
    parser.add_argument("--models-dir", default=PROJECT_ROOT, help="Složka s modely (*_best.pt).")
    parser.add_argument("--train-dir", default=os.path.join(PROJECT_ROOT, "structures"), help="Trénovací struktury pro Foldseek.")
    parser.add_argument("--split-suffix", default="mil_0.5", help="Přípona splitu pro trénovací sadu Foldseeku (např. mil_0.5).")
    parser.add_argument("--use-nr", action="store_true", default=True, help="Použít Non-Redundant variantu trénovací sady (např. train_mil_0.5_nr0.95).")
    parser.add_argument("--no-nr", dest="use_nr", action="store_false", help="Vypnout Non-Redundant filtr (použít train_mil_0.5 bez NR).")
    parser.add_argument("--no-split-filter", action="store_true", help="Vypnout filtr splitů pro Foldseek (prohledávat celou složku structures bez omezení).")
    parser.add_argument("--prank-exec", default=None, help="Cesta k binárce P2Rank (prank).")
    parser.add_argument("--out-prefix", default="nise_benchmark", help="Prefix výstupních souborů.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-download", action="store_true", help="Přeskočí stahování struktur z AFDB.")
    parser.add_argument("--skip-foldseek", action="store_true", help="Přeskočí Foldseek benchmark.")
    parser.add_argument("--clean-csv", default=None, help="Cesta k souboru predikcí CLEANu (*_maxsep.csv) pro zahrnutí do srovnání.")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("      AMICO: PIPELINE PRO BENCHMARK NEHOMOLOGNÍCH ENZYMŮ (NISE)      ")
    print("=" * 80)
    print(f"Zařízení: {device} | Počet vzorků na kofaktor: {args.sample_per_cofactor}")
    if not args.no_split_filter:
        print(f"Foldseek trénovací filtr: split={args.split_suffix}, use_nr={args.use_nr}")
    else:
        print("Foldseek trénovací filtr: VYPNUTO (všechny PDB v structures/)")

    # 1. Výběr vzorku
    sample_df = select_nise_sample(args.nise_tsv, sample_per_cofactor=args.sample_per_cofactor)

    # 2. Stažení struktur
    downloaded_paths = {}
    if not args.skip_download:
        downloaded_paths = download_alphafold_structures(sample_df, structures_dir=args.structures_dir)
    else:
        # Použijeme již existující PDB soubory
        for _, row in sample_df.iterrows():
            uid = str(row['entry']).strip()
            cof = str(row['cofactor']).strip()
            p = os.path.abspath(os.path.join(args.structures_dir, cof, f"{uid}.pdb"))
            if os.path.exists(p):
                downloaded_paths[uid] = p

    if not downloaded_paths:
        print("❌ Žádné PDB struktury nebyly staženy ani nalezeny v cache.")
        sys.exit(1)

    # 3. P2Rank kapsy
    prank_dirs = run_p2rank_for_structures(downloaded_paths, prank_exec=args.prank_exec, threads=args.threads)

    # 4. ESM-2 extrakce
    features_dict = extract_features_nise(downloaded_paths, prank_dirs, device=device)

    # 5. Foldseek
    all_results = {}
    if not args.skip_foldseek:
        sfx = None if args.no_split_filter else args.split_suffix
        fs_res = run_foldseek_benchmark(
            downloaded_paths, sample_df, 
            train_dir=args.train_dir, 
            split_suffix=sfx,
            use_nr=args.use_nr,
            threads=args.threads
        )
        if fs_res:
            all_results['foldseek_1nn'] = fs_res

    # 6. AMICO modely
    amico_res = evaluate_amico_models(sample_df, downloaded_paths, features_dict, models_dir=args.models_dir, device=device)
    all_results.update(amico_res)

    # 6.5 CLEAN benchmark (pokud je zadán nebo nalezen soubor předpočítaných predikcí CLEAN)
    clean_csv_cand = [
        getattr(args, 'clean_csv', None),
        "nise_clean_input_maxsep.csv",
        "benchmarks/nise_clean_input_maxsep.csv",
        os.path.join(PROJECT_ROOT, "nise_clean_input_maxsep.csv"),
        os.path.join(PROJECT_ROOT, "benchmarks", "nise_clean_input_maxsep.csv"),
    ]
    clean_csv_path = next((p for p in clean_csv_cand if p and os.path.exists(p)), None)
    if clean_csv_path:
        print(f"\n🧪 Vyhodnocuji model CLEAN ze souboru {clean_csv_path}...")
        try:
            from benchmarks.clean_nise_benchmark import load_kegg_ec_mapping, parse_clean_maxsep_csv, evaluate_clean_on_nise
            ec_map = load_kegg_ec_mapping()
            c_preds = parse_clean_maxsep_csv(clean_csv_path)
            c_res = evaluate_clean_on_nise(sample_df, c_preds, ec_map)
            all_results['clean_ec_contrastive'] = c_res
        except Exception as e:
            print(f"⚠️ Chyba při vyhodnocení CLEANu: {e}")

    # 7. Vyhodnocení a report
    if all_results:
        generate_nise_report(all_results, sample_df, out_prefix=args.out_prefix)
    else:
        print("⚠️ Žádný model nebyl úspěšně vyhodnocen.")


if __name__ == '__main__':
    main()
