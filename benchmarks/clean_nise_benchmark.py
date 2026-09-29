#!/usr/bin/env python3
"""
CLEAN (Contrastive Learning-enabled Enzyme Annotation) NISE Benchmark
====================================================================
Hodnotí sekvenční model CLEAN na datasetu nehomologních isofunkčních enzymů (NISE).

Postup:
1. Načte testovací NISE vzorky (nise_selected_sample.tsv).
2. Připraví / vyextrahuje sekvence do formátu FASTA (z PDB struktur nebo UniProt API).
3. Spustí CLEAN inferenci (pokud je dostupná binárka/skript) NEBO načte predikce z <vstup>_maxsep.csv.
4. Namapuje predikované EC kódy na 5 cílových kofaktorů AMICO:
   - acetyl-CoA (KEGG C00024, C00010)
   - ATP (KEGG C00002)
   - B12 (KEGG C00114, C00194, C00282, C06453)
   - FAD (KEGG C00016, C01352)
   - NAD (KEGG C00003, C00004, C00005, C00006)
5. Spočítá metriky (Accuracy, Macro F1, Weighted F1, Per-class F1) a změří čas běhu.
6. Exportuje přehledné CSV a Markdown reporty pro srovnání s Foldseekem a AMICO.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import ssl
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Přidání kořenového adresáře projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def compute_classification_metrics(y_true, y_pred, target_names):
    """Vypočítá Accuracy, Macro F1, Weighted F1 a per-class F1 (s fallbackem bez sklearn)."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        rep = classification_report(y_true, y_pred, target_names=target_names, labels=list(range(len(target_names))), output_dict=True, zero_division=0)
        per_class = {c: rep.get(c, {}).get('f1-score', 0.0) * 100 for c in target_names}
        return acc, f1_macro, f1_weighted, per_class
    except ImportError:
        total = len(y_true)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / max(total, 1)
        per_class = {}
        f1_list = []
        for i, name in enumerate(target_names):
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp == i)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != i and yp == i)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp != i)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[name] = f1 * 100.0
            f1_list.append(f1)
        f1_macro = sum(f1_list) / len(f1_list) if f1_list else 0.0
        f1_weighted = f1_macro
        return acc, f1_macro, f1_weighted, per_class

# Definice KEGG ID sloučenin pro naše kofaktory
KEGG_COFACTOR_COMPOUNDS = {
    'acetyl-CoA': ['cpd:C00024', 'cpd:C00010'],
    'ATP': ['cpd:C00002'],
    'B12': ['cpd:C00114', 'cpd:C00194', 'cpd:C00282', 'cpd:C06453'],
    'FAD': ['cpd:C00016', 'cpd:C01352'],
    'NAD': ['cpd:C00003', 'cpd:C00004', 'cpd:C00005', 'cpd:C00006']
}

# 3-to-1 aminokyselinový slovník
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O'
}


def load_kegg_ec_mapping(cache_file="kegg_ec_to_cofactor.json"):
    """
    Načte nebo stáhne z KEGG REST API mapování EC čísel na kofaktory.
    """
    cache_path = PROJECT_ROOT / cache_file
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Převod na sety
            ec_map = {k: set(v) for k, v in data.items()}
            print(f"📦 Načteno {len(ec_map)} EC->Kofaktor mapování z lokální cache ({cache_file}).")
            return ec_map
        except Exception as e:
            print(f"⚠️ Chyba při čtení cache ({e}), stahuji znovu z KEGG API...")

    print("🌐 Stahuji EC -> Compound mapování z KEGG API (https://rest.kegg.jp/link/cpd/ec)...")
    cpd_to_cof = {}
    for cof, cpds in KEGG_COFACTOR_COMPOUNDS.items():
        for cpd in cpds:
            cpd_to_cof[cpd] = cof

    ec_to_cofs = defaultdict(set)
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = "https://rest.kegg.jp/link/cpd/ec"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AMICO-Benchmark)'})
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            content = resp.read().decode('utf-8')

        for line in content.splitlines():
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ec = parts[0].replace('ec:', '').strip()
                cpd = parts[1].strip()
                if cpd in cpd_to_cof:
                    ec_to_cofs[ec].add(cpd_to_cof[cpd])

        print(f"✅ Staženo {len(ec_to_cofs)} relevantních EC čísel vázajících AMICO kofaktory.")
        
        # Uložení do cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({k: list(v) for k, v in ec_to_cofs.items()}, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Nelze stáhnout z KEGG API ({e}). Používám vestavěná heuristická pravidla.")
        
    return ec_to_cofs


def extract_sequence_from_pdb(pdb_path):
    """Extrahuje sekvenci aminokyselin z PDB souboru."""
    seen_residues = set()
    seq_list = []
    try:
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("ATOM  "):
                    chain = line[21].strip()
                    res_num = line[22:27].strip()
                    res_name = line[17:20].strip()
                    key = (chain, res_num)
                    if key not in seen_residues:
                        seen_residues.add(key)
                        seq_list.append(THREE_TO_ONE.get(res_name, 'X'))
        return "".join(seq_list)
    except Exception:
        return None


def fetch_sequence_uniprot(uniprot_id):
    """Stáhne kanonickou sekvenci z UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AMICO-Benchmark)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            fasta = resp.read().decode('utf-8')
            lines = fasta.splitlines()
            seq = "".join(l.strip() for l in lines if not l.startswith(">"))
            return seq
    except Exception:
        return None


def prepare_nise_fasta(sample_df, structures_dir="nise_structures", out_fasta="nise_clean_input.fasta"):
    """
    Vygeneruje FASTA soubor ze sekvencí testovacích NISE proteinů.
    """
    out_path = Path(out_fasta)
    out_csv = out_path.with_suffix('.csv')
    if out_path.exists() and out_path.stat().st_size > 0 and out_csv.exists() and out_csv.stat().st_size > 0:
        print(f"⚡ Vstupy FASTA a CSV již existují ({out_path}), používám z cache.")
        for d in [Path("data"), Path("benchmarks/data")]:
            try:
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_path, d / out_path.name)
                shutil.copy2(out_csv, d / out_csv.name)
            except Exception:
                pass
        return str(out_path.resolve())

    print(f"\n📝 Příprava FASTA souboru pro CLEAN ({out_path})...")
    
    records = []
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        cof = str(row['cofactor']).strip()
        
        # 1. Zkusíme z PDB
        pdb_p = Path(structures_dir) / cof / f"{uid}.pdb"
        seq = None
        if pdb_p.exists():
            seq = extract_sequence_from_pdb(pdb_p)
            
        # 2. Fallback na UniProt API
        if not seq:
            seq = fetch_sequence_uniprot(uid)
            
        if seq:
            records.append((uid, seq))
            
    with open(out_path, "w", encoding="utf-8") as f:
        for uid, seq in records:
            f.write(f">{uid}\n{seq}\n")
            
    print(f"✅ Uloženo {len(records)} sekvencí do {out_path}.")

    # Uložení CSV souboru pro CLEAN
    import shutil
    out_csv = out_path.with_suffix('.csv')
    entry_to_ec = dict(zip(sample_df['entry'].astype(str).str.strip(), sample_df['ec'].astype(str).str.strip()))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Entry,EC number,Sequence\n")
        for uid, seq in records:
            ec = entry_to_ec.get(uid, "1.1.1.1")
            f.write(f"{uid},{ec},{seq}\n")
    print(f"✅ Uloženo {len(records)} záznamů do CSV: {out_csv}.")

    for d in [Path("data"), Path("benchmarks/data")]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, d / out_path.name)
            shutil.copy2(out_csv, d / out_csv.name)
        except Exception:
            pass

    return str(out_path.resolve())


def parse_clean_maxsep_csv(clean_csv_path):
    """
    Parsuje standardní výstupní soubor CLEANu (*_maxsep.csv).
    Formát:
    ID, EC:x.x.x.x/score, EC:y.y.y.y/score, ...
    """
    clean_preds = {}
    if not os.path.exists(clean_csv_path):
        print(f"❌ Soubor s predikcemi CLEANu neexistuje: {clean_csv_path}")
        return clean_preds

    with open(clean_csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if not parts: continue
            
            # ID vzorku
            sample_id = parts[0].replace('.pdb', '').replace('clean_', '').replace('_MERGED', '')
            # Oříznutí eventuálního chain suffixu
            clean_id = sample_id.split('_Chain')[0].strip()
            
            ec_scores = {}
            for token in parts[1:]:
                if "/" not in token: continue
                ec_part, score_part = token.split("/", 1)
                ec = ec_part.replace("EC:", "").strip()
                try:
                    score = float(score_part)
                    ec_scores[ec] = score
                except ValueError:
                    continue
                    
            clean_preds[clean_id] = ec_scores

    print(f"📦 Načteno {len(clean_preds)} predikcí z CLEAN souboru: {clean_csv_path}")
    return clean_preds


def predict_cofactor_from_ecs(ec_scores, ec_to_cofs):
    """
    Agreguje skóre predikovaných EC čísel pro 5 kofaktorových tříd AMICO.
    Vrací predikovaný index třídy (0..4) nebo -1 (pokud nic neodpovídá).
    """
    scores = np.zeros(len(TARGET_NAMES), dtype=float)
    
    for ec, conf in ec_scores.items():
        # 1. Přímé mapování z KEGG
        if ec in ec_to_cofs:
            for cof in ec_to_cofs[ec]:
                if cof in TARGET_NAMES:
                    c_idx = TARGET_NAMES.index(cof)
                    scores[c_idx] += conf
        else:
            # 2. Heuristické pravidlo pro neúplné / specifické kódy
            parts = ec.split('.')
            if len(parts) >= 2:
                # 1.x.1.x -> obvykle NAD/NADP dehydrogenázy
                if parts[0] == '1' and len(parts) >= 3 and parts[2] == '1':
                    scores[TARGET_NAMES.index('NAD')] += conf * 0.7
                # 1.x.99.x nebo oxidoreduktázy s FAD
                elif parts[0] == '1' and len(parts) >= 3 and parts[2] in ['99', '3']:
                    scores[TARGET_NAMES.index('FAD')] += conf * 0.5
                # 2.7.x.x -> kinázy vázající ATP
                elif parts[0] == '2' and parts[1] == '7':
                    scores[TARGET_NAMES.index('ATP')] += conf * 0.8
                # 2.3.1.x -> acyltransferázy (acetyl-CoA)
                elif parts[0] == '2' and parts[1] == '3':
                    scores[TARGET_NAMES.index('acetyl-CoA')] += conf * 0.8

    if np.max(scores) <= 0.0:
        return -1, scores
        
    pred_idx = int(np.argmax(scores))
    return pred_idx, scores


def evaluate_clean_on_nise(sample_df, clean_preds, ec_to_cofs):
    """
    Vyhodnotí predikce CLEANu proti testovací sadě NISE.
    """
    y_true, y_pred = [], []
    preds_by_uid = {}
    matched_count = 0
    
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        true_l = TARGET_NAMES.index(row['cofactor'])
        y_true.append(true_l)
        
        # Najdeme predikci pro daný UniProt ID
        ec_scores = None
        for k in [uid, uid.lower(), uid.upper()]:
            if k in clean_preds:
                ec_scores = clean_preds[k]
                break
                
        if ec_scores:
            matched_count += 1
            pred_l, _ = predict_cofactor_from_ecs(ec_scores, ec_to_cofs)
        else:
            pred_l = -1
            
        y_pred.append(pred_l)
        preds_by_uid[uid] = pred_l

    print(f"📊 Spárováno {matched_count} / {len(sample_df)} vzorků s výstupem CLEANu.")
    return {'y_true': y_true, 'y_pred': y_pred, 'preds_by_uid': preds_by_uid}


def evaluate_ground_truth_ec_upper_bound(sample_df, ec_to_cofs):
    """
    Teoretický horní limit pro CLEAN:
    Co kdyby CLEAN predikoval 100% správně oficiální EC číslo anotované v NISE?
    Ukazuje maximální dosažitelnou přesnost mapování EC -> Kofaktor na tomto datasetu.
    """
    y_true, y_pred = [], []
    preds_by_uid = {}
    
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        ec = str(row['ec']).strip()
        true_l = TARGET_NAMES.index(row['cofactor'])
        y_true.append(true_l)
        
        pred_l, _ = predict_cofactor_from_ecs({ec: 1.0}, ec_to_cofs)
        y_pred.append(pred_l)
        preds_by_uid[uid] = pred_l
        
    return {'y_true': y_true, 'y_pred': y_pred, 'preds_by_uid': preds_by_uid}


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


def print_clean_report(all_results, sample_df, out_prefix="clean_nise_benchmark"):
    """Vytiskne a uloží report benchmarku."""
    print("\n" + "=" * 135)
    print("                    VÝSLEDKY NISE BENCHMARKU: CLEAN (EC Contrastive Learning)                    ")
    print("=" * 135)
    print(f"{'Implementace / Model':<28s} | {'Vzorků':<7s} | {'Accuracy':<10s} | {'Macro F1':<10s} | {'Čas (s)':<9s} | {'ms/vzorek':<10s} | {'acetyl-CoA':<10s} | {'ATP':<8s} | {'B12':<8s} | {'FAD':<8s} | {'NAD':<8s}")
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
        
        acc, f1_macro, f1_weighted, per_class = compute_classification_metrics(y_t, y_p, TARGET_NAMES)
        
        print(f"{mkey:<28s} | {len(y_t):<7d} | {acc * 100:6.2f} %  | {f1_macro * 100:6.2f} %  | {time_str:<9s} | {ms_str:<10s} | {per_class['acetyl-CoA']:6.1f} %   | {per_class['ATP']:6.1f} % | {per_class['B12']:6.1f} % | {per_class['FAD']:6.1f} % | {per_class['NAD']:6.1f} %")
        
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
    
    # Detailní CSV
    det_rows = []
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        r_item = {
            'UniProt_ID': uid,
            'Cofactor_GroundTruth': row['cofactor'],
            'Annotated_EC': row['ec'],
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
    
    # Markdown
    with open(f"{out_prefix}_report.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark CLEAN na Nehomologních Isofunkčních Enzymech (NISE)\n\n")
        f.write(f"Vygenerováno: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Srovnání modelu CLEAN (kontrastivní sekvenční predikce EC čísel) na datech nehomologních enzymů.\n\n")
        f.write("### Souhrnné výsledky\n\n")
        f.write(df_to_markdown_simple(sum_df))
        f.write("\n\n")
        f.write("### Ukázka detailních predikcí (prvních 15 vzorků)\n\n")
        f.write(df_to_markdown_simple(det_df.head(15)))
        f.write("\n")

    print(f"💾 Výstupy úspěšně uloženy:")
    print(f" - CSV souhrn: {out_prefix}_results.csv")
    print(f" - CSV detail: {out_prefix}_detailed.csv")
    print(f" - Markdown:   {out_prefix}_report.md\n")


def main():
    parser = argparse.ArgumentParser(description="Samostatný benchmark CLEANu na nehomologních enzymech (NISE).")
    parser.add_argument("--sample-tsv", default="nise_selected_sample.tsv", help="Cesta k vybranému vzorku NISE.")
    parser.add_argument("--structures-dir", default="nise_structures", help="Složka s PDB strukturami pro extrakci sekvencí.")
    parser.add_argument("--fasta-out", default="nise_clean_input.fasta", help="Cesta k vygenerovanému FASTA souboru.")
    parser.add_argument("--clean-csv", default=None, help="Cesta k existujícímu souboru predikcí CLEANu (*_maxsep.csv).")
    parser.add_argument("--clean-script", default=None, help="Cesta ke skriptu/binárce pro spuštění CLEANu (např. clean_infer).")
    parser.add_argument("--kegg-cache", default="kegg_ec_to_cofactor.json", help="Cesta k souboru s cache KEGG mapování.")
    parser.add_argument("--out-prefix", default="clean_nise_benchmark", help="Prefix výstupních reportů.")
    parser.add_argument("--only-generate-fasta", action="store_true", help="Pouze vygeneruje FASTA soubor a skončí.")
    parser.add_argument("--run-infer", action="store_true", help="Automaticky spustit inferenci CLEANu pomocí benchmarks/run_clean_infer.py.")
    
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / args.sample_tsv
    if not sample_path.exists():
        print(f"❌ Soubor vzorku {sample_path} nenalezen.")
        sys.exit(1)
        
    sample_df = pd.read_csv(sample_path, sep='\t')
    print("=" * 80)
    print("        BENCHMARK CLEAN: NEHOMOLOGNÍ ISOFUNKČNÍ ENZYMY (NISE)        ")
    print("=" * 80)
    print(f"Načteno {len(sample_df)} testovacích NISE proteinů.")

    # 1. Příprava FASTA
    fasta_path = prepare_nise_fasta(sample_df, structures_dir=args.structures_dir, out_fasta=args.fasta_out)
    if args.only_generate_fasta:
        print(f"🏁 FASTA soubor {fasta_path} připraven pro spuštění CLEANu.")
        return

    # 2. Načtení KEGG mapování
    ec_to_cofs = load_kegg_ec_mapping(args.kegg_cache)

    all_results = {}

    # 3. Teoretický horní limit (kdyby CLEAN predikoval přesné anotované EC z databáze)
    t0 = time.time()
    gt_ec_res = evaluate_ground_truth_ec_upper_bound(sample_df, ec_to_cofs)
    gt_ec_res['time_sec'] = time.time() - t0
    all_results['clean_upper_bound_oracle'] = gt_ec_res

    # 4. Spuštění nebo načtení skutečných CLEAN predikcí
    stem = Path(args.fasta_out).stem
    clean_csv_candidates = [
        args.clean_csv,
        f"{stem}_maxsep.csv",
        f"results/{stem}_maxsep.csv",
        f"data/{stem}_maxsep.csv",
        "nise_clean_input_maxsep.csv",
        "results/nise_clean_input_maxsep.csv",
        "extracted_sequences_maxsep.csv",
        "test_all_sequences_maxsep.csv"
    ]
    
    clean_csv = next((c for c in clean_csv_candidates if c and os.path.exists(c)), None)
    
    if not clean_csv and args.clean_script:
        print(f"\n🚀 Spouštím CLEAN inferenci přes skript {args.clean_script}...")
        t0 = time.time()
        try:
            cmd = [sys.executable, args.clean_script, "--fasta", fasta_path]
            subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            pred_csv = f"{Path(fasta_path).stem}_maxsep.csv"
            if os.path.exists(pred_csv):
                clean_csv = pred_csv
                all_results['clean_inference'] = evaluate_clean_on_nise(sample_df, parse_clean_maxsep_csv(clean_csv), ec_to_cofs)
                all_results['clean_inference']['time_sec'] = elapsed
        except Exception as e:
            print(f"❌ Chyba při spuštění CLEANu: {e}")
            
    elif clean_csv:
        print(f"\n📂 Načítám předpočítané predikce CLEANu z: {clean_csv}")
        t0 = time.time()
        preds = parse_clean_maxsep_csv(clean_csv)
        clean_res = evaluate_clean_on_nise(sample_df, preds, ec_to_cofs)
        clean_res['time_sec'] = time.time() - t0
        all_results['clean_predictions'] = clean_res
    elif getattr(args, 'run_infer', False):
        print("\n🚀 Spouštím automatickou CLEAN inferenci...")
        t0 = time.time()
        try:
            from benchmarks.run_clean_infer import find_clean_package
            infer_fn, clean_root = find_clean_package()
            if infer_fn:
                stem = Path(fasta_path).stem
                work_dir = Path(clean_root) if clean_root else Path.cwd()
                (work_dir / "data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(fasta_path, work_dir / "data" / f"{stem}.fasta")
                curr = os.getcwd()
                if clean_root: os.chdir(clean_root)
                infer_fn("split100", stem, report_metrics=False)
                os.chdir(curr)
                out_f = work_dir / "results" / f"{stem}_maxsep.csv"
                if out_f.exists():
                    preds = parse_clean_maxsep_csv(str(out_f))
                    clean_res = evaluate_clean_on_nise(sample_df, preds, ec_to_cofs)
                    clean_res['time_sec'] = time.time() - t0
                    all_results['clean_inference'] = clean_res
        except Exception as e:
            print(f"❌ Chyba při automatické inferenci CLEANu: {e}")
    else:
        print("\nℹ️  Žádný existující soubor predikcí CLEANu (*_maxsep.csv) nebyl nalezen.")
        print(f"   Vygenerovaný FASTA soubor: {fasta_path}")
        print("   Pro spuštění CLEAN inference použijte:")
        print(f"     python benchmarks/run_clean_infer.py --fasta {fasta_path}")
        print("   Nebo přímo v prostředí CLEANu:")
        print("     python -c \"from CLEAN.infer import infer_maxsep; infer_maxsep('split100', 'nise_clean_input')\"")
        print("   A poté znovu spusťte tento skript:")
        print("     python benchmarks/clean_nise_benchmark.py --clean-csv nise_clean_input_maxsep.csv")

    # 5. Vygenerování souhrnného reportu
    if all_results:
        print_clean_report(all_results, sample_df, out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
