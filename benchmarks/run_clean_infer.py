#!/usr/bin/env python3
"""
CLEAN Inference Runner
======================
Automatizovaný spouštěč inference modelu CLEAN (Contrastive Learning-enabled Enzyme Annotation).

Použití:
    python run_clean_infer.py --fasta nise_clean_input.fasta
    # Nebo:
    python benchmarks/run_clean_infer.py --fasta nise_clean_input.fasta --clean-dir /cesta/k/CLEAN
"""

import os
import sys
import shutil
import argparse
import inspect
from pathlib import Path

def find_clean_package(user_clean_dir=None):
    """Najde balíček nebo repositář CLEAN a přidá jej do sys.path."""
    if user_clean_dir and os.path.exists(user_clean_dir):
        sys.path.insert(0, str(Path(user_clean_dir).resolve()))
        if os.path.exists(os.path.join(user_clean_dir, "src")):
            sys.path.insert(0, str(Path(user_clean_dir, "src").resolve()))

    try:
        from CLEAN.infer import infer_maxsep
        return infer_maxsep, None
    except ImportError:
        pass

    # Hledáme v obvyklých umístěních na disku / clusteru
    candidates = [
        Path.cwd() / "CLEAN",
        Path.cwd().parent / "CLEAN",
        Path.cwd().parent / "Clean",
        Path.home() / "CLEAN",
        Path("/storage/brno2/home/urbany/CLEAN"),
        Path("/auto/brno2/brno2/urbany/CLEAN"),
        Path("/auto/brno2/brno2/urbany/Dev_amico/CLEAN"),
    ]

    for cand in candidates:
        if cand.exists() and (cand / "CLEAN" / "infer.py").exists():
            sys.path.insert(0, str(cand.resolve()))
            try:
                from CLEAN.infer import infer_maxsep
                return infer_maxsep, str(cand.resolve())
            except ImportError:
                pass
        elif cand.exists() and (cand / "src" / "CLEAN" / "infer.py").exists():
            sys.path.insert(0, str((cand / "src").resolve()))
            try:
                from CLEAN.infer import infer_maxsep
                return infer_maxsep, str(cand.resolve())
            except ImportError:
                pass

    return None, None


def parse_fasta_records(fasta_path):
    """Načte ID a sekvence z FASTA souboru."""
    records = []
    with open(fasta_path, 'r', encoding='utf-8', errors='ignore') as f:
        cur_id = None
        cur_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id:
            records.append((cur_id, "".join(cur_seq)))
    return records


def load_ec_annotations():
    """Pokusí se načíst oficiální EC anotace z nise_selected_sample.tsv."""
    candidates = [
        Path.cwd() / "nise_selected_sample.tsv",
        Path.cwd().parent / "nise_selected_sample.tsv",
        Path.cwd() / "data" / "nise_selected_sample.tsv",
        Path.cwd().parent / "data" / "nise_selected_sample.tsv",
        Path("/auto/brno2/brno2/urbany/Dev_amico/nise_selected_sample.tsv"),
        Path("/storage/brno2/home/urbany/Dev_amico/nise_selected_sample.tsv"),
    ]
    for c in candidates:
        if c.exists():
            try:
                import pandas as pd
                df = pd.read_csv(c, sep='\t')
                if 'entry' in df.columns and 'ec' in df.columns:
                    ec_map = dict(zip(df['entry'].astype(str).str.strip(), df['ec'].astype(str).str.strip()))
                    print(f"📖 Načteno {len(ec_map)} anotovaných EC z {c.name}")
                    return ec_map
            except Exception:
                pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Spuštění inference CLEAN pro zadaný FASTA soubor.")
    parser.add_argument("--fasta", default="nise_clean_input.fasta", help="Vstupní FASTA soubor.")
    parser.add_argument("--train-data", default="split100", help="Trénovací reference CLEANu (default: split100).")
    parser.add_argument("--clean-dir", default=None, help="Cesta ke složce s repositářem CLEAN (pokud není instalován v pip).")
    parser.add_argument("--out-csv", default=None, help="Výstupní CSV soubor s predikcemi (*_maxsep.csv).")
    
    args = parser.parse_args()

    # Vyhledání vstupního souboru
    fasta_cand = [
        Path(args.fasta),
        Path.cwd() / args.fasta,
        Path.cwd().parent / args.fasta,
        Path.cwd() / "data" / args.fasta,
        Path.cwd() / "benchmarks" / args.fasta,
    ]
    fasta_path = next((p.resolve() for p in fasta_cand if p.exists()), None)
    if not fasta_path:
        print(f"❌ Vstupní FASTA soubor neexistuje: {args.fasta}")
        sys.exit(1)

    print("=" * 80)
    print("                    SPOUŠTĚČ INFERENCE MODELU CLEAN                     ")
    print("=" * 80)
    print(f"Vstupní FASTA: {fasta_path}")

    infer_maxsep_fn, clean_root = find_clean_package(args.clean_dir)

    if infer_maxsep_fn is None:
        print("\n❌ Modul 'CLEAN.infer' nebyl nalezen v Python prostředí ani v obvyklých cestách.")
        print("💡 Řešení:")
        print("   1. Ujistěte se, že máte aktivní správné prostředí (např. Singularity/conda 'clean').")
        print("   2. Zadejte cestu k repositáři přes --clean-dir.")
        sys.exit(1)

    if clean_root:
        print(f"✅ Nalezen repositář CLEAN: {clean_root}")
    else:
        print(f"✅ CLEAN načten z nainstalovaného Python balíčku.")

    stem = fasta_path.stem
    work_dir = Path(clean_root) if clean_root else Path.cwd()
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Zajištění FASTA souboru v data/
    target_fasta = data_dir / f"{stem}.fasta"
    if target_fasta.resolve() != fasta_path.resolve():
        print(f"📁 Kopíruji FASTA do pracovní složky: {target_fasta}")
        shutil.copy2(fasta_path, target_fasta)
    local_data_fasta = Path.cwd() / "data" / f"{stem}.fasta"
    if local_data_fasta.resolve() != target_fasta.resolve():
        local_data_fasta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fasta_path, local_data_fasta)

    # 2. Zajištění CSV souboru v data/ (CLEAN infer_maxsep vyžaduje data/<test_data>.csv!)
    target_csv = data_dir / f"{stem}.csv"
    existing_csv = None
    for cand_csv in [fasta_path.with_suffix('.csv'), Path.cwd() / f"{stem}.csv", Path.cwd().parent / f"{stem}.csv"]:
        if cand_csv.exists() and cand_csv.stat().st_size > 0:
            existing_csv = cand_csv
            break

    # Detekce formátu referenčního souboru (split100.csv)
    ref_csv = data_dir / f"{args.train_data}.csv"
    delimiter = ','
    if ref_csv.exists():
        with open(ref_csv, 'r', encoding='utf-8', errors='ignore') as rf:
            first_line = rf.readline()
            if '\t' in first_line:
                delimiter = '\t'
            print(f"📋 Formát referenčního {ref_csv.name}: delimiter='{repr(delimiter)[1:-1]}', header: {first_line.strip()[:60]}")

    if existing_csv and existing_csv.resolve() != target_csv.resolve():
        print(f"📁 Používám existující CSV soubor: {existing_csv} -> {target_csv}")
        shutil.copy2(existing_csv, target_csv)
    elif not target_csv.exists():
        print(f"📝 Vytvářím chybějící CSV soubor vyžadovaný CLEANem: {target_csv}")
        records = parse_fasta_records(fasta_path)
        ec_map = load_ec_annotations()
        with open(target_csv, 'w', encoding='utf-8') as f:
            f.write(f"Entry{delimiter}EC number{delimiter}Sequence\n")
            for uid, seq in records:
                ec = ec_map.get(uid, "1.1.1.1")
                f.write(f"{uid}{delimiter}{ec}{delimiter}{seq}\n")
        print(f"✅ Vytvořeno {target_csv} ({len(records)} sekvencí).")

    # Zajištění kopie v ./data/<stem>.csv pokud se CWD liší
    local_data_csv = Path.cwd() / "data" / f"{stem}.csv"
    if local_data_csv.resolve() != target_csv.resolve():
        local_data_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_csv, local_data_csv)

    # 3. Kontrola / výpis implementace get_ec_id_dict pro transparentnost
    try:
        from CLEAN import utils
        if hasattr(utils, 'get_ec_id_dict'):
            print("\n🔍 [CLEAN interní kód] utils.get_ec_id_dict:")
            src = inspect.getsource(utils.get_ec_id_dict).strip().splitlines()
            for line in src[:15]:
                print(f"   {line}")
    except Exception as e:
        pass

    # 4. Kontrola ESM-1b embeddingů
    esm_dir = data_dir / "esm_data"
    esm_dir.mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "data" / "esm_data").mkdir(parents=True, exist_ok=True)
    esm_file = esm_dir / f"{stem}.pt"

    if not esm_file.exists():
        print(f"\n🧬 ESM-1b embeddingy pro '{stem}' nebyly nalezeny v {esm_file}.")
        print("   Spouštím retrive_esm1b_embedding z balíčku CLEAN...")
        curr_cwd = os.getcwd()
        try:
            if clean_root:
                os.chdir(clean_root)
            try:
                from CLEAN.utils import retrive_esm1b_embedding
                retrive_esm1b_embedding(stem)
                print("✅ ESM-1b embeddingy úspěšně spočteny.")
            except ImportError:
                try:
                    from CLEAN.utils import retrieve_esm1b_embedding
                    retrieve_esm1b_embedding(stem)
                    print("✅ ESM-1b embeddingy úspěšně spočteny.")
                except Exception as e_ret:
                    print(f"ℹ️ Nelze přímo zavolat retrive_esm1b_embedding ({e_ret}), infer_maxsep je spočte sám.")
        except Exception as e:
            print(f"⚠️ Výpočet embeddingů: {e}")
        finally:
            os.chdir(curr_cwd)
    else:
        print(f"⚡ Nalezeny existující ESM embeddingy: {esm_file}")

    # 5. Spuštění infer_maxsep
    print(f"\n🧠 Spouštím CLEAN inferenci (train_data={args.train_data}, test_data={stem})...")
    print("   (Může trvat několik minut v závislosti na GPU/CPU a velikosti sady)")

    curr_cwd = os.getcwd()
    try:
        if clean_root:
            os.chdir(clean_root)
        try:
            infer_maxsep_fn(train_data=args.train_data, test_data=stem, report_metrics=False)
        except TypeError:
            try:
                infer_maxsep_fn(args.train_data, stem)
            except Exception:
                infer_maxsep_fn(train_data=args.train_data, test_data=stem)
    finally:
        os.chdir(curr_cwd)

    # 6. Nalezení a zkopírování vygenerovaného výstupu
    candidates_out = [
        work_dir / "results" / f"{stem}_maxsep.csv",
        Path.cwd() / "results" / f"{stem}_maxsep.csv",
        work_dir / f"{stem}_maxsep.csv",
        Path.cwd() / f"{stem}_maxsep.csv",
        data_dir / f"{stem}_maxsep.csv",
        work_dir / "results" / f"{stem}_{args.train_data}_maxsep.csv",
        Path.cwd() / "results" / f"{stem}_{args.train_data}_maxsep.csv",
    ]
    
    final_csv = next((c for c in candidates_out if c.exists() and c.stat().st_size > 0), None)

    if final_csv:
        dest = Path(args.out_csv) if args.out_csv else Path.cwd() / f"{stem}_maxsep.csv"
        if dest.resolve() != final_csv.resolve():
            shutil.copy2(final_csv, dest)
        print(f"\n🎉 CLEAN inference úspěšně dokončena!")
        print(f"💾 Výsledný soubor predikcí: {dest.resolve()}")
        print("\nNyní můžete spustit vyhodnocení benchmarku:")
        print(f"   python clean_nise_benchmark.py --clean-csv {dest.name}")
    else:
        print(f"\n⚠️ Inference doběhla, ale výstupní soubor nebyl nalezen mezi kandidáty:")
        for c in candidates_out[:4]:
            print(f"   - {c}")
        print("Zkontrolujte složku 'results/' nebo výstupy v adresáři CLEANu.")

if __name__ == '__main__':
    main()
