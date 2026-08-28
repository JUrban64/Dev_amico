import os
import glob
import json
import argparse
import tempfile
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_pdb_paths(project_root):
    pdb_roots = [
        os.path.join(project_root, 'data_prep', 'structures'), 
        os.path.join(project_root, 'data_prep', 'Binding_Sites'),
        os.path.join(project_root, 'structures'),
        os.path.join(project_root, 'Binding_Sites'),
        os.path.join(project_root, '..', 'EquiPocket-MIL-', 'structures'),
        os.path.join(project_root, '..', 'EquiPocket-MIL-', 'Binding_Sites')
    ]
    pdb_files = {}
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                if '_pocket' not in p and 'prank_output' not in p:
                    base_id = os.path.basename(p).replace('.pdb', '')
                    if base_id not in pdb_files:
                        pdb_files[base_id] = p
    return pdb_files

def main():
    parser = argparse.ArgumentParser(description="Analyze homology within clusters and across splits.")
    parser.add_argument("--suffix", default="_mil_0.5", help="Suffix of split files and clusters (e.g., _mil_0.5)")
    parser.add_argument("--project_dir", default=".", help="Root directory of the AMICO project")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for Foldseek")
    args = parser.parse_args()

    data_prep_dir = os.path.join(args.project_dir, 'data_prep')
    stats_out_dir = os.path.join(args.project_dir, 'data_statistics')
    os.makedirs(stats_out_dir, exist_ok=True)
    
    # Load splits
    splits = {}
    protein_to_split = {}
    for sp in ["train", "validation", "test"]:
        file_path = os.path.join(data_prep_dir, f"{sp}{args.suffix}.txt")
        splits[sp] = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        splits[sp].append(pid)
                        protein_to_split[pid] = sp
        else:
            print(f"Warning: {file_path} not found.")

    if not protein_to_split:
        print("Error: No split data found.")
        return

    # Load clusters
    cluster_file = os.path.join(data_prep_dir, f"clusters{args.suffix}.json")
    protein_to_cluster = {}
    if os.path.exists(cluster_file):
        with open(cluster_file, "r") as f:
            clusters = json.load(f)
            for i, (rep, members) in enumerate(clusters.items()):
                # ensure rep is also in the cluster
                protein_to_cluster[rep] = i
                for m in members:
                    protein_to_cluster[m] = i
    else:
        print(f"Error: {cluster_file} not found. Please run structure_clustering.py first to generate it.")
        return

    pdb_files = get_pdb_paths(args.project_dir)
    print(f"Found {len(pdb_files)} PDB files in project.")

    proteins_to_use = list(protein_to_split.keys())
    
    with tempfile.TemporaryDirectory(prefix="homology_fs_") as tmp_dir:
        pdbs_dir = os.path.join(tmp_dir, "pdbs")
        os.makedirs(pdbs_dir)
        
        missing = 0
        for pid in proteins_to_use:
            if pid in pdb_files:
                dst = os.path.join(pdbs_dir, f"{pid}.pdb")
                # Using copy instead of symlink to avoid issues with some environments
                import shutil
                shutil.copy2(pdb_files[pid], dst)
            else:
                missing += 1
        
        if missing > 0:
            print(f"Warning: {missing} PDBs from splits not found on disk.")
            
        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        fs_tmp = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp, exist_ok=True)
        
        cmd = [
            "foldseek", "easy-search",
            pdbs_dir, pdbs_dir, out_tsv, fs_tmp,
            "--format-output", "query,target,fident,alnlen,qtmscore",
            "-e", "10.0",
            "--threads", str(args.threads)
        ]
        
        print("Running Foldseek easy-search (all-vs-all) on split proteins...")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("Error: foldseek executable not found in PATH.")
            return
        
        if not os.path.exists(out_tsv):
            print("Error: foldseek did not generate output.")
            return
            
        print("Loading results and analyzing...")
        df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "fident", "alnlen", "qtmscore"])
        
        # Strip .pdb
        df['query'] = df['query'].str.replace('.pdb', '', regex=False)
        df['target'] = df['target'].str.replace('.pdb', '', regex=False)
        
        # Remove self-hits
        df = df[df['query'] != df['target']]
        
        # Map back to clusters and splits
        def classify_pair(row):
            q, t = row['query'], row['target']
            q_split = protein_to_split.get(q)
            t_split = protein_to_split.get(t)
            q_clust = protein_to_cluster.get(q)
            t_clust = protein_to_cluster.get(t)
            
            if q_split is None or t_split is None:
                return "Unknown"
            
            if q_clust is not None and t_clust is not None and q_clust == t_clust:
                return "Same Cluster"
            elif q_split == t_split:
                return f"Same Split ({q_split})"
            else:
                splits = sorted([q_split, t_split])
                return f"Cross Split ({splits[0]}-{splits[1]})"
                
        df['category'] = df.apply(classify_pair, axis=1)
        
        # Filter Unknown
        df = df[df['category'] != "Unknown"]
        
        # Group stats
        stats = df.groupby('category').agg(
            pairs=('query', 'count'),
            fident_mean=('fident', 'mean'),
            fident_max=('fident', 'max'),
            qtmscore_mean=('qtmscore', 'mean'),
            qtmscore_max=('qtmscore', 'max')
        ).reset_index()
        
        print("\n--- Homology Analysis Results ---")
        print(stats.to_string(index=False))
        
        # Save to CSV
        csv_out = os.path.join(stats_out_dir, f"homology_stats{args.suffix}.csv")
        stats.to_csv(csv_out, index=False)
        print(f"\nStatistics saved to {csv_out}")
        
        # Generate plot
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='category', y='qtmscore')
        plt.xticks(rotation=45, ha='right')
        plt.title('TM-Score Distribution')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='category', y='fident')
        plt.xticks(rotation=45, ha='right')
        plt.title('Sequence Identity Distribution')
        
        plt.tight_layout()
        plot_out = os.path.join(stats_out_dir, f"homology_plot{args.suffix}.png")
        plt.savefig(plot_out)
        print(f"Plot saved to {plot_out}")

if __name__ == "__main__":
    main()
