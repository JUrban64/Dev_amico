import os
import argparse
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Cluster proteins using ESM embeddings (Two-level: Full protein + Pockets).")
    parser.add_argument("--full-embeddings-path", default="data_prep/esm_full_proteins.pt", help="Path to esm_full_proteins.pt")
    parser.add_argument("--pocket-dataset-path", default="data_prep/esm_dataset.pt", help="Path to esm_dataset.pt (contains pocket residues)")
    parser.add_argument("--global-distance-threshold", type=float, default=0.2, help="Threshold for full protein cosine distance")
    parser.add_argument("--pocket-distance-threshold", type=float, default=0.15, help="Threshold for pocket cosine distance")
    parser.add_argument("--use-pockets", action="store_true", help="Enable two-level clustering (including pocket similarity)")
    parser.add_argument("--suffix", default="_esm_0.2", help="Suffix for output files")
    args = parser.parse_args()

    if not os.path.exists(args.full_embeddings_path):
        print(f"Error: Could not find {args.full_embeddings_path}")
        return

    print(f"Loading full protein embeddings from {args.full_embeddings_path}...")
    full_embeddings_dict = torch.load(args.full_embeddings_path, weights_only=False)
    
    pids = sorted(list(full_embeddings_dict.keys()))
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    
    full_embs = [full_embeddings_dict[pid].numpy().flatten() for pid in pids]
    full_embs = np.array(full_embs)
    
    print(f"Loaded {len(pids)} full embeddings.")
    
    print("Computing global cosine distance matrix...")
    dist_matrix = cosine_distances(full_embs)
    
    # Adjacency matrix for proteins
    adj = (dist_matrix < args.global_distance_threshold).astype(int)
    
    if args.use_pockets:
        if not os.path.exists(args.pocket_dataset_path):
            print(f"Warning: Pocket dataset {args.pocket_dataset_path} not found. Proceeding with full proteins only.")
        else:
            print(f"Loading pocket embeddings from {args.pocket_dataset_path}...")
            raw_data = torch.load(args.pocket_dataset_path, weights_only=False)
            
            pocket_embs = []
            pocket_to_pid_idx = []
            
            for item in tqdm(raw_data, desc="Processing pockets"):
                raw_pid = item['protein_id']
                base_name = os.path.basename(raw_pid)
                pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
                
                if pid not in pid_to_idx:
                    continue # Skip pockets for proteins we don't have full embeddings for
                    
                feat = item['features'] # [N, 1280]
                # Average pooling over residues to get pocket embedding
                pocket_emb = feat.mean(dim=0).numpy().flatten()
                
                pocket_embs.append(pocket_emb)
                pocket_to_pid_idx.append(pid_to_idx[pid])
                
            pocket_embs = np.array(pocket_embs)
            pocket_to_pid_idx = np.array(pocket_to_pid_idx)
            
            print(f"Loaded {len(pocket_embs)} pockets. Computing pocket distance matrix...")
            # This can be large (e.g. 20000x20000), but manageable in RAM (1.6GB)
            # To avoid memory issues for very large datasets, we process in batches
            batch_size = 2000
            n_pockets = len(pocket_embs)
            
            edges_added = 0
            for i in tqdm(range(0, n_pockets, batch_size), desc="Computing pocket similarities"):
                end_i = min(i + batch_size, n_pockets)
                batch_embs = pocket_embs[i:end_i]
                
                # Compute distance against all pockets
                batch_dist = cosine_distances(batch_embs, pocket_embs)
                
                # Find pairs below threshold
                rows, cols = np.where(batch_dist < args.pocket_distance_threshold)
                
                for r, c in zip(rows, cols):
                    global_r = i + r
                    if global_r != c:
                        pid1 = pocket_to_pid_idx[global_r]
                        pid2 = pocket_to_pid_idx[c]
                        if pid1 != pid2 and adj[pid1, pid2] == 0:
                            adj[pid1, pid2] = 1
                            adj[pid2, pid1] = 1
                            edges_added += 1
                            
            print(f"Added {edges_added // 2} new edges between proteins due to pocket similarity.")

    print("Finding connected components (clusters)...")
    graph = csr_matrix(adj)
    n_clusters, cluster_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    print(f"Found {n_clusters} independent clusters.")
    
    clusters = {}
    for i, pid in enumerate(pids):
        c = int(cluster_labels[i])
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(pid)
        
    out_clusters = {}
    for c, members in clusters.items():
        rep = members[0]
        out_clusters[rep] = members
        
    cluster_reps = list(out_clusters.keys())
    
    print("Splitting clusters into train (80%), val (10%), test (10%)...")
    if len(cluster_reps) > 2:
        train_reps, temp_reps = train_test_split(cluster_reps, test_size=0.2, random_state=42)
        if len(temp_reps) > 1:
            val_reps, test_reps = train_test_split(temp_reps, test_size=0.5, random_state=42)
        else:
            val_reps, test_reps = temp_reps, []
    else:
        # Fallback if too few clusters
        train_reps, val_reps, test_reps = cluster_reps, [], []
    
    train_ids = [pid for r in train_reps for pid in out_clusters[r]]
    val_ids = [pid for r in val_reps for pid in out_clusters[r]]
    test_ids = [pid for r in test_reps for pid in out_clusters[r]]
        
    print(f"Train: {len(train_ids)} proteins")
    print(f"Validation: {len(val_ids)} proteins")
    print(f"Test: {len(test_ids)} proteins")
    
    base_dir = os.path.dirname(args.full_embeddings_path)
    if not base_dir: base_dir = "."
    
    train_path = os.path.join(base_dir, f"train{args.suffix}.txt")
    val_path = os.path.join(base_dir, f"validation{args.suffix}.txt")
    test_path = os.path.join(base_dir, f"test{args.suffix}.txt")
    clusters_path = os.path.join(base_dir, f"clusters{args.suffix}.json")
    
    with open(train_path, "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(test_path, "w") as f:
        f.write("\n".join(test_ids) + "\n")
    with open(clusters_path, "w") as f:
        json.dump(out_clusters, f, indent=4)
        
    print(f"Files saved to {base_dir} with suffix {args.suffix}")

if __name__ == "__main__":
    main()
