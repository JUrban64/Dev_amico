import os
import sys
import glob
import time
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Nastavení cest
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset import load_split_ids, load_data_from_tensors
from dataset_cross_mil import load_cross_mil_data, CrossMilDataset, custom_collate_fn
from model import AttentionMIL_ESM
from model_self_attention_mil import SelfAttentionMIL
from model_cross_attention_mil import CrossAttentionMIL
from model_ligand_cross_attention_mil import LigandCrossAttentionMIL

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def normalize_id(pid):
    if not pid:
        return ""
    p = str(pid).strip()
    p = os.path.basename(p)
    p = p.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '').replace('_predictions', '')
    p = p.replace('_MERGED', '').replace('_merged', '')
    return p

def match_id(pid, id_set):
    if not id_set:
        return False
    if pid in id_set:
        return True
    norm_p = normalize_id(pid)
    norm_p_lower = norm_p.lower()
    if norm_p in id_set or norm_p_lower in id_set:
        return True
    for x in id_set:
        norm_x = normalize_id(x)
        if norm_x == norm_p or norm_x.lower() == norm_p_lower:
            return True
        base_p = norm_p.split('_')[0]
        base_x = norm_x.split('_')[0]
        if base_p and base_p.lower() == base_x.lower():
            return True
    return False

def find_dataset_file(filename, data_dir=None):
    """Dynamicky vyhledá datový soubor v zadané složce, v data_prep/ i v kořenu projektu."""
    candidates = []
    if data_dir:
        candidates.extend([
            os.path.join(data_dir, filename),
            os.path.join(data_dir, 'data_prep', filename),
        ])
    candidates.extend([
        os.path.join(PROJECT_ROOT, 'data_prep', filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(PROJECT_ROOT, '..', filename),
        os.path.join(PROJECT_ROOT, '..', 'data_prep', filename),
    ])
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# ============================================================================
# 1. FOLDSEEK 1-NN BENCHMARK
# ============================================================================
def run_foldseek_benchmark(split_suffix, use_nr=False, all_pdbs=False):
    print(f"\n---> Spouštím Foldseek 1-NN Benchmark (split: {split_suffix})...")
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "benchmarks", "foldseek_benchmark.py"),
        "--split-suffix", split_suffix
    ]
    if use_nr:
        cmd.append("--use-nr")
    if all_pdbs:
        cmd.append("--all-pdbs")
        
    start_t = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout
        elapsed = time.time() - start_t
        
        # Parsování výstupu
        acc, f1_macro, f1_weighted = 0.0, 0.0, 0.0
        per_class_f1 = {name: 0.0 for name in TARGET_NAMES}
        
        for line in out.split('\n'):
            line_str = line.strip()
            if line_str.startswith("Accuracy:"):
                acc = float(line_str.split(":")[1].strip())
            elif line_str.startswith("Macro F1:"):
                f1_macro = float(line_str.split(":")[1].strip())
            for name in TARGET_NAMES:
                if line_str.startswith(name):
                    parts = line_str.split()
                    if len(parts) >= 4:
                        try:
                            per_class_f1[name] = float(parts[3])
                        except ValueError:
                            pass
                            
        return {
            "val_acc": None,
            "val_macro_f1": None,
            "test_acc": acc,
            "test_macro_f1": f1_macro,
            "per_class_f1": per_class_f1,
            "time_sec": round(elapsed, 1),
            "status": "SUCCESS"
        }
    except Exception as e:
        print(f"Chyba při běhu Foldseeku: {e}")
        return {"status": f"FAILED: {e}"}

# ============================================================================
# 2. STANDARD MIL (AttentionMIL_ESM)
# ============================================================================
def run_standard_mil(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-4, dropout=0.25, label_smoothing=0.1, batch_size=32, data_dir=None):
    print(f"\n---> Trénuji Standard Attention MIL (split: {split_suffix})...")
    data_path = find_dataset_file('esm_dataset.pt', data_dir)
    if not data_path:
        return {"status": "FAILED: esm_dataset.pt nenalezen"}

    bags = load_data_from_tensors(data_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    if len(train_bags) == 0:
        return {"status": f"FAILED: prázdný train set (načteno {len(train_ids)} train ID, v datasetu {len(bags)} proteinů)"}

    from torch.nn.utils.rnn import pad_sequence
    def collate_fn_mil(batch):
        features_list = [item['features'] for item in batch]
        labels = torch.cat([item['label'] for item in batch])
        padded_features = pad_sequence(features_list, batch_first=True)
        lengths = torch.tensor([f.size(0) for f in features_list])
        max_len = padded_features.size(1)
        padding_mask = torch.arange(max_len).expand(len(features_list), max_len) >= lengths.unsqueeze(1)
        return padded_features, padding_mask, labels

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_mil)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mil)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mil) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = AttentionMIL_ESM(
        in_features=1280,
        hidden_dim=256,
        num_classes=5,
        dropout=dropout,
        gated_attention=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for feat, mask, lbl in loader:
                feat, mask, lbl = feat.to(device), mask.to(device), lbl.to(device)
                logits, _ = model(feat, mask)
                loss = criterion(logits, lbl)
                loss_sum += loss.item() * len(lbl)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(lbl.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for feat, mask, lbl in train_loader:
            feat, mask, lbl = feat.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits, _ = model(feat, mask)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 2B. PURE SEQUENCE ESM-2 MLP (Bez kapes, bez ligandů)
# ============================================================================
def run_sequence_mlp(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-3, dropout=0.3, label_smoothing=0.1, batch_size=64, data_dir=None):
    print(f"\n---> Trénuji Pure Sequence ESM-2 MLP (split: {split_suffix})...")
    from dataset_cross_mil import load_cross_mil_data
    from model_sequence_mlp import SequenceMLPClassifier

    pockets_path = find_dataset_file('esm_dataset.pt', data_dir)
    full_proteins_path = find_dataset_file('esm_full_proteins.pt', data_dir)

    if not pockets_path or not full_proteins_path:
        missing = []
        if not pockets_path: missing.append('esm_dataset.pt')
        if not full_proteins_path: missing.append('esm_full_proteins.pt')
        return {"status": f"FAILED: chybí {', '.join(missing)}"}

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    if len(train_bags) == 0:
        return {"status": f"FAILED: prázdný train set (načteno {len(train_ids)} train ID, v datasetu {len(all_bags)} proteinů)"}

    def collate_seq(batch):
        feats = torch.stack([item['full_protein_feature'] for item in batch])
        labels = torch.cat([item['label'] for item in batch])
        return feats, labels

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=collate_seq)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_seq)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_seq) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = SequenceMLPClassifier(in_features=1280, hidden_dim=256, num_classes=5, dropout=dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for feats, lbl in loader:
                feats, lbl = feats.to(device), lbl.to(device)
                logits = model(feats)
                loss = criterion(logits, lbl)
                loss_sum += loss.item() * len(lbl)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(lbl.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for feats, lbl in train_loader:
            feats, lbl = feats.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 2C. PER-RESIDUE ATTENTION MIL (Všechna rezidua bez průměrování do kapes)
# ============================================================================
def run_residue_mil(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-4, dropout=0.25, label_smoothing=0.1, batch_size=32, data_dir=None):
    print(f"\n---> Trénuji Per-Residue Attention MIL (split: {split_suffix})...")
    data_path = find_dataset_file('esm_dataset.pt', data_dir)
    if not data_path:
        return {"status": "FAILED: esm_dataset.pt nenalezen"}

    bags = load_data_from_tensors(data_path, mode='residues')
    train_ids, val_ids, test_ids = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    if len(train_bags) == 0:
        return {"status": f"FAILED: prázdný train set (načteno {len(train_ids)} train ID, v datasetu {len(bags)} proteinů)"}

    from torch.nn.utils.rnn import pad_sequence
    def collate_fn_mil(batch):
        features_list = [item['features'] for item in batch]
        labels = torch.cat([item['label'] for item in batch])
        padded_features = pad_sequence(features_list, batch_first=True)
        lengths = torch.tensor([f.size(0) for f in features_list])
        max_len = padded_features.size(1)
        padding_mask = torch.arange(max_len).expand(len(features_list), max_len) >= lengths.unsqueeze(1)
        return padded_features, padding_mask, labels

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_mil)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mil)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mil) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = AttentionMIL_ESM(
        in_features=1280,
        hidden_dim=256,
        num_classes=5,
        dropout=dropout,
        gated_attention=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for feat, mask, lbl in loader:
                feat, mask, lbl = feat.to(device), mask.to(device), lbl.to(device)
                logits, _ = model(feat, mask)
                loss = criterion(logits, lbl)
                loss_sum += loss.item() * len(lbl)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(lbl.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for feat, mask, lbl in train_loader:
            feat, mask, lbl = feat.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits, _ = model(feat, mask)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 3. GENERIC POCKET + PROTEIN CROSS-ATTENTION / SELF-ATTENTION RUNNER
# ============================================================================
def run_cross_mil_variant(model_name, split_suffix, device, epochs=40, lr=4.8e-5, weight_decay=1e-4, dropout=0.2, label_smoothing=0.15, batch_size=32, data_dir=None):
    print(f"\n---> Trénuji {model_name} (split: {split_suffix})...")
    pockets_path = find_dataset_file('esm_dataset.pt', data_dir)
    full_proteins_path = find_dataset_file('esm_full_proteins.pt', data_dir)

    if not pockets_path or not full_proteins_path:
        missing = []
        if not pockets_path: missing.append('esm_dataset.pt')
        if not full_proteins_path: missing.append('esm_full_proteins.pt')
        return {"status": f"FAILED: chybí {', '.join(missing)}"}

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    if len(train_bags) == 0:
        return {"status": f"FAILED: prázdný train set (načteno {len(train_ids)} train ID, v datasetu {len(all_bags)} proteinů)"}

    train_loader = DataLoader(CrossMilDataset(train_bags), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(CrossMilDataset(val_bags), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(CrossMilDataset(test_bags), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Volba architektury
    if model_name == "self_attention_mil":
        model = SelfAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=dropout).to(device)
    elif model_name == "cross_attention_mil":
        model = CrossAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=dropout).to(device)
    elif model_name == "ligand_cross_mil":
        model = LigandCrossAttentionMIL(feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=dropout).to(device)
    else:
        return {"status": f"FAILED: neznámý model {model_name}"}

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for pocket_feats, mask, full_prot_feats, labels in loader:
                pocket_feats, mask, full_prot_feats = pocket_feats.to(device), mask.to(device), full_prot_feats.to(device)
                labels = labels.to(device).squeeze()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                logits, _ = model(pocket_feats, mask, full_prot_feats)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=-1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for pocket_feats, mask, full_prot_feats, labels in train_loader:
            pocket_feats, mask, full_prot_feats = pocket_feats.to(device), mask.to(device), full_prot_feats.to(device)
            labels = labels.to(device).squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            optimizer.zero_grad()
            logits, _ = model(pocket_feats, mask, full_prot_feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 4. EGNN SCORE-LEVEL MIL & ENCODER MIL RUNNERS
# ============================================================================
def run_egnn_mil(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-3, dropout=0.3, label_smoothing=0.1, batch_size=16, data_dir=None):
    print(f"\n---> Trénuji EGNN Score-Level MIL (split: {split_suffix})...")
    from dataset_egnn import get_egnn_splits
    from model_egnn_mil import EGNN_MIL_Classifier
    from torch_geometric.data import Batch

    data_path = find_dataset_file('egnn_dataset.pt', data_dir)
    if not data_path:
        return {"status": "FAILED: egnn_dataset.pt nenalezen"}

    train_bags, val_bags, test_bags = get_egnn_splits(data_path, PROJECT_ROOT, split_suffix=split_suffix)
    if len(train_bags) == 0:
        return {"status": "FAILED: prázdný train set"}

    def egnn_collate(batch):
        labels = torch.cat([item['label'].view(1) for item in batch])
        all_graphs = []
        protein_indices = []
        for i, item in enumerate(batch):
            graphs = item['batch_graph'].to_data_list()
            all_graphs.extend(graphs)
            protein_indices.extend([i] * len(graphs))
        mega_batch = Batch.from_data_list(all_graphs)
        protein_idx = torch.tensor(protein_indices, dtype=torch.long)
        return mega_batch, protein_idx, labels

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=egnn_collate)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_collate)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_collate) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = EGNN_MIL_Classifier(
        node_dim=1280,
        hidden_dim=128,
        num_gnn_layers=2,
        num_classes=5,
        dropout=dropout,
        gated_attention=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for mega_batch, protein_idx, labels in loader:
                mega_batch, protein_idx, labels = mega_batch.to(device), protein_idx.to(device), labels.to(device)
                logits, _ = model(mega_batch, protein_idx=protein_idx)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for mega_batch, protein_idx, labels in train_loader:
            mega_batch, protein_idx, labels = mega_batch.to(device), protein_idx.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx=protein_idx)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

def run_encoder_mil(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-3, dropout=0.3, label_smoothing=0.1, batch_size=16, data_dir=None):
    print(f"\n---> Trénuji EGNN Embedding-Level Encoder MIL (split: {split_suffix})...")
    from dataset_egnn import get_egnn_splits
    from model_encoder_mil import EGNN_Encoder_MIL_Classifier
    from torch_geometric.data import Batch

    data_path = find_dataset_file('egnn_dataset.pt', data_dir)
    if not data_path:
        return {"status": "FAILED: egnn_dataset.pt nenalezen"}

    train_bags, val_bags, test_bags = get_egnn_splits(data_path, PROJECT_ROOT, split_suffix=split_suffix)
    if len(train_bags) == 0:
        return {"status": "FAILED: prázdný train set"}

    def egnn_collate(batch):
        labels = torch.cat([item['label'].view(1) for item in batch])
        all_graphs = []
        protein_indices = []
        for i, item in enumerate(batch):
            graphs = item['batch_graph'].to_data_list()
            all_graphs.extend(graphs)
            protein_indices.extend([i] * len(graphs))
        mega_batch = Batch.from_data_list(all_graphs)
        protein_idx = torch.tensor(protein_indices, dtype=torch.long)
        return mega_batch, protein_idx, labels

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=egnn_collate)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_collate)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_collate) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = EGNN_Encoder_MIL_Classifier(
        node_dim=1280,
        egnn_hidden_dim=128,
        mil_hidden_dim=128,
        num_gnn_layers=2,
        num_classes=5,
        dropout=dropout,
        num_heads=2,
        gated_attention=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for mega_batch, protein_idx, labels in loader:
                mega_batch, protein_idx, labels = mega_batch.to(device), protein_idx.to(device), labels.to(device)
                logits, _ = model(mega_batch, protein_idx=protein_idx)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for mega_batch, protein_idx, labels in train_loader:
            mega_batch, protein_idx, labels = mega_batch.to(device), protein_idx.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx=protein_idx)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 5. EGNN HYBRID RUNNER (3D Graph + Ligand Cross-Attention)
# ============================================================================
def run_egnn_ligand_cross_mil(split_suffix, device, epochs=40, lr=5e-5, weight_decay=1e-3, dropout=0.35, label_smoothing=0.1, batch_size=16, data_dir=None):
    print(f"\n---> Trénuji EGNN + Ligand Cross-Attention MIL (split: {split_suffix})...")
    from dataset_egnn_cross_mil import get_egnn_cross_splits, egnn_cross_collate_fn
    from model_egnn_ligand_cross_attention_mil import EGNN_Ligand_Cross_Attention_MIL

    data_path = find_dataset_file('egnn_dataset.pt', data_dir)
    full_proteins_path = find_dataset_file('esm_full_proteins.pt', data_dir)

    if not data_path or not full_proteins_path:
        missing = []
        if not data_path: missing.append('egnn_dataset.pt')
        if not full_proteins_path: missing.append('esm_full_proteins.pt')
        return {"status": f"FAILED: chybí {', '.join(missing)}"}

    train_bags, val_bags, test_bags = get_egnn_cross_splits(
        data_path=data_path,
        full_proteins_path=full_proteins_path,
        base_dir=PROJECT_ROOT,
        split_suffix=split_suffix
    )

    if len(train_bags) == 0:
        return {"status": "FAILED: prázdný train set"}

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=egnn_cross_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = EGNN_Ligand_Cross_Attention_MIL(
        node_dim=1280,
        full_protein_dim=1280,
        ecfp_dim=1024,
        hidden_dim=128,
        num_gnn_layers=2,
        num_heads=4,
        num_classes=5,
        dropout=dropout,
        pocket_drop_prob=0.15
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for mega_batch, protein_idx, full_prot_feats, labels in loader:
                mega_batch = mega_batch.to(device)
                protein_idx = protein_idx.to(device)
                full_prot_feats = full_prot_feats.to(device)
                labels = labels.to(device)
                logits, _ = model(mega_batch, protein_idx, full_prot_feats)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for mega_batch, protein_idx, full_prot_feats, labels in train_loader:
            mega_batch = mega_batch.to(device)
            protein_idx = protein_idx.to(device)
            full_prot_feats = full_prot_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx, full_prot_feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 5b. EGNN SELF-ATTENTION MIL RUNNER (3D Graph + Sequence Self-Attention)
# ============================================================================
def run_egnn_self_attention_mil(split_suffix, device, epochs=40, lr=1e-4, weight_decay=1e-3, dropout=0.35, label_smoothing=0.1, batch_size=16, data_dir=None):
    print(f"\n---> Trénuji EGNN + Self-Attention MIL (split: {split_suffix})...")
    from dataset_egnn_cross_mil import get_egnn_cross_splits, egnn_cross_collate_fn
    from model_egnn_self_attention_mil import EGNN_Self_Attention_MIL

    data_path = find_dataset_file('egnn_dataset.pt', data_dir)
    full_proteins_path = find_dataset_file('esm_full_proteins.pt', data_dir)

    if not data_path or not full_proteins_path:
        missing = []
        if not data_path: missing.append('egnn_dataset.pt')
        if not full_proteins_path: missing.append('esm_full_proteins.pt')
        return {"status": f"FAILED: chybí {', '.join(missing)}"}

    train_bags, val_bags, test_bags = get_egnn_cross_splits(
        data_path=data_path,
        full_proteins_path=full_proteins_path,
        base_dir=PROJECT_ROOT,
        split_suffix=split_suffix
    )

    if len(train_bags) == 0:
        return {"status": "FAILED: prázdný train set"}

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=egnn_cross_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn)
    test_loader = DataLoader(test_bags, batch_size=batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = EGNN_Self_Attention_MIL(
        node_dim=1280,
        full_protein_dim=1280,
        hidden_dim=256,
        num_gnn_layers=2,
        num_heads=4,
        num_attn_layers=1,
        num_classes=5,
        dropout=dropout,
        pocket_drop_prob=0.15
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=12)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for mega_batch, protein_idx, full_prot_feats, labels in loader:
                mega_batch = mega_batch.to(device)
                protein_idx = protein_idx.to(device)
                full_prot_feats = full_prot_feats.to(device)
                labels = labels.to(device)
                logits, _ = model(mega_batch, protein_idx, full_prot_feats)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    start_t = time.time()
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for mega_batch, protein_idx, full_prot_feats, labels in train_loader:
            mega_batch = mega_batch.to(device)
            protein_idx = protein_idx.to(device)
            full_prot_feats = full_prot_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx, full_prot_feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_t
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, final_val_acc, final_val_f1, _ = evaluate(val_loader)
    test_acc, test_f1, per_class_f1 = 0.0, 0.0, {name: 0.0 for name in TARGET_NAMES}
    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

    return {
        "val_acc": final_val_acc,
        "val_macro_f1": final_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }

# ============================================================================
# 6. DISCOVERY SPLITŮ A HLAVNÍ ORCHESTRÁTOR
# ============================================================================
def discover_splits():
    """Najde všechny dostupné split soubory ve složkách data_prep a root."""
    splits = set()
    search_dirs = [os.path.join(PROJECT_ROOT, 'data_prep'), PROJECT_ROOT]
    for d in search_dirs:
        if os.path.exists(d):
            for p in glob.glob(os.path.join(d, 'train_*.txt')):
                fname = os.path.basename(p)
                sfx = fname.replace('train_', '').replace('.txt', '')
                if sfx.startswith('_'):
                    sfx = sfx[1:]
                if not sfx.endswith('_nr0.95'): # Základní splity
                    splits.add(sfx)
    return sorted(list(splits))

def main():
    parser = argparse.ArgumentParser(description="Master Benchmark všech AMICO modelů napříč všemi splity.")
    parser.add_argument('--models', nargs='+', default=['all'], 
                        help="Modely k evaluaci: foldseek, standard_mil, self_attention_mil, cross_attention_mil, ligand_cross_mil, egnn_mil, encoder_mil, egnn_ligand_cross_mil, egnn_self_attention_mil nebo all")
    parser.add_argument('--splits', nargs='+', default=['all'], 
                        help="Splity k evaluaci: mil_0.3, mil_0.5, mil_0.7, struct_pocket_0.5_0.5 nebo all")
    parser.add_argument('--epochs', type=int, default=40, help="Maximální počet epoch pro trénované modely")
    parser.add_argument('--force', action='store_true', help="Znovu spustit i již hotové evaluace")
    parser.add_argument('--data-dir', default=None, help="Cesta ke složce obsahující předpočítané tenzory (esm_dataset.pt, esm_full_proteins.pt, egnn_dataset.pt)")
    parser.add_argument('--out-prefix', default='benchmark_results', help="Prefix pro výstupní JSON, CSV a MD soubory")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print("\n" + "="*75)
    print("      AMICO MASTER BENCHMARK - SROVNÁNÍ VŠECH MODELŮ      ")
    print("="*75)
    print(f"Zařízení: {device}")
    if args.data_dir:
        print(f"Vlastní datová složka (--data-dir): {args.data_dir}")

    # 1. Zjištění splitů
    all_available_splits = discover_splits()
    if 'all' in args.splits:
        mil_splits = [s for s in all_available_splits if s.startswith('mil_')]
        target_splits = mil_splits if mil_splits else all_available_splits
        if not target_splits:
            target_splits = ['mil_0.3', 'mil_0.5', 'mil_0.7', 'mil_0.9']
    else:
        target_splits = []
        for s in args.splits:
            s_clean = s.strip()
            if s_clean.startswith('0.') or s_clean in ['0.3', '0.5', '0.7', '0.9']:
                target_splits.append(f"mil_{s_clean}")
            else:
                target_splits.append(s_clean)

    print(f"Vybrané splity ({len(target_splits)}): {target_splits}")

    # 2. Zjištění modelů
    all_model_keys = [
        "foldseek", 
        "sequence_mlp",
        "residue_mil",
        "standard_mil", 
        "self_attention_mil", 
        "cross_attention_mil", 
        "encoder_mil",
        "ligand_cross_mil",
        "egnn_mil",
        "egnn_ligand_cross_mil",
        "egnn_self_attention_mil"
    ]

    alias_map = {
        "egnn_self": "egnn_self_attention_mil",
        "egnn_self_attention": "egnn_self_attention_mil",
        "self": "self_attention_mil",
        "self_attention": "self_attention_mil",
        "cross": "cross_attention_mil",
        "cross_attention": "cross_attention_mil",
        "ligand_cross": "ligand_cross_mil",
        "ligand_cross_attention": "ligand_cross_mil",
        "egnn_ligand_cross": "egnn_ligand_cross_mil",
        "egnn_ligand_cross_attention": "egnn_ligand_cross_mil",
        "standard": "standard_mil",
        "sequence_mlp": "sequence_mlp",
        "seq_mlp": "sequence_mlp",
        "mlp": "sequence_mlp",
        "residue": "residue_mil",
        "encoder": "encoder_mil",
        "egnn": "egnn_mil",
    }

    def resolve_model(m_str):
        clean = m_str.strip().lower()
        if clean.endswith('.py'):
            clean = clean[:-3]
        if clean.startswith('model_'):
            clean = clean[6:]
        if clean in all_model_keys:
            return clean
        if clean in alias_map:
            return alias_map[clean]
        if f"{clean}_mil" in all_model_keys:
            return f"{clean}_mil"
        return None

    if 'all' in args.models:
        target_models = all_model_keys
    else:
        target_models = []
        for m in args.models:
            resolved = resolve_model(m)
            if resolved:
                if resolved not in target_models:
                    target_models.append(resolved)
            else:
                print(f"⚠️ Upozornění: Model '{m}' nebyl rozpoznán a bude přeskočen! Dostupné modely: {all_model_keys}")

    print(f"Vybrané modely ({len(target_models)}): {target_models}\n")

    # 3. Načtení existujících výsledků (pro možnost obnovení)
    json_path = f"{args.out_prefix}.json"
    csv_path = f"{args.out_prefix}.csv"
    md_path = f"{args.out_prefix}.md"

    results_data = []
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path, 'r') as f:
                results_data = json.load(f)
            print(f"Načteno {len(results_data)} existujících záznamů z {json_path}")
        except Exception:
            results_data = []

    def is_already_done(model_name, split_sfx):
        for entry in results_data:
            if entry.get("model") == model_name and entry.get("split") == split_sfx and entry.get("status") == "SUCCESS":
                return True
        return False

    # 4. Spouštění benchmarku pro každou kombinaci
    total_runs = len(target_splits) * len(target_models)
    run_idx = 0

    for sfx in target_splits:
        for model_name in target_models:
            run_idx += 1
            print("\n" + "-"*75)
            print(f"[{run_idx}/{total_runs}] Model: {model_name} | Split: {sfx}")
            print("-" * 75)

            if is_already_done(model_name, sfx) and not args.force:
                print(f"Přeskakuji (již úspěšně dokončeno). Použijte --force pro přetrénování.")
                continue

            res = {}
            try:
                if model_name == "foldseek":
                    res = run_foldseek_benchmark(sfx)
                elif model_name == "sequence_mlp":
                    res = run_sequence_mlp(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "residue_mil":
                    res = run_residue_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "standard_mil":
                    res = run_standard_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name in ["self_attention_mil", "cross_attention_mil", "ligand_cross_mil"]:
                    res = run_cross_mil_variant(model_name, sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "egnn_mil":
                    res = run_egnn_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "encoder_mil":
                    res = run_encoder_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "egnn_ligand_cross_mil":
                    res = run_egnn_ligand_cross_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
                elif model_name == "egnn_self_attention_mil":
                    res = run_egnn_self_attention_mil(sfx, device=device, epochs=args.epochs, data_dir=args.data_dir)
            except Exception as e:
                import traceback
                print(f"❌ Neočekávaná výjimka při trénování {model_name} na {sfx}:\n{e}")
                traceback.print_exc()
                res = {"status": f"FAILED: {e}"}

            # Přehledný log výsledku
            status = res.get("status", "UNKNOWN")
            if status == "SUCCESS":
                f1_val = res.get('test_macro_f1', 0.0)
                acc_val = res.get('test_acc', 0.0)
                t_sec = res.get('time_sec', 0.0)
                print(f"✅ [{model_name} | {sfx}] DOKONČENO ({t_sec:.1f} s) -> Test F1: {f1_val:.4f} | Test Acc: {acc_val:.4f}")
            else:
                print(f"❌ [{model_name} | {sfx}] SELHALO -> {status}")

            record = {
                "model": model_name,
                "split": sfx,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **res
            }

            # Odstraníme starý záznam pokud existoval a přidáme nový
            results_data = [r for r in results_data if not (r.get("model") == model_name and r.get("split") == sfx)]
            results_data.append(record)

            # Průběžné ukládání po každém běhu
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=4)

            # Průběžné generování CSV
            flat_rows = []
            for r in results_data:
                row = {
                    "Model": r.get("model"),
                    "Split": r.get("split"),
                    "Status": r.get("status", "UNKNOWN"),
                    "Val_Acc": r.get("val_acc"),
                    "Val_Macro_F1": r.get("val_macro_f1"),
                    "Test_Acc": r.get("test_acc"),
                    "Test_Macro_F1": r.get("test_macro_f1"),
                    "Time_s": r.get("time_sec")
                }
                if "per_class_f1" in r and isinstance(r["per_class_f1"], dict):
                    for cname in TARGET_NAMES:
                        row[f"Test_F1_{cname}"] = r["per_class_f1"].get(cname, 0.0)
                flat_rows.append(row)

            if flat_rows:
                df = pd.DataFrame(flat_rows)
                df.to_csv(csv_path, index=False)

    # 5. Generování finálního souhrnného Markdown reportu
    print("\n" + "="*80)
    print("                  FINÁLNÍ BENCHMARK SOUHRN                  ")
    print("="*80)

    flat_rows = []
    for r in results_data:
        row = {
            "Model": r.get("model"),
            "Split": r.get("split"),
            "Status": r.get("status", "UNKNOWN"),
            "Val_Acc": r.get("val_acc"),
            "Val_Macro_F1": r.get("val_macro_f1"),
            "Test_Acc": r.get("test_acc"),
            "Test_Macro_F1": r.get("test_macro_f1"),
            "Time_s": r.get("time_sec")
        }
        if "per_class_f1" in r and isinstance(r["per_class_f1"], dict):
            for cname in TARGET_NAMES:
                row[f"Test_F1_{cname}"] = r["per_class_f1"].get(cname, 0.0)
        flat_rows.append(row)

    if flat_rows:
        df = pd.DataFrame(flat_rows)
        for col in ["Val_Acc", "Val_Macro_F1", "Test_Acc", "Test_Macro_F1"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        if "Time_s" in df.columns:
            df["Time_s"] = df["Time_s"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")

        print(df.to_string(index=False))

        # Uložení Markdown reportu
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# AMICO: Kompletní Srovnání Všech Modelů\n\n")
            f.write(f"Vygenerováno: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("### Celkové výsledky\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # Pivot table (Model x Split -> Test Macro F1) pouze pro úspěšné běhy
            try:
                raw_df = pd.DataFrame(flat_rows)
                success_df = raw_df[raw_df['Status'] == 'SUCCESS'].copy()
                if not success_df.empty:
                    success_df['Test_Macro_F1'] = pd.to_numeric(success_df['Test_Macro_F1'], errors='coerce')
                    pivot = success_df.pivot(index="Model", columns="Split", values="Test_Macro_F1")
                    f.write("### Pivot Tabulka: Test Macro F1 napříč Splity\n\n")
                    f.write(pivot.to_markdown())
                    f.write("\n")
            except Exception:
                pass

        print(f"\nSouhrnné reporty uloženy:")
        print(f" - JSON: {json_path}")
        print(f" - CSV:  {csv_path}")
        print(f" - MD:   {md_path}")
    else:
        print("Žádné výsledky k zobrazení.")

if __name__ == '__main__':
    main()
