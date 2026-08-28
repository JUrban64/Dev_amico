import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from dataset import load_split_ids
from dataset_cross_mil import load_cross_mil_data, CrossMilDataset, custom_collate_fn
from model_ligand_cross_attention_mil import LigandCrossAttentionMIL

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
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

def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Použité zařízení pro trénink: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)
    
    if not os.path.exists(full_proteins_path):
        print(f"Chyba: Soubor s full protein embeddingy nenalezen: {full_proteins_path}")
        print("Nejprve spusťte: python data_prep/generate_full_protein_embeddings.py")
        return
        
    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix, use_nr=args.use_nr)
    
    print("\n--- KONTROLA SPLIT SOUBORŮ ---")
    print(f"Hledaný suffix: '{args.split_suffix}' (use_nr={args.use_nr})")
    print(f"Načteno unikátních ID - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    if len(val_ids) == 0 and len(test_ids) == 0:
        print("\n⚠️ VAROVÁNÍ: Val a Test ID jsou prázdné!")
        print("Dostupné split soubory ve složce data_prep/:")
        available_splits = [os.path.basename(f) for f in os.listdir(os.path.join(base_dir, 'data_prep')) if f.endswith('.txt') and ('train' in f or 'val' in f or 'test' in f)]
        print(" ", available_splits if available_splits else "Žádné .txt splity nenalezeny. Spusťte nejprve structure_clustering.py")
        print("Ujistěte se, že spouštíte trénink se správným argumentem --split-suffix (např. --split-suffix mil_0.5 nebo mil_0.7) a případně --use-nr.\n")
        
    print("-------------------------------\n")
    
    # Normalizace ID pro případ, že v textovém souboru je 'P12345' a v datasetu 'P12345_MERGED'
    def match_id(pid, id_set):
        if pid in id_set:
            return True
        clean_pid = pid.replace('_MERGED', '').replace('.pdb', '')
        if clean_pid in id_set:
            return True
        for x in id_set:
            if x.replace('_MERGED', '').replace('.pdb', '') == clean_pid:
                return True
        return False
    
    train_bags, val_bags, test_bags = [], [], []
    unassigned_count = 0
    
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)
        else:
            unassigned_count += 1
            if len(val_ids) == 0 and len(test_ids) == 0:
                # Pokud neexistuje žádný split, dáme do train jako nouzový režim
                train_bags.append(b)
            
    print(f"Dataset split - Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    if unassigned_count > 0:
        print(f"Informace: {unassigned_count} proteinů z datasetu nebylo obsaženo v zadaném splitu (vynecháno).")
    
    if len(train_bags) == 0:
        print("Chyba: Trénovací dataset je prázdný.")
        return
        
    train_loader = DataLoader(CrossMilDataset(train_bags), batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(CrossMilDataset(val_bags), batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    if len(test_bags) > 0:
        test_loader = DataLoader(CrossMilDataset(test_bags), batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)
    class_weights = total_samples / (5.0 * np.maximum(class_counts, 1))
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Počty vzorků ve třídách (Train): {class_counts}")
    
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    model = LigandCrossAttentionMIL(
        feature_dim=1280,
        ecfp_dim=args.ecfp_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=5,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)
    
    def evaluate_model(loader, desc="Eval"):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for pocket_feats, mask, full_prot_feats, labels in loader:
                pocket_feats = pocket_feats.to(device)
                mask = mask.to(device)
                full_prot_feats = full_prot_feats.to(device)
                labels = labels.to(device).squeeze()
                
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                logits, _ = model(pocket_feats, mask, full_prot_feats)
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_labels) > 0 else 0.0
        
        return avg_loss, acc, f1_macro, f1_weighted, all_labels, all_preds

    print("\n--- Zahájení tréninku Ligand-Cross-Attention (ECFP) MIL ---")
    best_val_loss = float('inf')
    
    log_file = f"train_log_ligand_cross_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "history": [],
        "results": {}
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total_train = 0
        
        for pocket_feats, mask, full_prot_feats, labels in train_loader:
            pocket_feats = pocket_feats.to(device)
            mask = mask.to(device)
            full_prot_feats = full_prot_feats.to(device)
            labels = labels.to(device).squeeze()
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            optimizer.zero_grad()
            logits, _ = model(pocket_feats, mask, full_prot_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            total_train += len(labels)
            
        train_loss /= max(total_train, 1)
        
        val_loss, val_acc, val_f1_m, val_f1_w, _, _ = evaluate_model(val_loader, desc="Val")
        
        if epoch % 5 == 0 or epoch == 1 or early_stopping(val_loss):
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1_m:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ligand_cross_attention_mil_best.pt')
            
        log_data["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_m
        })
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
            
        if early_stopping.early_stop:
            print(f"\nEarly stopping v epoše {epoch}!")
            break
            
    if os.path.exists('ligand_cross_attention_mil_best.pt'):
        model.load_state_dict(torch.load('ligand_cross_attention_mil_best.pt', weights_only=True))
        
    print("\n========================================================")
    print("      FINÁLNÍ VYHODNOCENÍ LIGAND-CROSS-ATTN (ECFP)      ")
    print("========================================================")
    
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    if len(val_bags) > 0:
        val_loss, val_acc, val_f1_m, val_f1_w, val_lbls, val_preds = evaluate_model(val_loader, desc="Val")
        print(f"\n[VALIDACE] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1_m:.4f}")
        print(classification_report(val_lbls, val_preds, target_names=target_names, zero_division=0))
        
    if len(test_bags) > 0:
        test_loss, test_acc, test_f1_m, test_f1_w, test_lbls, test_preds = evaluate_model(test_loader, desc="Test")
        print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1_m:.4f}")
        print(classification_report(test_lbls, test_preds, target_names=target_names, zero_division=0))
        
    log_data["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"\nVýsledky uloženy do {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trénování Ligand-Cross-Attention MIL modelu (ECFP)")
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt')
    parser.add_argument('--split-suffix', default='mil_0.5')
    parser.add_argument('--use-esm-split', action='store_true', help='Použít ESM embedding clustering split (_esm_0.2)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu (např. mil_0.5_nr0.95)')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--ecfp-dim', type=int, default=1024)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing pro CrossEntropyLoss')
    parser.add_argument('--config-json', default=None, help='Cesta k JSON souboru z Optuny s nejlepšími parametry')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    if args.config_json and os.path.exists(args.config_json):
        print(f"\nNačítám konfiguraci z {args.config_json}...")
        with open(args.config_json, "r") as f:
            cfg = json.load(f)
        params = cfg.get("best_params", cfg)
        for k, v in params.items():
            if hasattr(args, k):
                setattr(args, k, v)
                print(f" - Nastaveno {k} = {v}")

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'
    
    train_and_evaluate(args)
