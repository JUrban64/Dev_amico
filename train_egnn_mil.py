import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset_egnn import get_egnn_splits
from model_egnn_mil import EGNN_MIL_Classifier
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def egnn_collate_fn(batch):
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
    print(f"Použité zařízení pro trénink EGNN: {device}")
    
    # 1. Načtení dat a splitů
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_bags, val_bags, test_bags = get_egnn_splits(args.data_path, base_dir, split_suffix=args.split_suffix, use_nr=args.use_nr)
    
    if len(train_bags) == 0:
        print("Chyba: Trénovací dataset je prázdný.")
        return
        
    # 2. Výpočet vah pro CrossEntropy (vyvážení tříd)
    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)
    class_weights = total_samples / (5.0 * np.maximum(class_counts, 1))
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Počty vzorků ve třídách (Train): {class_counts}")
    print(f"Váhy tříd pro CrossEntropy: {class_weights.cpu().numpy().round(3)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 3. Inicilizace EGNN_MIL modelu
    model = EGNN_MIL_Classifier(
        node_dim=1280,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_classes=5,
        dropout=args.dropout,
        gated_attention=args.gated_attention
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)
    
    def evaluate_model(bags_list, desc="Eval"):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        loader = DataLoader(bags_list, batch_size=args.batch_size, shuffle=False, collate_fn=egnn_collate_fn)
        
        with torch.no_grad():
            for mega_batch, protein_idx, labels in loader:
                mega_batch = mega_batch.to(device)
                protein_idx = protein_idx.to(device)
                labels = labels.to(device)
                
                logits, _ = model(mega_batch, protein_idx=protein_idx)
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
        avg_loss = total_loss / max(len(bags_list), 1)
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_labels) > 0 else 0.0
        
        return avg_loss, acc, f1_macro, f1_weighted, all_labels, all_preds

    print("\n--- Zahájení tréninku EGNN-MIL ---")
    best_val_loss = float('inf')
    
    script_name = os.path.basename(__file__).replace(".py", "")
    log_file = f"train_log_{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "history": [],
        "results": {}
    }
    
    train_loader = DataLoader(train_bags, batch_size=args.batch_size, shuffle=True, collate_fn=egnn_collate_fn)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for mega_batch, protein_idx, labels in train_loader:
            mega_batch = mega_batch.to(device)
            protein_idx = protein_idx.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx=protein_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            
        train_loss /= max(len(train_bags), 1)
        val_loss, val_acc, val_f1_m, val_f1_w, _, _ = evaluate_model(val_bags, desc="Val")
        
        if epoch % 5 == 0 or epoch == 1 or early_stopping(val_loss):
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1_m:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'egnn_mil_best_model.pt')
            
        # Průběžné uložení logu
        log_data["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_m,
            "best_val_loss": best_val_loss
        })
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
            
        if early_stopping.early_stop:
            print(f"\nEarly stopping v epoše {epoch}!")
            break
            
    # Načtení nejlepšího modelu pro finální vyhodnocení
    if os.path.exists('egnn_mil_best_model.pt'):
        model.load_state_dict(torch.load('egnn_mil_best_model.pt', weights_only=True))
        
    print("\n==========================================")
    print("         FINÁLNÍ VYHODNOCENÍ EGNN-MIL      ")
    print("==========================================")
    
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    if len(val_bags) > 0:
        val_loss, val_acc, val_f1_m, val_f1_w, val_lbls, val_preds = evaluate_model(val_bags, desc="Val")
        print(f"\n[VALIDACE] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1_m:.4f} | Weighted F1: {val_f1_w:.4f}")
        print("\nDetailní report (Validace):")
        print(classification_report(val_lbls, val_preds, target_names=target_names, zero_division=0))
        
    if len(test_bags) > 0:
        test_loss, test_acc, test_f1_m, test_f1_w, test_lbls, test_preds = evaluate_model(test_bags, desc="Test")
        print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1_m:.4f} | Weighted F1: {test_f1_w:.4f}")
        print("\nDetailní report (Test):")
        print(classification_report(test_lbls, test_preds, target_names=target_names, zero_division=0))
        
    # Finální update logu
    log_data["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if len(val_bags) > 0:
        log_data["results"]["val"] = {
            "loss": val_loss,
            "accuracy": val_acc,
            "macro_f1": val_f1_m,
            "weighted_f1": val_f1_w
        }
    if len(test_bags) > 0:
        log_data["results"]["test"] = {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": test_f1_m,
            "weighted_f1": test_f1_w
        }
        
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"\nVýsledky a hyperparametry uloženy do {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trénování baseline modelu EGNN-MIL")
    parser.add_argument('--data-path', default='data_prep/egnn_dataset.pt', help='Cesta k PyG 3D datasetu')
    parser.add_argument('--split-suffix', default='mil_0.5', help='Sufix pro rozdělení (např. mil_0.5 nebo mil)')
    parser.add_argument('--use-esm-split', action='store_true', help='Použít ESM embedding clustering split (_esm_0.2)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Dimenzionalita EGNN a MIL vrstev')
    parser.add_argument('--gated-attention', action='store_true', help='Použít Gated Attention (defaultně vypnuto)')
    parser.add_argument('--num-gnn-layers', type=int, default=3, help='Počet EGNN vrstev')
    parser.add_argument('--epochs', type=int, default=50, help='Max počet epoch')
    parser.add_argument('--batch-size', type=int, default=16, help='Velikost dávky (počet proteinů)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularizace')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Patience pro early stopping')
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'
    
    train_and_evaluate(args)
