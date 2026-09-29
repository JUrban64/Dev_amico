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

from dataset_egnn_cross_mil import get_egnn_cross_splits, egnn_cross_collate_fn
from model_egnn_self_attention_mil import EGNN_Self_Attention_MIL

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

def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"\n========================================================")
    print(f"  TRÉNOVÁNÍ MODELU: EGNN + SELF-ATTENTION MIL")
    print(f"========================================================")
    print(f"Použité zařízení: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)
    
    # 1. Načtení dat a splitů
    train_bags, val_bags, test_bags = get_egnn_cross_splits(
        data_path=data_path,
        full_proteins_path=full_proteins_path,
        base_dir=base_dir,
        split_suffix=args.split_suffix,
        use_nr=args.use_nr
    )
    
    if len(train_bags) == 0:
        print("Chyba: Trénovací dataset je prázdný.")
        return
        
    print(f"\nVelikost datasetu: Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    # 2. Vyvážení tříd
    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)
    class_weights = total_samples / (5.0 * np.maximum(class_counts, 1))
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Počty vzorků ve třídách (Train): {class_counts}")
    print(f"Váhy tříd pro CrossEntropy: {class_weights.cpu().numpy().round(3)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    
    # 3. Inicializace EGNN + Self-Attention MIL modelu
    model = EGNN_Self_Attention_MIL(
        node_dim=1280,
        full_protein_dim=1280,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_heads=args.num_heads,
        num_attn_layers=args.num_attn_layers,
        num_classes=5,
        dropout=args.dropout,
        pocket_drop_prob=args.pocket_drop_prob,
        use_pocket_pooling_in_head=not args.no_pocket_pooling
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=args.patience)
    
    # 4. DataLoader
    train_loader = DataLoader(train_bags, batch_size=args.batch_size, shuffle=True, collate_fn=egnn_cross_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=args.batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn)
    test_loader = DataLoader(test_bags, batch_size=args.batch_size, shuffle=False, collate_fn=egnn_cross_collate_fn) if len(test_bags) > 0 else None
    
    def evaluate_loader(loader, desc="Eval"):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for mega_batch, protein_idx, full_prot_feats, labels in loader:
                mega_batch = mega_batch.to(device)
                protein_idx = protein_idx.to(device)
                full_prot_feats = full_prot_feats.to(device)
                labels = labels.to(device)
                
                logits, _ = model(mega_batch, protein_idx, full_prot_feats)
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / max(len(all_labels), 1)
        acc = accuracy_score(all_labels, all_preds)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return avg_loss, acc, f1_m, f1_w, all_labels, all_preds

    print("\n--- Spouštím trénování ---")
    best_val_loss = float('inf')
    best_model_path = 'best_egnn_self_attention_mil.pt'
    
    log_file = f"train_log_egnn_self_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "history": []
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_samples_epoch = 0
        
        for mega_batch, protein_idx, full_prot_feats, labels in train_loader:
            mega_batch = mega_batch.to(device)
            protein_idx = protein_idx.to(device)
            full_prot_feats = full_prot_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(mega_batch, protein_idx, full_prot_feats)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            total_train_loss += loss.item() * len(labels)
            total_samples_epoch += len(labels)
            
        train_loss = total_train_loss / max(total_samples_epoch, 1)
        
        # Validace
        val_loss, val_acc, val_f1_m, val_f1_w, _, _ = evaluate_loader(val_loader, desc="Val")
        scheduler.step(val_loss)
        
        saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            saved = True
            
        print(f"Epocha {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1_m:.4f} {'[ULOŽENO]' if saved else ''}")
        
        log_data["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_m,
            "val_f1_weighted": val_f1_w
        })
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping aktivováno po epoše {epoch} (nejlepší Val Loss: {best_val_loss:.4f})")
            break
            
    print("\n--- Načítám nejlepší model pro finální evaluaci ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    if len(val_bags) > 0:
        val_loss, val_acc, val_f1_m, val_f1_w, val_lbls, val_preds = evaluate_loader(val_loader, desc="Val")
        print(f"\n[VALIDACE] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1_m:.4f}")
        print(classification_report(val_lbls, val_preds, target_names=target_names, zero_division=0))
        
    if test_loader is not None and len(test_bags) > 0:
        test_loss, test_acc, test_f1_m, test_f1_w, test_lbls, test_preds = evaluate_loader(test_loader, desc="Test")
        print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1_m:.4f}")
        print(classification_report(test_lbls, test_preds, target_names=target_names, zero_division=0))
        
    log_data["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"\nVýsledky trénování uloženy do {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trénování EGNN + Self-Attention MIL modelu")
    parser.add_argument('--data-path', default='data_prep/egnn_dataset.pt', help='Cesta k 3D grafovému datasetu (PyG Data)')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt', help='Cesta k full protein embeddingům')
    parser.add_argument('--split-suffix', default='mil_0.5', help='Přípona souborů se splity')
    parser.add_argument('--use-esm-split', action='store_true', help='Použít ESM embedding clustering split (_esm_0.2)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Dimenze skrytých vrstev')
    parser.add_argument('--num-gnn-layers', type=int, default=2, help='Počet EGNN vrstev')
    parser.add_argument('--num-heads', type=int, default=4, help='Počet attention hlav v Transformeru')
    parser.add_argument('--num-attn-layers', type=int, default=1, help='Počet vrstev Self-Attention Transformeru')
    parser.add_argument('--no-pocket-pooling', action='store_true', help='Použít pouze CLS token pro klasifikaci (bez mean pooling kapes)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='L2 regularizace')
    parser.add_argument('--dropout', type=float, default=0.35, help='Dropout')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--pocket-drop-prob', type=float, default=0.15, help='Stochastic Pocket Drop pravděpodobnost')
    parser.add_argument('--patience', type=int, default=12)
    
    args = parser.parse_args()
    
    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'
        
    train_and_evaluate(args)
