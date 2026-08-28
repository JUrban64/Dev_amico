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
from model_encoder_mil import EGNN_Encoder_MIL_Classifier

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
    
    # 3. Inicilizace sjednoceného EGNN Encoder MIL modelu
    model = EGNN_Encoder_MIL_Classifier(
        node_dim=1280,
        egnn_hidden_dim=args.hidden_dim,
        mil_hidden_dim=args.mil_hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_classes=5,
        dropout=args.dropout,
        num_heads=args.num_heads,
        attention_temp=args.attention_temp,
        gated_attention=args.gated_attention
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)
    
    def evaluate_model(bags_list, desc="Eval"):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for b in bags_list:
                batch_graph = b['batch_graph'].to(device)
                label = b['label'].to(device).view(1)
                
                logits, _ = model.forward_bag(batch_graph)
                loss = criterion(logits, label)
                
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=-1).item()
                
                all_preds.append(pred)
                all_labels.append(label.item())
                
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
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_bags)
        train_loss = 0.0
        
        for b in train_bags:
            batch_graph = b['batch_graph'].to(device)
            label = b['label'].to(device).view(1)
            
            optimizer.zero_grad()
            logits, _ = model.forward_bag(batch_graph)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= max(len(train_bags), 1)
        val_loss, val_acc, val_f1_m, val_f1_w, _, _ = evaluate_model(val_bags, desc="Val")
        
        if epoch % 5 == 0 or epoch == 1 or early_stopping(val_loss):
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1_m:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'egnn_encoder_mil_best.pt')
            
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
    if os.path.exists('egnn_encoder_mil_best.pt'):
        model.load_state_dict(torch.load('egnn_encoder_mil_best.pt', weights_only=True))
        
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
    parser.add_argument('--hidden-dim', type=int, default=512, help='Dimenzionalita EGNN')
    parser.add_argument('--mil-hidden-dim', type=int, default=256, help='Dimenzionalita MIL vrstev')
    parser.add_argument('--num-heads', type=int, default=1, help='Počet attention hlav')
    parser.add_argument('--attention-temp', type=float, default=1.0, help='Attention teplota')
    parser.add_argument('--gated-attention', action='store_true', help='Použít Gated Attention')
    parser.add_argument('--num-gnn-layers', type=int, default=3, help='Počet EGNN vrstev')
    parser.add_argument('--epochs', type=int, default=50, help='Max počet epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularizace')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Patience pro early stopping')
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'
    
    train_and_evaluate(args)
