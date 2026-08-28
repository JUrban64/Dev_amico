import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset import load_data_from_tensors, load_split_ids
from model import AttentionMIL_ESM
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features_list = [item['features'] for item in batch]
    labels = torch.cat([item['label'] for item in batch])
    
    padded_features = pad_sequence(features_list, batch_first=True)
    
    lengths = torch.tensor([f.size(0) for f in features_list])
    batch_size = len(features_list)
    max_len = padded_features.size(1)
    
    padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    
    return padded_features, padding_mask, labels


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
    print(f"Using device: {device}")
    
    # 1. Load data
    bags = load_data_from_tensors(args.data_path, mode=args.mode)
    
    # 2. Split data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix, use_nr=args.use_nr)
    
    print("\n--- KONTROLA SPLIT SOUBORŮ ---")
    print(f"Hledaný suffix: '{args.split_suffix}' (use_nr={args.use_nr})")
    print(f"Nalezené ID v train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    if not train_ids or not test_ids:
        print("Varování: Split soubory nebyly nalezeny ve složce data_prep/.")
        print("Dostupné split soubory ve složce data_prep/:")
        available_splits = [os.path.basename(f) for f in os.listdir(os.path.join(base_dir, 'data_prep')) if f.endswith('.txt') and ('train' in f or 'val' in f or 'test' in f)]
        print(" ", available_splits if available_splits else "Žádné .txt splity nenalezeny.")
        print("Ujistěte se, že spouštíte trénink se správným argumentem --split-suffix (např. --split-suffix mil_0.5 nebo struct_pocket_0.5_0.5) a případně --use-nr.\n")
    print("-------------------------------\n")
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        print("Using splits from text files...")
        for b in bags:
            pid = b['protein_id']
            if pid in train_ids:
                train_bags.append(b)
            elif pid in val_ids:
                val_bags.append(b)
            elif pid in test_ids:
                test_bags.append(b)
            else:
                pass
    else:
        print("Split files not found. Using random 80/10/10 split...")
        np.random.seed(42)
        np.random.shuffle(bags)
        n = len(bags)
        train_bags = bags[:int(n*0.8)]
        val_bags = bags[int(n*0.8):int(n*0.9)]
        test_bags = bags[int(n*0.9):]
        
    print(f"Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    if len(train_bags) == 0:
        return
        
    # 3. Model setup
    all_labels = set([b['label'].item() for b in bags])
    num_classes = max(all_labels) + 1
    
    # ESM dim is 1280
    model = AttentionMIL_ESM(
        in_features=1280,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        num_heads=args.num_heads,
        attention_temp=args.attention_temp,
        gated_attention=args.gated_attention
    ).to(device)
    
    # Class weights for imbalanced datasets
    class_weights = None
    if args.balance_classes:
        train_labels = [b['label'].item() for b in train_bags]
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes).float()
        # Invert counts to get weights (add epsilon to avoid division by zero)
        class_weights = 1.0 / (class_counts + 1e-5)
        # Normalize weights so they sum to num_classes
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=args.patience)
    
    train_loader = DataLoader(train_bags, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_bags, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_bags, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for features, mask, labels in train_loader:
            features, mask, labels = features.to(device), mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for features, mask, labels in val_loader:
                features, mask, labels = features.to(device), mask.to(device), labels.to(device)
                logits, _ = model(features, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            saved_str = "(Model Saved)"
        else:
            saved_str = ""
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or saved_str != "":
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f} {saved_str}")
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print("\nTraining completed.")
    
    # 4. Final Evaluation
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    eval_loader = test_loader if args.evaluate_test and len(test_bags) > 0 else val_loader
    eval_name = "Test" if args.evaluate_test and len(test_bags) > 0 else "Validation"
    
    eval_preds, eval_labels = [], []
    with torch.no_grad():
        for features, mask, labels in eval_loader:
            features, mask, labels = features.to(device), mask.to(device), labels.to(device)
            logits, _ = model(features, mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            eval_preds.extend(preds)
            eval_labels.extend(labels.cpu().numpy())
            
    print(f"\nFinal {eval_name} Results:")
    acc = accuracy_score(eval_labels, eval_preds)
    f1 = f1_score(eval_labels, eval_preds, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    # Target names for report
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    present_classes = np.unique(np.concatenate([eval_labels, eval_preds]))
    target_names_subset = [target_names[i] for i in present_classes if i < len(target_names)]
    
    print("\nClassification Report:")
    try:
        print(classification_report(eval_labels, eval_preds, target_names=target_names_subset, zero_division=0))
    except Exception:
        print(classification_report(eval_labels, eval_preds, zero_division=0))
        
    # Uložení výsledků do JSON logu
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"esm_mil_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "args": vars(args),
        "dataset_split": {
            "train": len(train_bags),
            "val": len(val_bags),
            "test": len(test_bags)
        },
        "evaluation_set": eval_name,
        "accuracy": acc,
        "macro_f1": f1
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"\nVýsledky a hyperparametry uloženy do {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='residues', 
                        help='Mód dat: pockets (1 vektor per kapsa) nebo residues (1 vektor per aminokyselina)')
    parser.add_argument('--split-suffix', default='mil_0.5', help='Přípona textových souborů se splity (např. mil_0.5, struct_pocket_0.5_0.5).')
    parser.add_argument('--use-esm-split', action='store_true', help='Použít ESM embedding clustering split (_esm_0.2)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--attention-temp', type=float, default=1.0)
    parser.add_argument('--gated-attention', action='store_true') # False by default
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--balance-classes', action='store_true', default=True, help='Použít váženou CrossEntropy loss pro nevyvážené třídy')
    parser.add_argument('--model-path', default='best_esm_mil.pt')
    
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'

    train_and_evaluate(args)
