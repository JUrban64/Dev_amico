import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from sklearn.metrics import f1_score

from dataset import load_data_from_tensors, load_split_ids
from model import AttentionMIL_ESM

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
            return True # Zlepšeno
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def objective(trial, train_bags, val_bags, num_classes, class_weights, device):
    # Definice prohledávaného prostoru (Hyperparametry)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
    gated_attention = trial.suggest_categorical("gated_attention", [True, False])
    attention_temp = trial.suggest_float("attention_temp", 0.5, 2.0, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = AttentionMIL_ESM(
        in_features=1280,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        num_heads=num_heads,
        attention_temp=attention_temp,
        gated_attention=gated_attention
    ).to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=10) # Menší patience pro zrychlení hledání
    
    epochs = 50 # Max epocha pro Optunu, spoléháme na early stopping
    
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_bags)
        
        optimizer.zero_grad()
        batch_count = 0
        
        for i, bag in enumerate(train_bags):
            features = bag['features'].to(device)
            label = bag['label'].to(device)
            
            logits, _ = model(features)
            loss = criterion(logits, label)
            
            loss_scaled = loss / batch_size
            loss_scaled.backward()
            
            batch_count += 1
            if batch_count == batch_size or i == len(train_bags) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0
                
        # Validation faza
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for bag in val_bags:
                features = bag['features'].to(device)
                label = bag['label'].to(device)
                logits, _ = model(features)
                
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                pred = torch.argmax(logits, dim=1)
                val_preds.append(pred.item())
                val_targets.append(label.item())
                
        avg_val_loss = val_loss / max(len(val_bags), 1)
        current_f1 = f1_score(val_targets, val_preds, average='macro')
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            
        # Report for early stopping pruning (optimalizujeme na F1, takže vracíme F1)
        trial.report(current_f1, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            break
            
    # Pro hyperparameter tuning Maximalizujeme F1:
    return best_f1

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 1. Load data
    bags = load_data_from_tensors(args.data_path, mode=args.mode)
    
    # 2. Split data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_ids, val_ids, test_ids = load_split_ids(base_dir)
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        print("Using splits from text files...")
        for b in bags:
            # Oprava _pocket_ už proběhla v dataset.py, b['protein_id'] je čisté
            pid = b['protein_id']
            if pid in train_ids:
                train_bags.append(b)
            elif pid in val_ids:
                val_bags.append(b)
            elif pid in test_ids:
                test_bags.append(b)
            else:
                train_bags.append(b)
    else:
        print("Error: Splity nenalezeny.")
        return
        
    print(f"Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    all_labels = set([b['label'].item() for b in bags])
    num_classes = max(all_labels) + 1
    
    class_weights = None
    if args.balance_classes:
        train_labels = [b['label'].item() for b in train_bags]
        class_counts = np.bincount(train_labels, minlength=num_classes)
        total_samples = len(train_bags)
        
        weights = []
        for count in class_counts:
            if count > 0:
                weights.append(total_samples / (num_classes * count))
            else:
                weights.append(1.0)
        
        class_weights = torch.FloatTensor(weights).to(device)
    
    # Vytvoření studie s SQLite databází (pro možnost navázání a logování)
    study_name = "AMICO_optuna"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True
    )
    
    print(f"Spouštím Optunu pro hledání nejlepších hyperparametrů (ukládám do {study_name}.db)...")
    
    # Lambda wrapper for objective function
    func = lambda trial: objective(trial, train_bags, val_bags, num_classes, class_weights, device)
    study.optimize(func, n_trials=args.n_trials)
    
    print("\n[HOTOVO] Nejlepší nalezené parametry:")
    print(study.best_params)
    print(f"Nejlepší hodnota F1 (validation): {study.best_value}")
    
    # Uložení do JSONu
    import json
    with open("optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print("Parametry byly úspěšně uloženy do souboru 'optuna_best_params.json'.")
    print(f"Kompletní historie běhů je v databázi '{study_name}.db'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='residues')
    parser.add_argument('--balance-classes', type=bool, default=True)
    parser.add_argument('--n-trials', type=int, default=30, help="Počet pokusů pro Optunu")
    
    args = parser.parse_args()
    main(args)
