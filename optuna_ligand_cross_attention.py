import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("CHYBA: Optuna není nainstalována! Nainstalujte ji pomocí: pip install optuna")
    sys.exit(1)

from dataset import load_split_ids
from dataset_cross_mil import load_cross_mil_data, CrossMilDataset, custom_collate_fn
from model_ligand_cross_attention_mil import LigandCrossAttentionMIL

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

def evaluate_model(model, loader, criterion, device):
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
            
    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    return avg_loss, acc, f1_macro

def create_objective(all_bags, train_ids, val_ids, test_ids, device, max_epochs=30, base_patience=8):
    # Připravení dat
    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)
            
    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)
    class_weights = total_samples / (5.0 * np.maximum(class_counts, 1))
    class_weights = torch.FloatTensor(class_weights).to(device)

    def objective(trial):
        # 1. Výběr hyperparametrů
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 384])
        
        # Num heads musí dělit hidden_dim
        valid_heads = [h for h in [2, 4, 8] if hidden_dim % h == 0]
        num_heads = trial.suggest_categorical("num_heads", valid_heads)
        
        lr = trial.suggest_float("lr", 3e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        scheduler_type = trial.suggest_categorical("scheduler_type", ["plateau", "cosine", "none"])

        train_loader = DataLoader(CrossMilDataset(train_bags), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(CrossMilDataset(val_bags), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
        model = LigandCrossAttentionMIL(
            feature_dim=1280,
            ecfp_dim=1024,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=5,
            dropout=dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        else:
            scheduler = None

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_loss, val_acc, val_f1_m = evaluate_model(model, val_loader, criterion, device)
            
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(val_f1_m)
                else:
                    scheduler.step()

            # Optuna pruning report
            trial.report(val_f1_m, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_f1_m > best_val_f1:
                best_val_f1 = val_f1_m
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= base_patience:
                    break

        return best_val_f1

    return objective

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimizer pro Ligand-Cross-Attention MIL")
    parser.add_argument('--n-trials', type=int, default=40, help='Počet Optuna pokusů')
    parser.add_argument('--timeout', type=int, default=None, help='Maximální čas běhu v sekundách')
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt')
    parser.add_argument('--split-suffix', default='mil_0.5')
    parser.add_argument('--use-esm-split', action='store_true')
    parser.add_argument('--use-nr', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=25, help='Max epoch na trial')
    parser.add_argument('--study-name', default=None, help='Název Optuna study (výchozí: ligand_cross_<suffix>)')
    parser.add_argument('--storage', default=None, help='SQLite URI (výchozí: sqlite:///optuna_ligand_cross_<suffix>.db)')
    
    args = parser.parse_args()
    if args.use_esm_split:
        args.split_suffix = 'esm_0.2'

    # Výchozí persistentní SQLite databáze a název study
    clean_sfx = args.split_suffix.replace('_', '').replace('.', '')
    if args.study_name is None:
        args.study_name = f"ligand_cross_{clean_sfx}"
    if args.storage is None:
        args.storage = f"sqlite:///optuna_ligand_cross_{clean_sfx}.db"

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"\n========================================================")
    print(f"   OPTUNA OPTIMALIZACE PRO LIGAND-CROSS-ATTENTION MIL")
    print(f"========================================================")
    print(f"Zařízení: {device}")
    print(f"Split suffix: {args.split_suffix}")
    print(f"Databáze (průběžně ukládáno): {args.storage}")
    print(f"Počet trials: {args.n_trials} | Max epoch per trial: {args.max_epochs}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.data_path)
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path)

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix, use_nr=args.use_nr)

    objective_fn = create_objective(all_bags, train_ids, val_ids, test_ids, device, max_epochs=args.max_epochs)

    # Optuna Study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    out_json = f"best_params_ligand_cross_{args.split_suffix}.json"

    # Průběžný callback pro ukládání nejlepších parametrů
    def live_callback(study, trial):
        if study.best_trial.number == trial.number:
            print(f"\n🌟 Nový nejlepší výsledek v trialu #{trial.number}! Val Macro F1: {trial.value:.4f}")
            with open(out_json, "w") as f:
                json.dump({
                    "best_trial_number": trial.number,
                    "best_val_macro_f1": trial.value,
                    "best_params": trial.params,
                    "split_suffix": args.split_suffix,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4)
            print(f"   (Průběžně uloženo do {out_json})")

    print("\n--- Spouštím optimalizaci hyperparametrů ---")
    study.optimize(objective_fn, n_trials=args.n_trials, timeout=args.timeout, callbacks=[live_callback])

    print("\n========================================================")
    print("              OPTUNA VÝSLEDKY OPTIMALIZACE              ")
    print("========================================================")
    print(f"Nejlepší trial č.: {study.best_trial.number}")
    print(f"Nejlepší Val Macro F1: {study.best_value:.4f}")
    print("\nNejlepší parametry:")
    for k, v in study.best_params.items():
        print(f" - {k}: {v}")

    print(f"\nFinální konfigurace uložena do {out_json}")

if __name__ == '__main__':
    main()
