import numpy as np
import os
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import torch

from dataset import load_data_from_tensors, load_split_ids

def aggregate_features(bag_features, pooling='mean'):
    """
    Agreguje features kapes do jednoho vektoru pro protein.
    bag_features: tensor tvaru [num_pockets, 1280]
    Vrací: 1D numpy array tvaru [1280] nebo [2560]
    """
    if pooling == 'mean':
        return bag_features.mean(dim=0).numpy()
    elif pooling == 'max':
        return bag_features.max(dim=0).values.numpy()
    elif pooling == 'mean_max':
        mean_feat = bag_features.mean(dim=0).numpy()
        max_feat = bag_features.max(dim=0).values.numpy()
        return np.concatenate([mean_feat, max_feat])
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

def train_and_evaluate(args):
    print("--- Načítání dat ---")
    bags = load_data_from_tensors(args.data_path, mode=args.mode)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix, use_nr=args.use_nr)
    
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
                train_bags.append(b)
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
        print("Chyba: Žádná trénovací data.")
        return

    print(f"\nAgreguji kapsy do vektorů pro každý protein (metoda: {args.pooling})...")
    X_train = np.array([aggregate_features(b['features'], args.pooling) for b in train_bags])
    y_train = np.array([b['label'].item() for b in train_bags])
    
    X_val = np.array([aggregate_features(b['features'], args.pooling) for b in val_bags])
    y_val = np.array([b['label'].item() for b in val_bags])
    
    if len(test_bags) > 0:
        X_test = np.array([aggregate_features(b['features'], args.pooling) for b in test_bags])
        y_test = np.array([b['label'].item() for b in test_bags])
    else:
        X_test, y_test = np.array([]), np.array([])
        
    all_labels = set(y_train)
    num_classes = max(all_labels) + 1
    
    sample_weights = None
    if args.balance_classes:
        class_counts = np.bincount(y_train, minlength=num_classes)
        total_samples = len(y_train)
        
        # Calculate class weights inversely proportional to class frequencies
        weights = {i: total_samples / (num_classes * count) if count > 0 else 1.0 
                   for i, count in enumerate(class_counts)}
        
        print(f"Class counts in Train: {dict(enumerate(class_counts))}")
        print(f"Calculated class weights: {weights}")
        
        sample_weights = np.array([weights[y] for y in y_train])

    print("\n--- Trénování XGBoost ---")
    
    log_file = f"train_log_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "model_type": "xgboost",
        "results": {}
    }
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective='multi:softmax' if num_classes > 2 else 'binary:logistic',
        num_class=num_classes if num_classes > 2 else None,
        random_state=42,
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        early_stopping_rounds=args.patience,
        n_jobs=args.n_jobs
    )
    
    # Eval set for early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=True
    )
    
    best_iteration = model.best_iteration
    print(f"\nNejlepší iterace (podle val_loss): {best_iteration}")
    
    # Save model
    model.save_model(args.model_path)
    print(f"Model uložen do {args.model_path}")
    
    print("\n--- Evaluace ---")
    eval_name = "Test" if args.evaluate_test and len(test_bags) > 0 else "Validation"
    X_eval = X_test if args.evaluate_test and len(test_bags) > 0 else X_val
    y_eval = y_test if args.evaluate_test and len(test_bags) > 0 else y_val
    
    if len(X_eval) > 0:
        y_pred = model.predict(X_eval)
        
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average='macro')
        
        print(f"\n--- Results on {eval_name} Set ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_eval, y_pred, zero_division=0))
        
        log_data["results"] = {
            "eval_name": eval_name,
            "accuracy": acc,
            "macro_f1": f1
        }
    else:
        print(f"Žádná data pro {eval_name} množinu.")
        
    log_data["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
        
    print(f"\nVýsledky a hyperparametry uloženy do {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='pockets', 
                        help='Mód dat. Pro XGBoost by měl být pockets.')
    parser.add_argument('--pooling', choices=['mean', 'max', 'mean_max'], default='mean',
                        help='Jak agregovat vektory kapes pro získání jednoho vektoru za protein.')
    
    # XGBoost hyperparametry
    parser.add_argument('--n-estimators', type=int, default=200, help='Počet stromů.')
    parser.add_argument('--max-depth', type=int, default=6, help='Maximální hloubka stromu.')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--subsample', type=float, default=0.8, help='Podíl vzorků pro každý strom.')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='Podíl příznaků pro každý strom.')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience.')
    parser.add_argument('--n-jobs', type=int, default=4, help='Počet vláken pro XGBoost.')
    
    # Ostatní nastavení
    parser.add_argument('--split-suffix', default='_mil_0.5', help='Přípona textových souborů se splity (např. _mil_0.9 nebo _mil).')
    parser.add_argument('--use-esm-split', action='store_true', help='Použít ESM embedding clustering split (_esm_0.2)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant (NR) variantu splitu')
    parser.add_argument('--balance-classes', action='store_true', default=True, help='Vyvážení tříd pomocí sample_weights.')
    parser.add_argument('--evaluate-test', action='store_true', help='Vyhodnotit na testovací množině místo validační.')
    parser.add_argument('--model-path', default='best_xgboost.json', help='Cesta pro uložení modelu.')
    
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'
    train_and_evaluate(args)
