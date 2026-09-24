import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset_cross_mil import load_cross_mil_data
from dataset import load_split_ids
from model_sequence_mlp import SequenceMLPClassifier

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def match_id(pid, id_set):
    if pid in id_set:
        return True
    base = pid.split('_')[0]
    return base in id_set

def main():
    parser = argparse.ArgumentParser(description="Trénování Sequence-only ESM-2 MLP baseline modelu (bez kapes a bez ligandů)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--split-suffix', type=str, default='mil_0.5')
    parser.add_argument('--pockets-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt')
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Používám zařízení: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.pockets_path) if not os.path.isabs(args.pockets_path) else args.pockets_path
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path) if not os.path.isabs(args.full_proteins_path) else args.full_proteins_path

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    print(f"Rozdělení -> Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")

    def collate_seq(batch):
        feats = torch.stack([item['full_protein_feature'] for item in batch])
        labels = torch.cat([item['label'] for item in batch])
        return feats, labels

    train_loader = DataLoader(train_bags, batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq)
    val_loader = DataLoader(val_bags, batch_size=args.batch_size, shuffle=False, collate_fn=collate_seq)
    test_loader = DataLoader(test_bags, batch_size=args.batch_size, shuffle=False, collate_fn=collate_seq) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    model = SequenceMLPClassifier(
        in_features=1280,
        hidden_dim=args.hidden_dim,
        num_classes=5,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=args.patience)

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
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, zero_division=0) if len(truths) > 0 else ""
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    print("\n--- Spouštím trénování Sequence ESM-2 MLP ---")
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for feats, lbl in train_loader:
            feats, lbl = feats.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(lbl)

        train_loss = train_loss_sum / max(len(train_bags), 1)
        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_msg = "🔥 (Model uložen)"
        else:
            save_msg = ""

        if epoch % 2 == 0 or epoch == 1 or save_msg:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} {save_msg}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping aktivován po {epoch} epochách.")
            break

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    print("\n" + "="*50)
    print("      VÝSLEDKY EVALUACE (Best Checkpoint)     ")
    print("="*50)
    _, val_acc, val_f1, val_rep = evaluate(val_loader)
    print(f"VALIDACE -> Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        print(f"TEST     -> Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
        print("\nDetailní Testovací Report:")
        print(test_rep)

if __name__ == '__main__':
    main()
