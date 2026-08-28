import os
import argparse
import json
import torch
import numpy as np

from model import LigandCrossAttentionMIL, TARGET_NAMES

class AMICOPredictor:
    """
    Inferenční třída pro LigandCrossAttentionMIL.
    Umožňuje predikovat kofaktor, lokalizovat vazebnou kapsu a odhadovat epistemickou nejistotu pomocí MC Dropoutu.
    """
    def __init__(self, checkpoint_path=None, config_json=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)

        hidden_dim = 256
        num_heads = 4
        dropout = 0.15

        if config_json and os.path.exists(config_json):
            with open(config_json, 'r') as f:
                cfg = json.load(f)
                hidden_dim = cfg.get('hidden_dim', hidden_dim)
                num_heads = cfg.get('num_heads', num_heads)
                dropout = cfg.get('dropout', dropout)

        self.model = LigandCrossAttentionMIL(
            feature_dim=1280,
            ecfp_dim=1024,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=5,
            dropout=dropout
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state)
            print(f"Checkpoint načten z {checkpoint_path}")
        else:
            print("Upozornění: Model inicializován bez načtení checkpointu.")

        self.model.eval()

    def _enable_mc_dropout(self):
        """Ponechá model v eval módu, ale aktivuje Dropout vrstvy pro MC vzorkování."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict(self, pocket_features, full_protein_feature, mc_samples=30, 
                confidence_threshold=0.50, uncertainty_threshold=0.15):
        """
        Provede inferenci s volitelným Monte Carlo Dropout vzorkováním.
        
        Args:
            pocket_features: Tensor [N_pockets, 1280] nebo np.ndarray
            full_protein_feature: Tensor [1280] nebo np.ndarray
            mc_samples: Počet stochastických průchodů (1 = standardní deterministický režim, >=20 pro MC Dropout)
            confidence_threshold: Minimální průměrná pravděpodobnost pro potvrzení vazby kofaktoru
            uncertainty_threshold: Maximální povolená směrodatná odchylka (rozptyl) pro jistou predikci

        Returns:
            dict s kompletními výsledky predikce, nejistoty a detekce True Negatives
        """
        if isinstance(pocket_features, np.ndarray):
            pocket_features = torch.FloatTensor(pocket_features)
        if isinstance(full_protein_feature, np.ndarray):
            full_protein_feature = torch.FloatTensor(full_protein_feature)

        if pocket_features.dim() == 2:
            pocket_features = pocket_features.unsqueeze(0) # [1, N, 1280]
        if full_protein_feature.dim() == 1:
            full_protein_feature = full_protein_feature.unsqueeze(0) # [1, 1280]

        pocket_features = pocket_features.to(self.device)
        full_protein_feature = full_protein_feature.to(self.device)
        padding_mask = torch.zeros(1, pocket_features.size(1), dtype=torch.bool, device=self.device)

        if mc_samples > 1:
            # === MONTE CARLO DROPOUT REŽIM ===
            self._enable_mc_dropout()
            all_probs = []
            all_attns = []

            with torch.no_grad():
                for _ in range(mc_samples):
                    logits, attn_weights = self.model(pocket_features, padding_mask, full_protein_feature)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    attn = attn_weights.squeeze(0).cpu().numpy()
                    all_probs.append(probs)
                    all_attns.append(attn)

            all_probs = np.stack(all_probs, axis=0) # [T, 5]
            all_attns = np.stack(all_attns, axis=0) # [T, 5, N+1]

            mean_probs = np.mean(all_probs, axis=0) # [5]
            std_probs = np.std(all_probs, axis=0)   # [5] (Epistemická nejistota)
            mean_attn = np.mean(all_attns, axis=0)  # [5, N+1]
            
            # Prediktivní entropie: H = - sum(p * log(p))
            predictive_entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-12)))

            pred_idx = int(np.argmax(mean_probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(mean_probs[pred_idx])
            uncertainty = float(std_probs[pred_idx])
            
            # Detekce True Negatives (Nevazačů / Out-of-Distribution proteinů)
            is_non_binder = (confidence < confidence_threshold) or (uncertainty > uncertainty_threshold)
            binding_status = "NON_BINDER / UNKNOWN_COFACTOR" if is_non_binder else "BINDER"

            probs_dict = {
                TARGET_NAMES[i]: {
                    "mean_probability": round(float(mean_probs[i]), 4),
                    "uncertainty_std": round(float(std_probs[i]), 4)
                }
                for i in range(len(TARGET_NAMES))
            }
            used_attn = mean_attn

        else:
            # === DETERMINISTICKÝ REŽIM ===
            self.model.eval()
            with torch.no_grad():
                logits, attn_weights = self.model(pocket_features, padding_mask, full_protein_feature)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                used_attn = attn_weights.squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            uncertainty = 0.0
            predictive_entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            is_non_binder = confidence < confidence_threshold
            binding_status = "NON_BINDER / UNKNOWN_COFACTOR" if is_non_binder else "BINDER"

            probs_dict = {
                TARGET_NAMES[i]: {
                    "probability": round(float(probs[i]), 4)
                }
                for i in range(len(TARGET_NAMES))
            }

        # Interpretace vazebných kapes
        cofactor_attn = used_attn[pred_idx]
        global_context_weight = float(cofactor_attn[0])
        pocket_attn_weights = cofactor_attn[1:]

        best_pocket_idx = int(np.argmax(pocket_attn_weights)) + 1 if len(pocket_attn_weights) > 0 else None
        best_pocket_weight = float(np.max(pocket_attn_weights)) if len(pocket_attn_weights) > 0 else 0.0

        pocket_rankings = [
            {"pocket_id": i + 1, "attention_score": round(float(w), 4)}
            for i, w in sorted(enumerate(pocket_attn_weights), key=lambda x: x[1], reverse=True)
        ]

        return {
            "predicted_cofactor": "NONE (Non-binder)" if is_non_binder else pred_label,
            "raw_top_class": pred_label,
            "binding_status": binding_status,
            "confidence": round(confidence, 4),
            "uncertainty_std": round(uncertainty, 4),
            "predictive_entropy": round(predictive_entropy, 4),
            "is_confident_prediction": not is_non_binder,
            "mc_samples_used": mc_samples,
            "probabilities": probs_dict,
            "best_binding_pocket": best_pocket_idx if not is_non_binder else None,
            "best_pocket_attention": round(best_pocket_weight, 4),
            "global_context_weight": round(global_context_weight, 4),
            "pocket_rankings": pocket_rankings
        }

def main():
    parser = argparse.ArgumentParser(description="AMICO Ligand Cross-Attention Inference s Monte Carlo Dropoutem & Dokováním")
    parser.add_argument('--protein-id', type=str, default='sample_protein', help='ID proteinu')
    parser.add_argument('--checkpoint', type=str, default='ligand_cross_mil_best.pt', help='Cesta k vahám modelu')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguraci JSON')
    parser.add_argument('--mc-samples', type=int, default=30, help='Počet MC Dropout vzorků (1 = deterministický, 30 = MC Dropout)')
    parser.add_argument('--confidence-thresh', type=float, default=0.50, help='Minimální jistota pro klasifikaci jako vazač')
    parser.add_argument('--uncertainty-thresh', type=float, default=0.15, help='Maximální rozptyl pro klasifikaci jako vazač')
    parser.add_argument('--dock', action='store_true', help='Automaticky nadokovat předpovězený kofaktor do identifikované kapsy')
    parser.add_argument('--protein-pdb', type=str, default=None, help='Cesta k PDB souboru proteinu pro dokování')
    parser.add_argument('--pocket-center', nargs=3, type=float, default=None, help='Souřadnice středu kapsy x y z (volitelné)')
    parser.add_argument('--dock-out', type=str, default='docking_results', help='Složka pro uložení výsledků dokování')
    args = parser.parse_args()

    predictor = AMICOPredictor(checkpoint_path=args.checkpoint, config_json=args.config)

    # Ukázková data (3 predikované kapsy + 1 globální sekvenční embedding)
    dummy_pockets = torch.randn(3, 1280)
    dummy_full_prot = torch.randn(1280)

    res = predictor.predict(
        dummy_pockets, 
        dummy_full_prot, 
        mc_samples=args.mc_samples,
        confidence_threshold=args.confidence_thresh,
        uncertainty_threshold=args.uncertainty_thresh
    )

    print("\n" + "="*60)
    print(f"      VÝSLEDEK INFERENCE: {args.protein_id}")
    print("="*60)
    print(f"Status vazby:            {res['binding_status']}")
    print(f"Předpovězený kofaktor:   {res['predicted_cofactor']}")
    print(f"Průměrná jistota (mean): {res['confidence']*100:.2f} %")
    if args.mc_samples > 1:
        print(f"Nejistota (MC std dev):  ±{res['uncertainty_std']*100:.2f} % (vzorků: {res['mc_samples_used']})")
        print(f"Prediktivní entropie:    {res['predictive_entropy']:.4f}")
    
    if res['is_confident_prediction']:
        print(f"\nLokalizace vazebného místa:")
        print(f" -> Nejlepší kapsa:      Pocket #{res['best_binding_pocket']} (Attention: {res['best_pocket_attention']:.4f})")
        print(f" -> Vliv celého enzymu:  {res['global_context_weight']:.4f}")
    else:
        print("\n⚠️ Model vyhodnotil protein jako NE-VAZAČ (True Negative) nebo je predikce příliš nejistá.")

    print("\nPravděpodobnosti jednotlivých kofaktorů:")
    for name, p_data in res['probabilities'].items():
        if args.mc_samples > 1:
            mean_p = p_data['mean_probability'] * 100
            std_p = p_data['uncertainty_std'] * 100
            print(f" - {name:<12}: {mean_p:6.2f} %  (±{std_p:4.2f} %)")
        else:
            p = p_data['probability'] * 100
            print(f" - {name:<12}: {p:6.2f} %")

    print("\nPořadí kapes podle chemické kompatibility:")
    for p_info in res['pocket_rankings']:
        print(f" - Kapsa #{p_info['pocket_id']:<2}: Attention = {p_info['attention_score']:.4f}")

    # Dokování předpovězeného kofaktoru, pokud je vyžádáno
    if args.dock:
        if not res['is_confident_prediction']:
            print("\n[DOCKING SKIP] Dokování přeskočeno: protein byl vyhodnocen jako nevazač.")
        else:
            from docking_utils import dock_predicted_cofactor, get_pocket_center_from_pdb

            pred_cofactor = res['raw_top_class']
            pdb_path = args.protein_pdb if args.protein_pdb else "sample_protein.pdb"
            
            if args.pocket_center:
                center = np.array(args.pocket_center)
            elif os.path.exists(pdb_path):
                center = get_pocket_center_from_pdb(pdb_path)
            else:
                center = np.array([0.0, 0.0, 0.0])

            dock_res = dock_predicted_cofactor(
                protein_pdb=pdb_path,
                cofactor_name=pred_cofactor,
                pocket_center=center,
                out_dir=args.dock_out
            )
            print(f"\n✨ Dokování dokončeno. Výstupní soubory uloženy do: {dock_res['output_dir']}/")

if __name__ == '__main__':
    main()
