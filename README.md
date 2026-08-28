# AMICO (Attention-based Multiple Instance Classification Optimizer)

AMICO je vysoce výkonný systém pro predikci interakcí mezi proteiny a kofaktory. Využívá **Multiple Instance Learning (MIL)** aplikovaný nad hrubými embeddingy aminokyselin vytažených z velkých jazykových modelů (ESM-2), čímž zcela eliminuje nutnost stavět výpočetně drahé geometrické grafy (EGNN).

Tento projekt je "čistým řezem" (clean slate), který staví výlučně na architektuře **MIL + ESM Residue-level**, což je nejvýkonnější přístup objevený v předchozím projektu (EquiPocket-MIL).

## Architektura
Místo toho, aby se předpovězená kapsa z P2Ranku redukovala do jednoho zprůměrovaného vektoru, AMICO zpracovává **každou aminokyselinu kapsy zvlášť** (Residue-level). Všechna rezidua se vloží do jednoho "pytle" (bag) reprezentujícího protein.

Síť (`AttentionMIL`) následně provádí Multi-Head Attention, aby se sama naučila, na která rezidua (z průměrných ~250 instancí) se má zaměřit a která má ignorovat coby šum.

## Struktura repozitáře
* `model.py`: Definice architektury `AttentionMIL` (podporuje Gated Attention a Multi-Head Self-Attention).
* `dataset.py`: Zjednodušený dataloader, který nahrává čisté PyTorch tensory z `esm_dataset.pt` rovnou do paměti (kompletně opuštěno od PyTorch Geometric).
* `train.py`: Trénovací smyčka s automatickým balancováním tříd (Cross-Entropy s váhami), aby model neignoroval minoritní kofaktory (např. acetyl-CoA).
* `data_prep/`: Složka s nástroji pro přípravu dat z čistých PDB souborů.
  - `build_p2rank_dataset.py`: Hlavní pipeline, parsuje PDB soubory.
  - `extract_esm.py`: Extrakce ESM-2 embeddingů pro nalezená rezidua. Nahrazuje staré grafové generátory a je řádově rychlejší a prostorově úspornější.
  - `p2rank_sanity_check.py`: Diagnostika schopnosti P2Ranku nacházet kapsy pro konkrétní kofaktory (DCA < 4.0 Å).

## Spuštění tréninku
```bash
# Výchozí konfigurace automaticky používá nejlepší parametry (residue mode, 2 attention hlavy, class balancing)
python train.py --data-path data_prep/esm_dataset.pt
```

## Příprava dat od nuly (Data Prep Pipeline)
1. Rozdělte data do struktury podle tříd (např. `Binding_Sites/FAD/positive/`).
2. Vygenerujte hrubé JSON popisovače:
```bash
python data_prep/build_p2rank_dataset.py
```
3. Převeďte sekvence do ESM-2 embeddingů (už neřešíme PyG):
```bash
python data_prep/extract_esm.py --input p2rank_dataset.json --output esm_dataset.pt
```
4. Natrénujte model s použitím vygenerovaného `esm_dataset.pt`.
