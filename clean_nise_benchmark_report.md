# Benchmark CLEAN na Nehomologních Isofunkčních Enzymech (NISE)

Vygenerováno: 2026-09-24 19:07:51

Srovnání modelu CLEAN (kontrastivní sekvenční predikce EC čísel) na datech nehomologních enzymů.

### Souhrnné výsledky

| Model | Total_Samples | Accuracy | Macro_F1 | Weighted_F1 | Time_Sec | ms_per_sample | F1_acetyl-CoA | F1_ATP | F1_B12 | F1_FAD | F1_NAD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_upper_bound_oracle | 75 | 70.67 | 70.25 | 70.25 | 0.0 | 0.0 | 88.89 | 82.35 | 80.0 | 33.33 | 66.67 |
| clean_predictions | 75 | 62.67 | 65.93 | 65.93 | 0.0 | 0.0 | 84.62 | 70.97 | 74.07 | 33.33 | 66.67 |

### Ukázka detailních predikcí (prvních 15 vzorků)

| UniProt_ID | Cofactor_GroundTruth | Annotated_EC | SCOP_Superfamily | Protein_Name | clean_upper_bound_oracle_Pred | clean_upper_bound_oracle_Correct | clean_predictions_Pred | clean_predictions_Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0A0I9QGZ7 | acetyl-CoA | 2.3.1.46 | SSF53474 | Homoserine O-succinyltransferase (HST) (EC 2.3.1.4 | acetyl-CoA | True | acetyl-CoA | True |
| O55171 | acetyl-CoA | 3.1.2.2 | SSF53474 | Acyl-coenzyme A thioesterase 2, mitochondrial (Acy | acetyl-CoA | True | acetyl-CoA | True |
| A0A1D3PCI9 | acetyl-CoA | 2.3.1.46 | SSF52317 | Homoserine O-succinyltransferase (HST) (EC 2.3.1.4 | acetyl-CoA | True | acetyl-CoA | True |
| P81185 | acetyl-CoA | 6.4.1.3 | SSF52440 | Propionyl-CoA carboxylase alpha chain (PCCase) (EC | ATP | False | ATP | False |
| Q7TN73 | acetyl-CoA | 2.3.1.45 | SSF47954 | N-acetylneuraminate (7)9-O-acetyltransferase (EC 2 | acetyl-CoA | True | acetyl-CoA | True |
| A0PSI5 | acetyl-CoA | 2.3.3.9 | SSF51645 | Malate synthase G (EC 2.3.3.9) | acetyl-CoA | True | acetyl-CoA | True |
| P16558 | acetyl-CoA | 2.3.1.235 | SSF51182 | Tetracenomycin polyketide synthase protein TcmJ (E | acetyl-CoA | True | Miss (-1) | False |
| A0A0D2YG05 | acetyl-CoA | 2.3.1.31 | SSF53474 | Homoserine O-acetyltransferase FUB5 (EC 2.3.1.31)  | acetyl-CoA | True | acetyl-CoA | True |
| Q5RCH8 | acetyl-CoA | 1.3.1.38 | SSF51735 | Peroxisomal trans-2-enoyl-CoA reductase (TERP) (EC | NAD | False | NAD | False |
| Q1LW89 | acetyl-CoA | 2.3.1.45 | SSF52266 | N-acetylneuraminate (7)9-O-acetyltransferase (EC 2 | acetyl-CoA | True | acetyl-CoA | True |
| P0A651 | acetyl-CoA | 2.3.1.20 | SSF52777 | Putative diacyglycerol O-acyltransferase Mb3154c ( | acetyl-CoA | True | acetyl-CoA | True |
| O27114 | acetyl-CoA | 1.2.7.3 | SSF53323 | 2-oxoglutarate synthase subunit KorC (EC 1.2.7.3)  | acetyl-CoA | True | acetyl-CoA | True |
| P02904 | acetyl-CoA | 2.1.3.1 | SSF51230 | Methylmalonyl-CoA carboxyltransferase 1.3S subunit | acetyl-CoA | True | acetyl-CoA | True |
| Q42561 | acetyl-CoA | 3.1.2.14 | SSF54637 | Oleoyl-acyl carrier protein thioesterase 1, chloro | Miss (-1) | False | Miss (-1) | False |
| P07855 | acetyl-CoA | 2.3.1.85 | SSF47336 | Fatty acid synthase (EC 2.3.1.85) | acetyl-CoA | True | acetyl-CoA | True |
