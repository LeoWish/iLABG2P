# Supplementary Material for iLAB-G2P

Genome-to-phenotype prediction and population-aware genomic analysis of acid and bile-salt tolerance in *Bifidobacterium*.

## Contents

### Supplementary_Tables_S1-S20.xlsx

One workbook with a README sheet and one sheet per table.

| Sheet | Rows | Content |
|---|---:|---|
| Table_S1_Cohort | 256 | Cohort of 256 genomes: species, B. longum subspecies, strain and phenotype source (in-house; NCBI; literature), tolerance labels, ANI >= 99.5% group, validation fold of each labeled strain in the balanced ANI-group five-fold cross-validation, assembly statistics and QC flag, median ANI to own species, and NCBI accessions of the 45 NCBI genomes |
| Table_S2_Top100_models | 200 | Top 100 configurations per task ranked by the selection score (first 100 rows of Supplementary Data S1 and S2) |
| Table_S3_Component_summary | 84 | OOF performance summarized by route, fusion, classifier, single encoder, number of encoders, single k-mer scale and number of k-mer scales |
| Table_S4_Taxonomy_check | 22 | Pooled and within-species ROC-AUC of the ten best models of each task and the fold-local species-rate baseline |
| Table_S5_Top1_by_species | 21 | Per-species OOF performance of the Top1 model of each task |
| Table_S6_Pangenome_summary | 16 | Pangenome construction, association screen and enrichment summary |
| Table_S7_Population_structure | 30 | PERMANOVA, Mantel / partial Mantel and within-species distance statistics on Jaccard P/A distances |
| Table_S8_Profiles_CMH | 1088 | All pooled-significant exact P/A profiles (BH q < 0.05) with the species-stratified CMH re-test (no continuity correction; BH over all testable profiles) |
| Table_S9_Candidate_modules | 27 | All ranked positive candidate modules passing the prespecified prioritization rules (acid 12, bile 15) |
| Table_S10_Top5_evidence | 10 | Quantitative evidence for the ten Top5 candidate modules (carriers recomputed from P/A vectors) |
| Table_S11_Top5_members | 68 | All member families of the ten Top5 modules with eggNOG, GO, COG, KEGG and CAZy annotation |
| Table_S12_CMH_sensitivity | 211 | CMH stratification cascade (species; QC-flagged assemblies excluded; B. longum subspecies; species x source; subspecies x source) and all profiles with CMH q < 0.05 under species strata |
| Table_S13_Enriched_terms | 26 | GO, COG and KEGG terms with BH q < 0.05 (both association directions pooled) |
| Table_S14_Direction_enrichment | 22 | Direction-stratified COG and KEGG pathway enrichment (positive and negative families tested separately; terms with q < 0.25) |
| Table_S15_CV_fold_design | 20 | Composition of the five validation folds (strains, ANI >= 99.5% groups, class balance, training-fold size and species composition) under the current balanced ANI-group assignment and the previous StratifiedGroupKFold assignment |
| Table_S16_Top10_per_fold | 100 | Per-fold metrics, confusion matrices and species composition of the validation folds for the Top10 models of each task |
| Table_S17_Fold_scheme_change | 32 | Change of pooled OOF metrics of all configurations between the previous and the current fold assignment, by task and route (identical configurations, encoders, k-mers, fusions, classifiers and hyper-parameters) |
| Table_S18_Top10_rank_change | 40 | Top10 models of each task under the current and previous fold assignments with their rank under the other assignment |
| Table_S19_Split_stability | 40 | Split stability of the Top10 models (current and previous Top10 of each task) re-evaluated on 20 alternative balanced ANI-group partitions: mean, SD and range of pooled metrics, rank within the model set and within-partition fold SD |
| Table_S20_Ranking_agreement | 2 | Agreement of the Top10 model ranking between the current partition, the 20 alternative balanced partitions and the previous folds |

### Supplementary Data (CSV, UTF-8)

| File | Rows | Content |
|---|---:|---|
| Data_S1_Acid_all_8280_models_OOF_performance.csv | 8,280 | All 8,280 acid-tolerance configurations: specification and pooled OOF metrics, ranked by selection score |
| Data_S2_Bile_all_8500_models_OOF_performance.csv | 8,500 | All 8,500 bile-salt-tolerance configurations: specification and pooled OOF metrics, ranked by selection score |
| Data_S3_Acid_RouteA_encoder_OOF_predictions.csv | 271,250 | Strain-level OOF predictions, acid task, Route A (pretrained encoder embeddings) |
| Data_S4_Acid_RouteB_kmer_OOF_predictions.csv | 1,111,250 | Strain-level OOF predictions, acid task, Route B (multiscale k-mer profiles) |
| Data_S5_Acid_RouteC_cross_OOF_predictions.csv | 66,500 | Strain-level OOF predictions, acid task, Route C (cross-route combinations) |
| Data_S6_Bile_RouteA_encoder_OOF_predictions.csv | 339,450 | Strain-level OOF predictions, bile task, Route A (pretrained encoder embeddings) |
| Data_S7_Bile_RouteB_kmer_OOF_predictions.csv | 1,390,650 | Strain-level OOF predictions, bile task, Route B (multiscale k-mer profiles) |
| Data_S8_Bile_RouteC_cross_OOF_predictions.csv | 131,400 | Strain-level OOF predictions, bile task, Route C (cross-route combinations) |
| Data_S9_All_models_per_fold_metrics_and_confusion.csv | 83,900 | Per-fold metrics and confusion matrices (TP, TN, FP, FN) of every configuration in each of the five validation folds |
| Data_S10_Extra_RouteC_exploration_not_in_grid.csv | 800 | Extra exploration only: Route C combinations re-derived from the v2 Route A/B Top5 that are not in the frozen grid (reported separately; excluded from Data S1-S9 and from model selection) |

## Conventions

- Tasks: acid tolerance (survival >= 50% after 3 h at pH 3.0 for in-house strains) and bile-salt tolerance (bile-induced growth delay LT < 4 h for in-house strains). Labels: 1 = tolerant, 0 = non-tolerant; strains without a label were not used for the corresponding task.
- Sources: In-house (this study); NCBI (genomes downloaded from NCBI; accessions in Table S1); Literature (62 *B. adolescentis* strains provided by the corresponding author of Li et al., 2026, China Dairy Industry 54(8):5-12, doi:10.19827/j.issn1001-2230.2026.08.001).
- Cross-validation: five folds built separately for each task from the connected components of the skani ANI >= 99.5% graph (89 groups). All labeled strains of a group are assigned to the same validation fold. The assignment was obtained by lexicographic mixed-integer optimisation (scipy/HiGHS) that uses only group sizes and labels: (1) balance of strain numbers and of the tolerant fraction across folds, with at least two tolerant and two non-tolerant strains in every validation fold; (2) balance of the number of ANI groups per fold; (3) a fixed, data-independent tie-break. Largest groups cannot be split, so the fold that contains them is larger and holds fewer groups. Resulting folds:
  - acid: strains per fold 42/34/33/33/33; ANI groups per fold 3/6/17/17/16; tolerant fraction per fold 0.619/0.382/0.394/0.394/0.394.
  - bile: strains per fold 53/43/41/41/41; ANI groups per fold 1/15/19/19/19; tolerant fraction per fold 0.528/0.395/0.415/0.415/0.415.
  - Fold file SHA-256: aad2d123f916e32e32ba54730954d533f834b2828dc0a312e606e6e52a0c22d6.
- All feature selection, scaling and classifier fitting are performed within the training folds. OOF = out-of-fold.
- Selection score = 0.6 x PR-AUC (average precision) + 0.4 x MCC; binary calls at threshold 0.5.
- Encoders: DNABERT-S; prokbert-mini; NTv2-100M; GENERanno-0.5B; JEPA-DNA-NTv3-100M. k-mers: canonical, hashed to 4,096 bins per k (k = 4-10).
- Route C: the cross-route candidate set is identical to the previously reported grid (acid 380, bile 600 configurations). Route C candidates re-derived from the current Route A/B ranking are reported separately (Data S10) and are not part of the 16,780 configurations.
- Tables S15-S20 compare the current fold assignment with the previous StratifiedGroupKFold assignment. A higher metric under one assignment is not by itself evidence of a better scheme; the current assignment was chosen for group integrity and balance before any model was evaluated.
- Pangenome statistics (Tables S6-S14): Panaroo families collapsed to exact presence/absence profiles; Fisher's exact test with Benjamini-Hochberg FDR; species-stratified Cochran-Mantel-Haenszel re-test. All gene associations are exploratory.
- File integrity: SHA-256 checksums of all files are listed in MANIFEST.json.
