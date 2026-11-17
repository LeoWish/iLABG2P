# Supplementary Material for iLAB-G2P

Genome-to-phenotype prediction and population-aware genomic analysis of acid and bile-salt tolerance in *Bifidobacterium*.

## Contents

### Supplementary_Tables_S1-S14.xlsx

One workbook with a README sheet and one sheet per table, numbered in the order of first citation in the main text.

| Sheet | Rows | Content | Cited in |
|---|---:|---|---|
| Table_S1_Cohort | 256 | Cohort of 256 genomes (species, subspecies, source, labels, ANI group, assembly statistics, QC flag, NCBI accessions) | Sections 2.1, 2.2.3, 3.6 |
| Table_S2_Top100_models | 200 | Top 100 configurations per task | Section 3.3, Table 3 |
| Table_S3_Component_summary | 84 | Performance by route, fusion, classifier, encoder and k-mer scale | Section 3.4 |
| Table_S4_Taxonomy_check | 22 | Pooled and within-species ROC-AUC; species-rate baseline | Section 3.5 |
| Table_S5_Top1_by_species | 21 | Per-species performance of the Top1 models | Section 3.5 |
| Table_S6_Pangenome_summary | 16 | Pangenome construction, association screen and enrichment summary | Section 3.7 |
| Table_S7_Population_structure | 30 | PERMANOVA, Mantel and partial Mantel statistics | Section 3.7 |
| Table_S8_Profiles_CMH | 1088 | Pooled-significant P/A profiles with species-stratified CMH re-test | Section 3.8 |
| Table_S9_Candidate_modules | 27 | Ranked positive candidate modules | Section 3.8 |
| Table_S10_Top5_evidence | 10 | Evidence for the ten Top5 modules | Section 3.8, Table 4 |
| Table_S11_Top5_members | 68 | Member families and annotations of the Top5 modules | Section 3.8, Table 4 |
| Table_S12_CMH_sensitivity | 211 | CMH stratification sensitivity cascade | Section 3.8 |
| Table_S13_Enriched_terms | 26 | Enriched GO, COG and KEGG terms (pooled directions) | Section 3.9 |
| Table_S14_Direction_enrichment | 22 | Direction-stratified COG and KEGG enrichment | Section 3.9 |

### Supplementary Data S1–S8 (CSV, UTF-8)

Cited in Section 2.4.4 and the Data Availability Statement.

| File | Rows | Content |
|---|---:|---|
| Data_S1_Acid_all_8280_models_OOF_performance.csv | 8,280 | All 8,280 acid-tolerance configurations: specification and pooled OOF metrics, ranked by selection score |
| Data_S2_Bile_all_8500_models_OOF_performance.csv | 8,500 | All 8,500 bile-tolerance configurations: specification and pooled OOF metrics, ranked by selection score |
| Data_S3_Acid_RouteA_encoder_OOF_predictions.csv | 271,250 | Strain-level OOF predictions, acid task, Route A (pretrained encoder embeddings) |
| Data_S4_Acid_RouteB_kmer_OOF_predictions.csv | 1,111,250 | Strain-level OOF predictions, acid task, Route B (multiscale k-mer profiles) |
| Data_S5_Acid_RouteC_cross_OOF_predictions.csv | 66,500 | Strain-level OOF predictions, acid task, Route C (cross-route combinations) |
| Data_S6_Bile_RouteA_encoder_OOF_predictions.csv | 339,450 | Strain-level OOF predictions, bile task, Route A |
| Data_S7_Bile_RouteB_kmer_OOF_predictions.csv | 1,390,650 | Strain-level OOF predictions, bile task, Route B |
| Data_S8_Bile_RouteC_cross_OOF_predictions.csv | 131,400 | Strain-level OOF predictions, bile task, Route C |

## Conventions

- Tasks: acid tolerance (survival >= 50% after 3 h at pH 3.0 for in-house strains) and bile-salt tolerance (bile-induced growth delay LT < 4 h for in-house strains). Labels: 1 = tolerant, 0 = non-tolerant; strains without a label were not used for the corresponding task.
- Sources: In-house (this study); NCBI (genomes downloaded from NCBI; accessions in Table S1); Literature (62 *B. adolescentis* strains provided by the corresponding author of Li et al., 2026, China Dairy Industry 54(8):5-12, doi:10.19827/j.issn1001-2230.2026.08.001).
- Cross-validation: StratifiedGroupKFold (5 folds, seed 20261114) with groups defined as connected components of the skani ANI >= 99.5% graph. OOF = out-of-fold.
- Selection score = 0.6 x PR-AUC (average precision) + 0.4 x MCC; binary calls at threshold 0.5.
- Encoders: DNABERT-S; prokbert-mini; NTv2-100M; GENERanno-0.5B; JEPA-DNA-NTv3-100M. k-mers: canonical, hashed to 4,096 bins per k (k = 4-10).
- Pangenome statistics: Panaroo families collapsed to exact presence/absence profiles; Fisher's exact test with Benjamini-Hochberg FDR; species-stratified Cochran-Mantel-Haenszel re-test. All gene associations are exploratory.
- File integrity: SHA-256 checksums of all files are listed in MANIFEST.json.
