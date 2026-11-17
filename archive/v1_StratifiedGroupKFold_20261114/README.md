# Archived v1 results (do not mix with v2)

Fold scheme: StratifiedGroupKFold (5 folds, seed 20261114) over the 89 ANI >= 99.5% groups.

- plan/00_Frozen_ANI995_Folds.csv, plan/00_Fold_Provenance.json - v1 fold membership and provenance
- plan/v1_16780_pooled_OOF_metrics.csv - pooled OOF metrics of the 16,780 configurations under v1
- plan/10_FullFive_Completion_Manifest.json - v1 completion manifest
- save_models_full/ - v1 Top10 deployable models per task
- core/ - v1 fold loader and the pre-fix engine copy

v2 reads these files only for the explicit before/after comparison (results/v2_balancedANI/cv_reevaluation/12_*, 13_*, 14_*)
and for re-running the Top10 models on the v1 folds in the stability analysis. Nothing here is overwritten.
