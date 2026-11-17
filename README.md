# iLAB-G2P v2 — balanced ANI-group cross-validation of all 16,780 configurations

Genome-to-phenotype prediction of acid and bile-salt tolerance for 256 *Bifidobacterium* genomes.
Version 2 replaces the fold assignment only: the ANI ≥ 99.5% groups are now distributed over five folds by a
deterministic multi-objective optimisation, and all 16,780 configurations are re-run on the new frozen folds with the
same features, fusions, classifiers and hyper-parameters.

## Quick start (AutoDL server)

```bash
cd /root/autodl-tmp/iLABG2P
bash setup_server.sh            # install only missing packages (lightgbm, catboost, openpyxl) and self-check
screen -S g2p                   # keep running after SSH disconnect (or use tmux / nohup)
python run.py 2>&1 | tee run.log
```

- `python run.py` runs every step in one command: five-fold OOF for all 16,780 configurations → metric aggregation → before/after comparison report → Top10 per-fold stability →
  extra Route C exploration (separate report) → Top10 deployment model refitting → generation of the new `Supplementary_Material/`.
- Automatically uses all CPUs actually allocated to the container (cgroup quota of 18 cores → 18 single-threaded processes in parallel); override with `--workers N`.
- Resumable: each configuration is written to `results/v2_balancedANI/models/` as soon as it finishes; re-running `python run.py` after an interruption skips completed configurations.
- No GPU needed: encoder embeddings are precomputed and all classifiers are CPU models (no speed-up from a GPU at this sample size, and it would hurt reproducibility).
- Try a quick run first: `python run.py --check-only` (check only) or `python run.py --smoke` (end-to-end smoke test on a few dozen configurations).

## Frozen v2 folds

| Task | Strains per fold | ANI groups per fold | Tolerant fraction per fold |
|---|---|---|---|
| acid (175; 78+) previous | 18/6/88/31/32 | 5/3/17/14/20 | 0.056/0.667/0.489/0.419/0.531 |
| acid **v2** | 42/34/33/33/33 | 3/6/17/17/16 | 0.619/0.382/0.394/0.394/0.394 |
| bile (219; 96+) previous | 25/23/10/67/94 | 11/16/3/23/20 | 0.280/0.609/0.600/0.358/0.479 |
| bile **v2** | 53/43/41/41/41 | 1/15/19/19/19 | 0.528/0.395/0.415/0.415/0.415 |

The largest groups cannot be split (acid: ANI995_A6, 40 strains, 26 tolerant; bile: ANI995_A6, 53 strains, 28
tolerant), so fold 1 is necessarily larger, has fewer groups and a higher tolerant fraction. The remaining four folds
are balanced to within one strain. Group-count balance is a soft objective and is sacrificed where needed.

## Outputs

```text
results/v2_balancedANI/
  00_Frozen_ANI995_Folds.csv, 00_Fold_Provenance.json, 00_Protocol.json, 00_Cohort_256_*
  01_All_Models_All_Folds_All_Strains_OOF.csv   one row per configuration x strain (3.3 M rows)
  02_Per_Fold_Metrics.csv                       16,780 x 5 folds, incl. TP/TN/FP/FN
  03_Pooled_OOF_All_Metrics.csv                 16,780 pooled OOF metrics
  04_Top5_*, 05_Each_Strain_*, 99_*             as in v1
  models/                                       resumable per-configuration checkpoints
  cv_reevaluation/                              fold balance (10_*), Top10 per fold (11_*), v1 vs v2 (12_*),
                                                Top10 rank change (13_*), split stability (14_*),
                                                15_CV_Reevaluation_Report.md, taxonomy check (16_*), components (17_*)
  stability/                                    checkpoints of the 20 alternative partitions and the v1 folds
  extra_exploration_routeC/                     v2-derived Route C candidates (not part of the grid)
save_models/full/{acid,bile}/Top01..Top10.joblib   deployable refits selected on v2 OOF
Supplementary_Material/                        Data_S1–S10 (CSV), Supplementary_Tables_S1-S20.xlsx, README.md, MANIFEST.json
```

Tables S6–S14 (pangenome/enrichment) do not depend on the folds and are copied from `supplement_template/`.

## Run time and resources

About 2–4 h on 18 cores (benchmark: the heaviest configuration, all seven k-mer scales with concat_z and
GradientBoosting, takes ≈ 2 min; most take seconds); the stability analysis adds ≈ 840 model
fits and the packaging/supplement export ≈ 20–30 min. Peak memory is a few GB; results need ≈ 6 GB of disk
space (per-strain prediction CSVs ≈ 1.6 GB, supplement ≈ 1 GB).

## Prediction for new genomes

Put FASTA files in `test_data/` and run `python test.py` (default Top1 of each task; `--rank N` for others).
Models that use encoder features need authentic embeddings in `test_data/embeddings/<encoder>.npz`.

## Notes

- All metrics are internal OOF estimates; Top-ranked models are selected on the same folds and are optimistically
  biased. They are not independent external-test results.
- Do not judge v2 by whether its scores are higher. v2 is preferred for group integrity, balance and lower
  fold-to-fold variance; the reports quantify these directly.
- `--rebuild-folds` re-optimises the folds (only needed if genomes, labels or ANI groups change) and invalidates all
  previous checkpoints.
- Third-party encoder licences: see `THIRD_PARTY_LICENSES.md`.
