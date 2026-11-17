"""iLAB-G2P v2 settings (balanced ANI-group five-fold cross-validation)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# Parallelism. 'auto' = number of CPUs actually granted to this container (cgroup quota and CPU
# affinity are both respected; on the AutoDL A800 instance this is 18). Every worker process runs
# exactly one single-threaded model configuration, so 'auto' saturates the granted CPUs without
# oversubscription. Override with `python run.py --workers N`.
WORKERS = 'auto'
MAX_TASKS_PER_CHILD = 400      # recycle worker processes periodically (bounded memory growth)

# ----------------------------------------------------------------------------------------------
# Frozen experimental protocol (identical to v1 except for the fold assignment)
WINDOWS = 16
WINDOW_BP = 1024
KMER_BINS = 4096
PREDICTION_THRESHOLD = 0.5
ENCODERS = ('DNABERT-S', 'prokbert-mini', 'NTv2-100M', 'GENERanno-0.5B', 'JEPA-DNA-NTv3-100M')
TASKS = ('acid', 'bile')
SAVE_TOP_PER_TASK = 10         # deployable models refitted on all labeled strains

# ----------------------------------------------------------------------------------------------
# Split-stability analysis of the Top10 models
STABILITY_PARTITIONS = 20      # alternative balanced ANI-group partitions (fixed seeds)
STABILITY_SEED0 = 20261200
STABILITY_TOL_STAGE1 = 0.10    # alternatives may be up to 10% less balanced (strain/class criterion)
STABILITY_TOL_STAGE2 = 0.30    # ... and up to 30% less balanced in ANI-group counts

# ----------------------------------------------------------------------------------------------
# Paths
PLAN = ROOT / 'plan'
CONFIGS = PLAN / '16780_model_configurations.csv'
ANI_GROUPS = PLAN / 'frozen_256_ANI995_groups.csv'
BALANCED_FOLDS = PLAN / '01_Balanced_ANI995_Folds.csv'
BALANCED_PROVENANCE = PLAN / '01_Balanced_Fold_Provenance.json'
STABILITY_FOLDS = PLAN / '02_Stability_Alternative_Partitions.csv'
STABILITY_PROVENANCE = PLAN / '02_Stability_Partitions_Provenance.json'
ARCHIVE_V1 = ROOT / 'archive' / 'v1_StratifiedGroupKFold_20261114'
V1_FOLDS = ARCHIVE_V1 / 'plan' / '00_Frozen_ANI995_Folds.csv'
V1_METRICS = ARCHIVE_V1 / 'plan' / 'v1_16780_pooled_OOF_metrics.csv'
RESULTS = ROOT / 'results' / 'v2_balancedANI'
SUPPLEMENT_TEMPLATE = ROOT / 'supplement_template' / 'Supplementary_Tables_S1-S14_v1.xlsx'
SUPPLEMENT_OUT = ROOT / 'Supplementary_Material'
