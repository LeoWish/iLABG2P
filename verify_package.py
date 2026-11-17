"""Read-only package checks for iLAB-G2P v2 (no training, no skani needed).

    python verify_package.py --basic   source layout only
    python verify_package.py           data, frozen folds, features, packages and (if present) saved models
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ('run.py', 'train.py', 'test.py', 'model.py', 'settings.py', 'launcher.py', 'requirements.txt',
           'core/legacy_engine.py', 'core/balanced_ani_folds.py', 'core/pipeline_v2.py', 'core/reports_v2.py',
           'core/supplementary_v2.py', 'plan/16780_model_configurations.csv', 'plan/frozen_256_ANI995_groups.csv',
           'plan/01_Balanced_ANI995_Folds.csv', 'plan/01_Balanced_Fold_Provenance.json',
           'plan/02_Stability_Alternative_Partitions.csv', 'plan/02_Stability_Partitions_Provenance.json',
           'supplement_template/Supplementary_Tables_S1-S14_v1.xlsx',
           'archive/v1_StratifiedGroupKFold_20261114/plan/00_Frozen_ANI995_Folds.csv',
           'archive/v1_StratifiedGroupKFold_20261114/plan/v1_16780_pooled_OOF_metrics.csv')


def need(cond, msg):
    if not cond:
        raise SystemExit('[BLOCKED] ' + msg)


def check_basic():
    for f in SOURCES:
        need((ROOT / f).is_file(), 'missing ' + f)
    print('[PASS] source code, frozen plan, balanced folds, stability partitions, v1 archive')
    print('[PYTHON]', sys.version.split()[0])


def check_real():
    check_basic()
    sys.path.insert(0, str(ROOT))
    import settings as S
    import train
    from core import legacy_engine as engine
    from core import pipeline_v2 as P
    out = ROOT / 'cacher' / 'package_audit'
    records, cohort, groups, fold_of, folds, prov = train.prepare(out)
    print('[PASS] 256 genomes, 89 ANI groups, frozen balanced folds', prov['folds_sha256'][:16])
    for t in ('acid', 'bile'):
        idx, ff = folds[t]
        print(f'       {t}: labeled {len(idx)} | validation folds ' + '/'.join(str(len(v)) for _, v in ff))
    P.load_configs(S.CONFIGS)
    print('[PASS] 16,780 configurations (acid 8,280 / bile 8,500)')
    for enc in S.ENCODERS:
        p = ROOT / 'encodemodels' / enc / 'features' / (enc + '.npz')
        need(p.is_file(), 'missing encoder features ' + str(p))
        a = engine.load_npz_matrix(p, records)
        print('[PASS] encoder features', enc, a.shape)
    ready, missing = engine.available_clfs()
    print('[PASS] classifiers', ', '.join(ready)) if not missing else print('[MISSING]', ', '.join(missing),
                                                                           '-> bash setup_server.sh')
    import joblib
    full = ROOT / 'save_models' / 'full'
    if (full / 'acid' / 'Top01.joblib').is_file():
        for task in ('acid', 'bile'):
            for i in range(1, S.SAVE_TOP_PER_TASK + 1):
                need((full / task / f'Top{i:02d}.joblib').is_file(), f'missing {task} Top{i:02d}')
            obj = joblib.load(full / task / 'Top01.joblib')
            need(obj.get('folds_sha256') == prov['folds_sha256'], 'saved models were not selected on the current folds')
            print(f'[PASS] {task}: 10 refitted models | Top1 {"+".join(obj["upstream"])} {obj["fusion"]} {obj["classifier_name"]}')
    else:
        print('[INFO] save_models/full is empty until python run.py has finished')
    print('[READY]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--basic', action='store_true')
    a = ap.parse_args()
    check_basic() if a.basic else check_real()
