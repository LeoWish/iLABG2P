"""iLAB-G2P v2 one-command run.

    python run.py                 full pipeline: 16,780 configurations on the balanced ANI-group folds,
                                  reports, split stability, Top10 refit and Supplementary_Material/
    python run.py --check-only    verify cohort, ANI groups, frozen folds, features and packages (no training)
    python run.py --folds-only    verify/build the frozen folds and write the fold-balance reports only
    python run.py --smoke         small end-to-end test (results/smoke, save_models/smoke)
    python run.py --refit-only    refit the deployable Top10 from an existing complete v2 result
    python run.py --workers 16    override the number of worker processes (default: all granted CPUs)

The run is resumable: every finished configuration is checkpointed under results/v2_balancedANI/models and is
reused when python run.py is started again.
"""
import os
# One BLAS/OpenMP thread per worker process (set before numpy is imported); parallelism comes from the pool.
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTHONHASHSEED', '0')

import argparse
import sys


def check_only():
    import settings as S
    from core import legacy_engine as engine
    from core import pipeline_v2 as P
    import train
    out = S.ROOT / 'cacher' / 'check'
    records, cohort, groups, fold_of, folds, prov = train.prepare(out)
    configs = P.load_configs(S.CONFIGS)
    feats = train.load_features(records)
    ready, missing = engine.available_clfs()
    print('[PASS] cohort 256 genomes | ANI groups', len(set(groups)), '| folds', prov['scheme'], prov['folds_sha256'][:16])
    print('[PASS] configurations', len(configs), '| features', len(feats))
    print('[CPU] granted', P.granted_cpus())
    if missing:
        print('[MISSING CLASSIFIERS]', ', '.join(missing), '-> run: bash setup_server.sh')
        sys.exit(2)
    print('[READY] python run.py can start')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--folds-only', action='store_true')
    ap.add_argument('--rebuild-folds', action='store_true', help='re-optimise and re-freeze the folds (not needed normally)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--refit-only', action='store_true')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--skip-stability', action='store_true')
    ap.add_argument('--skip-extra-routec', action='store_true')
    args = ap.parse_args()
    if args.check_only:
        check_only()
    else:
        import train
        if args.refit_only:
            train.refit_only(args)
        else:
            train.execute(args)
