"""iLAB-G2P v2: cohort/fold provenance, parallel OOF execution and post-hoc reports.

Re-uses the frozen v1 model engine (core/legacy_engine.py: features, fold-local transforms, the ten
classifiers with their fixed hyper-parameters, metrics). Only the fold assignment changes: the
ANI >= 99.5% groups are distributed over five folds by core/balanced_ani_folds.py.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, os, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from core import legacy_engine as engine
from core import balanced_ani_folds as BF

VERSION = 'iLAB-G2P-256-OOF-v2.0-BalancedANI-20261116'
OOF_PROTOCOL_TEXT = 'ANI>=99.5% skani groups; balanced group assignment (5 folds)'
METRIC_KEYS = ('ROC_AUC', 'PR_AUC', 'MCC', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1', 'Specificity',
               'Accuracy', 'Selection_Score')


def log(*a):
    engine.log(*a)


def need(cond, msg):
    if not cond:
        raise RuntimeError('[BLOCKED] ' + msg)


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def file_sha(p):
    return engine.sha(p)


# ------------------------------------------------------------------------------------------------
# Resources
def granted_cpus():
    """CPUs this process may really use: min(affinity, cgroup quota)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = None
    try:
        q, p = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        if q != 'max':
            quota = float(q) / float(p)
    except Exception:
        try:
            q = float(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            p = float(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if q > 0:
                quota = q / p
        except Exception:
            pass
    if quota:
        n = min(n, max(1, int(math.floor(quota + 1e-9))))
    return max(1, n)


def resolve_workers(setting):
    if isinstance(setting, int) and setting > 0:
        return setting
    return granted_cpus()


# ------------------------------------------------------------------------------------------------
# Cohort, ANI groups and folds
def load_cohort(root, out):
    records, cohort = engine.audit_cohort(root, out)
    return records, cohort


def labels_digest(records):
    return engine.digest([(r['id'], r['acid'], r['bile']) for r in records])


def load_groups(records, path):
    rows = read_csv(path)
    index = {r['id']: r for r in records}
    need(len(rows) == 256, 'ANI group file must list 256 strains')
    g = {}
    for row in rows:
        sid = row['Strain_ID']
        need(sid in index and sid not in g, 'Unexpected/repeated strain in ANI group file: ' + sid)
        need(row['Genome_SHA256'] == index[sid]['genome_sha'], 'Genome SHA256 differs from frozen ANI grouping: ' + sid)
        need(bool(row['ANI995_Group']), 'Empty ANI group: ' + sid)
        g[sid] = row['ANI995_Group']
    groups = [g[r['id']] for r in records]
    need(len(set(groups)) == 89, 'Frozen ANI >= 99.5% grouping must contain 89 groups')
    for i, r in enumerate(records):                      # identical genome files must share a group
        for j in range(i):
            if records[j]['genome_sha'] == r['genome_sha']:
                need(groups[i] == groups[j], 'Identical genomes in different ANI groups')
    return groups


def _fold_rows_to_map(rows):
    out = {t: {} for t in BF.TASKS}
    for r in rows:
        out[r['Task']][r['Strain_ID']] = int(r['Fold'])
    return out


def balanced_folds(records, groups, cohort, settings, rebuild=False):
    """Load the frozen balanced folds (verifying every lock) or build and freeze them."""
    fpath, ppath = settings.BALANCED_FOLDS, settings.BALANCED_PROVENANCE
    gsha = file_sha(settings.ANI_GROUPS)
    expect = {'genome_digest': cohort['genome_digest'], 'labels_digest': labels_digest(records),
              'ani_groups_sha256': gsha}
    if fpath.exists() and ppath.exists() and not rebuild:
        prov = json.loads(ppath.read_text(encoding='utf-8'))
        need(prov.get('folds_sha256') == file_sha(fpath), 'Balanced fold CSV was modified after freezing')
        for k, v in expect.items():
            need(prov.get(k) == v, f'Balanced folds were frozen for a different {k}; rebuild with --rebuild-folds')
        rows = read_csv(fpath)
        for r in rows:
            i = next(j for j, x in enumerate(records) if x['id'] == r['Strain_ID'])
            need(groups[i] == r['ANI_Group'], 'ANI group mismatch in frozen balanced folds: ' + r['Strain_ID'])
            need(str(records[i][r['Task']]) == r['True_Label'], 'Label mismatch in frozen balanced folds: ' + r['Strain_ID'])
        fold_of = _fold_rows_to_map(rows)
        log('[FOLDS] frozen balanced ANI folds verified:', fpath.name)
    else:
        log('[FOLDS] building balanced ANI-group folds (MILP; labels and group sizes only)')
        t0 = time.time()
        fold_of, rows, report = BF.build_partition(records, groups, cohort['genome_digest'])
        BF.write_csv(fpath, rows, ['Task', 'Fold', 'Strain_ID', 'True_Label', 'ANI_Group', 'Group_Protocol', 'Cohort'])
        import scipy
        prov = dict(expect, version=VERSION, scheme=BF.SCHEME, folds_sha256=file_sha(fpath),
                    folds_file=fpath.name, n_folds=BF.K, min_class_per_validation_fold=BF.MIN_CLASS,
                    objective_stage1='sum_f Q(|n_f-N/5|) + %.1f * sum_f Q(|p_f-pi*n_f|); Q = convex piecewise-linear '
                                     'approximation of d(d+1)/2' % BF.W_POS,
                    objective_stage2='sum_f Q(|k_f-G/5|) subject to stage1 <= optimum + %.1f' % BF.STAGE1_SLACK,
                    objective_stage3='fixed canonical tie-break (data-independent)',
                    uses_model_predictions_or_metrics=False, solver='scipy.optimize.milp (HiGHS)',
                    scipy_version=scipy.__version__, solve_seconds=round(time.time() - t0, 1),
                    solver_report=report, independent_test_set=False)
        engine.atomic_json(ppath, prov)
        log('[FOLDS] balanced folds frozen:', fpath.name, prov['folds_sha256'][:16])
    folds = BF.to_index_folds(records, groups, fold_of)
    return fold_of, folds, prov


def v1_folds(records, groups, settings):
    """Original v1 StratifiedGroupKFold assignment (archived), validated against labels and ANI groups."""
    rows = read_csv(settings.V1_FOLDS)
    index = {r['id']: i for i, r in enumerate(records)}
    for r in rows:
        i = index[r['Strain_ID']]
        need(groups[i] == r['ANI_Group'], 'v1 fold ANI group mismatch: ' + r['Strain_ID'])
        need(str(records[i][r['Task']]) == r['True_Label'], 'v1 fold label mismatch: ' + r['Strain_ID'])
    fold_of = _fold_rows_to_map(rows)
    return fold_of, BF.to_index_folds(records, groups, fold_of)


def stability_partitions(records, groups, cohort, settings, rebuild=False):
    """Alternative balanced partitions (fixed seeds) used only for the split-stability analysis."""
    fpath, ppath = settings.STABILITY_FOLDS, settings.STABILITY_PROVENANCE
    if fpath.exists() and ppath.exists() and not rebuild:
        prov = json.loads(ppath.read_text(encoding='utf-8'))
        need(prov.get('sha256') == file_sha(fpath), 'Stability partition file modified after freezing')
        need(prov.get('genome_digest') == cohort['genome_digest'] and prov.get('labels_digest') == labels_digest(records),
             'Stability partitions frozen for another cohort; rebuild with --rebuild-folds')
        rows = read_csv(fpath)
    else:
        log('[STABILITY] generating', settings.STABILITY_PARTITIONS, 'alternative balanced partitions')
        rows, seen, reports, seed = [], set(), {}, settings.STABILITY_SEED0
        main = _fold_rows_to_map(read_csv(settings.BALANCED_FOLDS))
        main_key = tuple(sorted((t, frozenset(s for s in main[t] if main[t][s] == f)) for t in main for f in range(1, 6)))
        seen.add(main_key)
        pid = 0
        while pid < settings.STABILITY_PARTITIONS:
            seed += 1
            fo, frows, rep = BF.build_partition(records, groups, cohort['genome_digest'], alt_seed=seed,
                                                tol1=settings.STABILITY_TOL_STAGE1, tol2=settings.STABILITY_TOL_STAGE2)
            key = tuple(sorted((t, frozenset(s for s in fo[t] if fo[t][s] == f)) for t in fo for f in range(1, 6)))
            if key in seen:
                continue
            seen.add(key); pid += 1
            name = f'ALT{pid:02d}'
            reports[name] = dict(seed=seed, **{t: {k: v for k, v in rep[t].items() if k != 'solver_messages'} for t in rep})
            rows += [dict(r, Partition=name, Seed=seed) for r in frows]
            log('[STABILITY] partition', name, 'seed', seed)
        BF.write_csv(fpath, rows, ['Partition', 'Seed', 'Task', 'Fold', 'Strain_ID', 'True_Label', 'ANI_Group',
                                   'Group_Protocol', 'Cohort'])
        prov = {'version': VERSION, 'sha256': file_sha(fpath), 'genome_digest': cohort['genome_digest'],
                'labels_digest': labels_digest(records), 'partitions': settings.STABILITY_PARTITIONS,
                'tolerance_stage1': settings.STABILITY_TOL_STAGE1, 'tolerance_stage2': settings.STABILITY_TOL_STAGE2,
                'uses_model_predictions_or_metrics': False, 'reports': reports}
        engine.atomic_json(ppath, prov)
    parts = defaultdict(list)
    for r in rows:
        parts[r['Partition']].append(r)
    out = {}
    for name in sorted(parts):
        fo = _fold_rows_to_map(parts[name])
        out[name] = (fo, BF.to_index_folds(records, groups, fo))
    return out


# ------------------------------------------------------------------------------------------------
# Model configurations
EXPECTED = {('acid', 'A_ENCODER'): 1550, ('acid', 'B_KMER'): 6350, ('acid', 'C_CROSS'): 380,
            ('bile', 'A_ENCODER'): 1550, ('bile', 'B_KMER'): 6350, ('bile', 'C_CROSS'): 600}


def load_configs(path):
    rows = read_csv(path)
    need(len(rows) == 16780, 'Configuration list must contain 16,780 rows')
    need(Counter((r['Task'], r['Route']) for r in rows) == Counter(EXPECTED), 'Route/task counts differ from v1')
    ident = {(r['Task'], r['Route'], r['Upstream'], r['Middle'], r['Downstream']) for r in rows}
    need(len(ident) == 16780, 'Duplicate configurations')
    for r in rows:
        need(r['Middle'] in engine.MIDDLES and r['Downstream'] in engine.CLASSIFIERS, 'Unknown fusion/classifier')
    return rows


def job_cost(job, dims):
    task, route, up, mid, clf = job
    if mid == 'concat_z':
        d = sum(dims[u] for u in up)
    else:
        d = sum(min(dims[u], 256 if mid in ('block256_z', 'select128') else 128) for u in up)
        if mid == 'select128':
            d = 128 + 0.5 * d
    w = {'LR_L2': 0.3, 'LR_L1': 0.3, 'SVM_RBF': 0.5, 'RF': 1.0, 'ExtraTrees': 0.8, 'HistGB': 1.5,
         'GradientBoosting': 3.0, 'XGBoost': 1.2, 'LightGBM': 1.0, 'CatBoost': 2.5}[clf]
    return (d + 200) * w * (1.25 if task == 'bile' else 1.0)


# ------------------------------------------------------------------------------------------------
# Parallel execution (fork-based process pool; read-only data inherited by copy-on-write)
_G = {}


def _worker(job):
    import warnings
    warnings.simplefilter('ignore')          # library deprecation chatter only; failures are still captured
    try:
        res = engine.evaluate(job, _G['records'], _G['folds'], _G['features'], _G['groups'], _G['meta'], _G['out'])
        return res, None
    except Exception as e:
        return None, {'job': str(job), 'reason': repr(e), 'traceback': traceback.format_exc(limit=5)}


def run_jobs(jobs, records, folds, features, groups, meta, out, workers, label, max_tasks_per_child=400):
    import multiprocessing as mp
    dims = {k: v.shape[1] for k, v in features.items()}
    jobs = sorted(jobs, key=lambda j: -job_cost(j, dims))          # longest first (LPT scheduling)
    _G.update(records=records, folds=folds, features=features, groups=groups, meta=meta, out=out)
    results, failures = [], []
    t0 = time.time()
    log(f'[START] {label}: {len(jobs)} configurations on {workers} worker processes')
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=workers, maxtasksperchild=max_tasks_per_child) as pool:
        for n, (res, fail) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            if res is not None:
                results.append(res)
            else:
                failures.append(fail)
                if len(failures) <= 6:
                    log('[MODEL FAILED]', fail['job'], fail['reason'][:250])
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                eta = el / n * (len(jobs) - n)
                log(f'[PROGRESS] {label} {n}/{len(jobs)} pass {len(results)} fail {len(failures)} '
                    f'elapsed {el/3600:.2f} h ETA {eta/3600:.2f} h')
    engine.write_csv(out / '99_Model_Failures.csv', failures, ['job', 'reason', 'traceback'])
    return results, failures


def set_engine_identity(features):
    """Version/feature fingerprints that key every cached OOF checkpoint (prevents v1/v2 mixing)."""
    engine.VERSION = VERSION
    engine.SOURCE_SHA256 = engine.sha(Path(engine.__file__).resolve())
    engine.FEATURE_HASHES = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                             for k, v in sorted(features.items())}


# ------------------------------------------------------------------------------------------------
# Lightweight OOF scoring (for reports; reads the per-configuration checkpoints)
def oof_scores(result, folds, records):
    task = result['task']
    idx, ff = folds[task]
    with np.load(result['path'], allow_pickle=False) as z:
        s, p, fid = z['score'], z['pred'], z['fold']
    y = np.array([records[i][task] for i in idx], dtype=int)
    return idx, y, s, p, fid


def pooled_and_fold_metrics(result, folds, records):
    idx, y, s, p, fid = oof_scores(result, folds, records)
    pooled = engine.metrics(y, s[idx], p[idx])
    per = []
    for k, (_, te) in enumerate(folds[result['task']][1], 1):
        yk = np.array([records[i][result['task']] for i in te], dtype=int)
        per.append(dict(Fold=k, **engine.metrics(yk, s[te], p[te])))
    return pooled, per


def within_species_auc(y, s, sp):
    num = den = 0.0
    for g in np.unique(sp):
        m = sp == g
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) and len(neg):
            num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            den += len(pos) * len(neg)
    return (num / den if den else float('nan')), int(den)


def kendall_tau(a, b):
    """Kendall tau-b between two rankings given as dicts item->rank (common items only)."""
    items = sorted(set(a) & set(b))
    n = len(items)
    if n < 2:
        return float('nan')
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(a[items[i]] - a[items[j]]); db = np.sign(b[items[i]] - b[items[j]])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    d = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / d if d else float('nan')
