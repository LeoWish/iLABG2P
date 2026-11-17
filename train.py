"""iLAB-G2P v2 pipeline: balanced ANI-group five-fold OOF for all 16,780 configurations, post-hoc
re-evaluation reports, split-stability analysis, deployable Top10 refit and Supplementary Material export.

Steps (python run.py):
  1  audit the 256-genome cohort (labels, species, genome SHA-256)
  2  load the frozen ANI >= 99.5% groups (89 groups)
  3  load + verify (or build + freeze) the balanced ANI-group folds   -> plan/01_*.csv/json
  4  load features (cached k-mers, five encoder embeddings)
  5  run all 16,780 configurations on the frozen v2 folds (parallel, resumable checkpoints)
  6  package per-strain OOF predictions, per-fold and pooled metrics
  7  re-evaluation reports (fold balance before/after, performance and Top10 rank change)
  8  split stability of the Top10 models (20 alternative balanced partitions + the v1 folds)
  9  extra exploration: Route C candidates re-derived from v2 (reported separately, never merged)
 10  refit the Top10 per task on all labeled strains -> save_models/full
 11  export the new Supplementary Material                                 -> Supplementary_Material/
"""
from __future__ import annotations
import csv, json, os, shutil, sys, time, warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import settings as S
from core import legacy_engine as engine
from core import balanced_ani_folds as BF
from core import pipeline_v2 as P
from core import reports_v2 as R

log = P.log
ANI_METHOD = 'skani_ANI995_groups_BalancedMILP_v2'
V1_SCHEME, V2_SCHEME = 'v1_StratifiedGroupKFold', 'v2_balanced'


# ------------------------------------------------------------------------------------------------
def load_features(records):
    opts = SimpleNamespace(windows=S.WINDOWS, window_bp=S.WINDOW_BP, kmer_bins=S.KMER_BINS)
    t0 = time.time()
    km = engine.kmer_features(records, S.ROOT / 'cacher', opts)
    log(f'[FEATURES] k-mer profiles k=4..10 ready ({time.time() - t0:.0f} s)')
    feats = dict(km)
    for name in S.ENCODERS:
        f = S.ROOT / 'encodemodels' / name / 'features' / (name + '.npz')
        P.need(f.is_file(), 'Missing 256-strain encoder features: ' + str(f))
        feats[name] = engine.load_npz_matrix(f, records)          # validates Strain_ID + genome SHA-256
    log('[FEATURES]', {k: v.shape[1] for k, v in feats.items()})
    return feats


def prepare(out, rebuild_folds=False):
    out.mkdir(parents=True, exist_ok=True)
    records, cohort = P.load_cohort(S.ROOT, out)
    groups = P.load_groups(records, S.ANI_GROUPS)
    fold_of, folds, prov = P.balanced_folds(records, groups, cohort, S, rebuild=rebuild_folds)
    shutil.copy2(S.BALANCED_FOLDS, out / '00_Frozen_ANI995_Folds.csv')
    shutil.copy2(S.BALANCED_PROVENANCE, out / '00_Fold_Provenance.json')
    return records, cohort, groups, fold_of, folds, prov


def fold_reports(records, groups, fold_of, out):
    v1_of, _ = P.v1_folds(records, groups, S)
    rep = out / 'cv_reevaluation'
    per, summ = R.fold_balance(records, groups, {V1_SCHEME: v1_of, V2_SCHEME: fold_of}, rep)
    for _, r in summ.iterrows():
        log(f"[FOLD BALANCE] {r.Scheme:24s} {r.Task}: strains {r.Strains_per_fold} | groups {r.Groups_per_fold} | "
            f"tolerant fraction {r.Tolerant_fraction_per_fold}")
    return per, summ


def config_jobs(configs, clfs=None):
    jobs = [(r['Task'], r['Route'], tuple(r['Upstream'].split('+')), r['Middle'], r['Downstream']) for r in configs]
    if clfs is not None:
        jobs = [j for j in jobs if j[4] in clfs]
    return jobs


def smoke_subset(configs, ready):
    """Small, deterministic cross-section of the grid: every route x task, several fusions/classifiers."""
    pick, seen = [], set()
    for r in configs:
        if r['Downstream'] not in ready:
            continue
        key = (r['Task'], r['Route'], r['Middle'])
        if key in seen:
            continue
        if r['Route'] == 'B_KMER' and len(r['Upstream'].split('+')) > 2:
            continue
        seen.add(key)
        pick.append(r)
    extra = [r for r in configs if r['Downstream'] in ready and r['Route'] == 'B_KMER' and r['Middle'] == 'block128_z'
             and r['Upstream'] in ('4mer', '5mer', '6mer')]
    rows = {(r['Task'], r['Route'], r['Upstream'], r['Middle'], r['Downstream']): r for r in pick + extra}
    return list(rows.values())


# ------------------------------------------------------------------------------------------------
def stability_analysis(ranked, results_by_id, records, groups, cohort, features, out, workers, smoke):
    """Re-score the union of v2 Top10 and v1 Top10 (per task) on 20 alternative balanced partitions and the v1 folds."""
    sdir = out / 'stability'
    parts = P.stability_partitions(records, groups, cohort, S, rebuild=False)
    if smoke:
        parts = dict(list(parts.items())[:2])
    _, v1f = P.v1_folds(records, groups, S)
    parts = dict(parts, v1_original=(None, v1f))
    v1 = pd.read_csv(S.V1_METRICS, encoding='utf-8-sig')
    v1 = v1.sort_values(['Task', 'Selection_Score', 'Combination_ID'], ascending=[True, False, True])
    v1['Rank'] = v1.groupby('Task').cumcount() + 1
    idkey = lambda r: (r['Task'], r['Route'], r['Upstream'], r['Middle'], r['Downstream'])
    v1_top = {idkey(r) for _, r in v1[v1.Rank <= 10].iterrows()}
    v1_score = {idkey(r): float(r['Selection_Score']) for _, r in v1.iterrows()}
    by_ident = {idkey(r): r for _, r in ranked.iterrows()}
    models = []
    for task in BF.TASKS:
        top = ranked[(ranked.Task == task) & (ranked.Rank <= 10)]
        ids = [idkey(r) for _, r in top.iterrows()] + [k for k in sorted(v1_top) if k[0] == task and k in by_ident]
        models += list(dict.fromkeys(ids))
    v2_top = {idkey(r) for _, r in ranked[ranked.Rank <= 10].iterrows()}
    log(f'[STABILITY] {len(models)} models (v2 Top10 + v1 Top10) x {len(parts)} partitions')

    def row(ident, pname, pooled, per):
        r = by_ident[ident]
        roc = [x['ROC_AUC'] for x in per]
        return dict(Task=ident[0], Partition=pname, Combination_ID=r['Combination_ID'], Route=ident[1],
                    Upstream=ident[2], Middle=ident[3], Downstream=ident[4], Main_Rank=int(r['Rank']),
                    In_v2_Top10=ident in v2_top, In_v1_Top10=ident in v1_top,
                    Published_v1_Selection_Score=v1_score.get(ident, np.nan),
                    **{k: pooled[k] for k in P.METRIC_KEYS},
                    Fold_SD_ROC_AUC=float(np.nanstd(roc, ddof=1)),
                    Fold_SD_PR_AUC=float(np.nanstd([x['PR_AUC'] for x in per], ddof=1)),
                    Fold_SD_MCC=float(np.nanstd([x['MCC'] for x in per], ddof=1)),
                    Fold_min_ROC_AUC=float(np.nanmin(roc)), Fold_max_ROC_AUC=float(np.nanmax(roc)))

    rows = []
    for ident in models:
        res = results_by_id[by_ident[ident]['Combination_ID']]
        pooled, per = P.pooled_and_fold_metrics(res, _MAIN['folds'], records)
        rows.append(row(ident, 'v2_main', pooled, per))
    for pname, (fo, pf) in parts.items():
        pout = sdir / pname
        fdig = engine.digest([pname, {t: [sorted(int(i) for i in te) for _, te in pf[t][1]] for t in pf}])
        meta = dict(cohort, folds_sha256=fdig, ANI_method=ANI_METHOD + '_' + pname)
        jobs = [(t, rt, tuple(u.split('+')), m, c) for (t, rt, u, m, c) in models]
        res, fails = P.run_jobs(jobs, records, pf, features, groups, meta, pout, _MAIN['workers'], f'stability {pname}',
                                S.MAX_TASKS_PER_CHILD)
        P.need(not fails, f'{len(fails)} stability jobs failed on {pname}')
        for x in res:
            ident = (x['task'], x['route'], '+'.join(x['up']), x['mid'], x['clf'])
            pooled, per = P.pooled_and_fold_metrics(x, pf, records)
            rows.append(row(ident, pname, pooled, per))
    d = pd.DataFrame(rows)
    d['Rank_within_set'] = (d.sort_values(['Selection_Score', 'Main_Rank'], ascending=[False, True])
                             .groupby(['Task', 'Partition']).cumcount() + 1).reindex(d.index)
    return R.stability(None, d.to_dict('records'), out / 'cv_reevaluation')


_MAIN = {}


def extra_route_c(ranked, configs, records, folds, features, groups, meta, out, workers, ready):
    """Route C candidates re-derived from the v2 Route A/B Top5 (as in v1). Reported separately; the frozen
    16,780-configuration grid (acid 8,280 / bile 8,500) is never modified."""
    xdir = out / 'extra_exploration_routeC'
    frozen = {(r['Task'], r['Route'], r['Upstream'], r['Middle'], r['Downstream']) for r in configs}
    cand, jobs = [], []
    for task in BF.TASKS:
        a = ranked[(ranked.Task == task) & (ranked.Route == 'A_ENCODER')].head(5)
        b = ranked[(ranked.Task == task) & (ranked.Route == 'B_KMER')].head(5)
        ups = list(dict.fromkeys(tuple(dict.fromkeys(x.split('+') + y.split('+'))) for x in a.Upstream for y in b.Upstream))
        for up in ups:
            for m in ('block128_z', 'block256_z'):
                for c in ready:
                    ident = (task, 'C_CROSS', '+'.join(up), m, c)
                    new = ident not in frozen
                    cand.append(dict(Task=task, Upstream='+'.join(up), Middle=m, Downstream=c, In_frozen_grid=not new))
                    if new:
                        jobs.append((task, 'C_CROSS', up, m, c))
    cdf = pd.DataFrame(cand)
    summary = cdf.groupby('Task').agg(Candidates=('In_frozen_grid', 'size'), Already_in_frozen_grid=('In_frozen_grid', 'sum')).reset_index()
    log('[EXTRA ROUTE C]', summary.to_dict('records'), '| new configurations to explore:', len(jobs))
    xdir.mkdir(parents=True, exist_ok=True)
    R._csv(cdf, xdir / 'v2_RouteC_candidates.csv')
    rows = []
    if jobs:
        res, fails = P.run_jobs(jobs, records, folds, features, groups, meta, xdir, workers, 'extra Route C',
                                S.MAX_TASKS_PER_CHILD)
        for x in res:
            pooled, _ = P.pooled_and_fold_metrics(x, folds, records)
            rows.append(dict(Task=x['task'], Route='C_CROSS_EXTRA', Combination_ID=x['id'], Upstream='+'.join(x['up']),
                             Middle=x['mid'], Downstream=x['clf'], **{k: pooled[k] for k in P.METRIC_KEYS}))
    d = pd.DataFrame(rows)
    if len(d):
        best = ranked.groupby('Task').Selection_Score.max()
        d['Grid_Top1_Score'] = d.Task.map(best)
        d['Would_rank_above_grid_Top1'] = d.Selection_Score > d.Grid_Top1_Score
        d = d.sort_values(['Task', 'Selection_Score'], ascending=[True, False])
    R._csv(d, xdir / 'v2_RouteC_extra_exploration_metrics.csv')
    (xdir / 'README.md').write_text(
        '# Extra exploration (not part of the 16,780-configuration grid)\n\n'
        'Route C candidates were re-derived from the v2 Route A and Route B Top5 (same rule as v1). Combinations '
        'already present in the frozen grid are listed but not re-run; only new combinations were evaluated, on the '
        'same frozen v2 folds. These results are reported separately and are never merged into Data S1-S9, '
        'Tables S2-S5 or the deployment ranking.\n', encoding='utf-8')
    return summary, d


# ------------------------------------------------------------------------------------------------
def final_refit(ranked, records, features, prov, dest, top_n):
    from model import FittedFusion
    import joblib
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in BF.TASKS:
        top = ranked[(ranked.Task == task)].head(top_n)
        P.need(len(top) == top_n, 'Missing Top models for ' + task)
        idx = np.array([i for i, r in enumerate(records) if r[task] is not None], dtype=int)
        y = np.array([records[i][task] for i in idx], dtype=int)
        tdir = dest / task; tdir.mkdir(parents=True, exist_ok=True)
        for _, r in top.iterrows():
            rank = int(r.Rank); up = r.Upstream.split('+')
            fus = FittedFusion(up, r.Middle)
            X = fus.fit({u: features[u][idx] for u in up}, y)
            ref, _ = engine.transform([features[u] for u in up], idx, idx, y, r.Middle)
            P.need(X.shape == ref.shape and np.allclose(X, ref, atol=2e-5, rtol=2e-5),
                   'Deployable preprocessing differs from the OOF engine: ' + r.Combination_ID)
            clf = engine.classifier(r.Downstream)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                clf.fit(X, y)
            art = {'task': task, 'rank': rank, 'combination_id': r.Combination_ID, 'route': r.Route,
                   'upstream': up, 'fusion': r.Middle, 'classifier_name': r.Downstream,
                   'fusion_pipeline': fus, 'classifier': clf, 'threshold': S.PREDICTION_THRESHOLD,
                   'cohort_digest': engine.digest([(z['id'], z['genome_sha']) for z in records]),
                   'n_labeled': int(len(y)), 'oof_score': float(r.Selection_Score), 'oof_pr_auc': float(r.PR_AUC),
                   'oof_roc_auc': float(r.ROC_AUC), 'oof_mcc': float(r.MCC),
                   'cv_scheme': prov['scheme'], 'folds_sha256': prov['folds_sha256'], 'pipeline_version': P.VERSION,
                   'windows': S.WINDOWS, 'window_bp': S.WINDOW_BP, 'kmer_bins': S.KMER_BINS,
                   'note': 'Refit on all labeled strains after selection by v2 balanced-ANI five-fold OOF; '
                           'the OOF metrics are internal model-selection estimates, not external validation.'}
            f = tdir / f'Top{rank:02d}.joblib'; tmp = f.with_suffix('.joblib.tmp')
            joblib.dump(art, tmp, compress=3); os.replace(tmp, f)
            manifest.append({'Task': task, 'Rank': rank, 'Model': str(f.relative_to(S.ROOT)), 'Combination_ID': r.Combination_ID,
                             'Route': r.Route, 'Upstream': r.Upstream, 'Middle': r.Middle, 'Downstream': r.Downstream,
                             'OOF_Score': r.Selection_Score, 'OOF_PR_AUC': r.PR_AUC, 'OOF_ROC_AUC': r.ROC_AUC, 'OOF_MCC': r.MCC})
            log('[SAVED]', task, f'Top{rank:02d}', r.Upstream, r.Middle, r.Downstream, round(float(r.Selection_Score), 4))
    engine.write_csv(dest / 'model_manifest.csv', manifest, list(manifest[0]))
    engine.atomic_json(dest / 'refit_provenance.json', {
        'pipeline_version': P.VERSION, 'cv_scheme': prov['scheme'], 'folds_sha256': prov['folds_sha256'],
        'selection': 'Top%d per task by pooled v2 OOF Selection_Score (0.6*PR_AUC+0.4*MCC); ties by Combination_ID' % top_n,
        'refit': 'all labeled strains of the task', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'v1_models_archived_in': str(S.ARCHIVE_V1.relative_to(S.ROOT) / 'save_models_full')})
    (dest / 'README.md').write_text(
        'Deployable classifiers refitted on all labeled strains (v2 balanced ANI-group cross-validation).\n'
        'Top01 = highest pooled v2 OOF selection score per task. The v1 models are archived separately in\n'
        'archive/v1_StratifiedGroupKFold_20261114/save_models_full and are never mixed with these files.\n'
        'Predictions for new genomes have no independent outcome validation.\n', encoding='utf-8')


# ------------------------------------------------------------------------------------------------
def acquire_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
        fh = open(out / '.run.lock', 'a+')
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another run.py is already using ' + str(out))
        return fh
    except ImportError:
        return None


def execute(args):
    t_start = time.time()
    smoke = bool(getattr(args, 'smoke', False))
    out = (S.ROOT / 'results' / 'smoke') if smoke else S.RESULTS
    lock = acquire_lock(out)
    try:
        workers = P.resolve_workers(args.workers or S.WORKERS)
        log(f'[RESOURCES] granted CPUs {P.granted_cpus()} -> {workers} single-threaded worker processes'
            f'{" (smoke test)" if smoke else ""}')
        records, cohort, groups, fold_of, folds, prov = prepare(out, rebuild_folds=args.rebuild_folds)
        fold_per, fold_sum = fold_reports(records, groups, fold_of, out)
        if args.rebuild_folds or not S.STABILITY_FOLDS.exists():
            P.stability_partitions(records, groups, cohort, S, rebuild=True)
        if args.folds_only:
            log('[FOLDS ONLY] frozen folds and fold-balance reports written to', out / 'cv_reevaluation')
            return
        configs = P.load_configs(S.CONFIGS)
        ready, missing = engine.available_clfs()
        if missing and not smoke:
            raise RuntimeError('Missing classifier packages (run: bash setup_server.sh): ' + ', '.join(missing))
        if smoke:
            configs = smoke_subset(configs, ready)
            log(f'[SMOKE] {len(configs)} configurations; classifiers available here: {ready}')
        features = load_features(records)
        P.set_engine_identity(features)
        meta = dict(cohort, folds_sha256=prov['folds_sha256'], ANI_method=ANI_METHOD)
        engine.atomic_json(out / '00_Protocol.json', {
            'version': P.VERSION, 'configurations': len(configs), 'fold_scheme': prov['scheme'],
            'folds_sha256': prov['folds_sha256'], 'ani_groups_sha256': prov['ani_groups_sha256'],
            'genome_digest': cohort['genome_digest'], 'labels_digest': prov['labels_digest'],
            'classifier_seed': engine.SEED, 'classifiers': list(engine.CLASSIFIERS), 'fusions': list(engine.MIDDLES),
            'encoders': list(S.ENCODERS), 'kmer': {'k': '4-10', 'windows': S.WINDOWS, 'window_bp': S.WINDOW_BP, 'bins': S.KMER_BINS},
            'threshold': S.PREDICTION_THRESHOLD, 'score': '0.6*PR_AUC+0.4*MCC',
            'fold_local_fitting': 'feature selection, scaling and the classifier are fitted on the training folds only',
            'route_c': 'frozen v1 Route C candidate set (acid 380 / bile 600); v2-derived candidates are extra exploration only',
            'workers': workers, 'python': sys.version.split()[0], 'external_test': False})
        jobs = config_jobs(configs)
        _MAIN.update(folds=folds, workers=workers)
        results, failures = P.run_jobs(jobs, records, folds, features, groups, meta, out, workers,
                                       'v2 OOF grid', S.MAX_TASKS_PER_CHILD)
        if failures and not smoke:
            raise RuntimeError(f'{len(failures)} configurations failed (see {out / "99_Model_Failures.csv"}); '
                               're-run python run.py to resume')
        P.need(len({r['id'] for r in results}) == len(jobs), 'Incomplete model grid')
        log('[PACKAGE] writing per-strain OOF predictions and metrics')
        availability = {n: {'status': 'READY'} for n in S.ENCODERS}
        sum_rows = engine.package_results(results, records, folds, groups, meta, out,
                                          missing if smoke else {}, availability, failures)
        ranked = R.rank_table(sum_rows)
        results_by_id = {r['id']: r for r in results}
        post_process(ranked, results_by_id, records, groups, cohort, fold_of, folds, prov, features, meta, out,
                     workers, smoke, configs, ready, fold_per, fold_sum, args)
        log(f'[FINISHED] total wall time {(time.time() - t_start) / 3600:.2f} h')
    except Exception:
        import traceback
        (out / '99_FATAL_ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if lock:
            lock.close()


def post_process(ranked, results_by_id, records, groups, cohort, fold_of, folds, prov, features, meta, out, workers,
                 smoke, configs, ready, fold_per, fold_sum, args):
    from core import supplementary_v2 as SUP
    rep = out / 'cv_reevaluation'
    R._csv(ranked, rep / '00_v2_Ranked_All_Configurations.csv')
    top_pf = R.top10_per_fold(ranked, rep, out / '02_Per_Fold_Metrics.csv', fold_per)
    change, change_sum = R.performance_change(ranked, S.V1_METRICS, rep, strict=not smoke)
    rank_x, rank_sum = R.top10_rank_change(change, rep)
    tax, per_sp = R.taxonomy_check(ranked, records, folds, results_by_id, rep)
    comp = R.component_summary(ranked, rep)
    _MAIN.update(folds=folds, workers=workers)
    stab_sum = agreement = None
    if not args.skip_stability:
        stab_sum, agreement = stability_analysis(ranked, results_by_id, records, groups, cohort, features, out,
                                                 workers, smoke)
    extra_sum = extra = None
    if not args.skip_extra_routec and not smoke:
        extra_sum, extra = extra_route_c(ranked, configs, records, folds, features, groups, meta, out, workers, ready)
    R.markdown_report(rep, fold_sum, change_sum, rank_sum, agreement, prov, extra_sum)
    dest = S.ROOT / 'save_models' / ('smoke' if smoke else 'full')
    final_refit(ranked, records, features, prov, dest, min(S.SAVE_TOP_PER_TASK, 3) if smoke else S.SAVE_TOP_PER_TASK)
    supp = (out / 'Supplementary_Material') if smoke else S.SUPPLEMENT_OUT
    SUP.build(dict(out=out, rep=rep, supp=supp, records=records, groups=groups, fold_of=fold_of, ranked=ranked,
                   fold_per=fold_per, fold_sum=fold_sum, top_pf=top_pf, change=change, change_sum=change_sum,
                   rank_x=rank_x, rank_sum=rank_sum, tax=tax, per_sp=per_sp, comp=comp, stab_sum=stab_sum,
                   agreement=agreement, extra=extra, prov=prov, smoke=smoke))


def refit_only(args):
    """Refit deployable Top10 and re-export the supplement from an existing, complete v2 result directory."""
    out = S.RESULTS
    P.need((out / '03_Pooled_OOF_All_Metrics.csv').is_file(), 'No v2 results found; run python run.py first')
    records, cohort, groups, fold_of, folds, prov = prepare(out)
    df = pd.read_csv(out / '03_Pooled_OOF_All_Metrics.csv', encoding='utf-8-sig')
    ranked = R.rank_table(df.to_dict('records'))
    features = load_features(records)
    final_refit(ranked, records, features, prov, S.ROOT / 'save_models' / 'full', S.SAVE_TOP_PER_TASK)
