"""Post-hoc reports for the v2 re-evaluation: fold balance, per-fold metrics, v1 vs v2 performance,
Top10 ranking change, split stability, taxonomy-aware checks and component summaries.

None of these reports feeds back into the fold assignment (which is frozen before any model is run).
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from core import legacy_engine as engine
from core import balanced_ani_folds as BF
from core import pipeline_v2 as P

ID = ['Task', 'Route', 'Upstream', 'Middle', 'Downstream']
M4 = ['Selection_Score', 'PR_AUC', 'ROC_AUC', 'MCC']


def _csv(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding='utf-8-sig')


def rank_table(sum_rows):
    """Pooled metrics with a deterministic rank per task (score desc, Combination_ID asc)."""
    d = pd.DataFrame(sum_rows)
    d = d.sort_values(['Task', 'Selection_Score', 'Combination_ID'], ascending=[True, False, True])
    d['Rank'] = d.groupby('Task').cumcount() + 1
    return d.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------
def fold_balance(records, groups, schemes, out):
    """schemes: {name: fold_of}. Writes per-fold composition and balance summaries."""
    per, summ = [], []
    for name, fo in schemes.items():
        for task in BF.TASKS:
            rows = BF.fold_table(records, groups, task, fo[task], name)
            per += rows
            summ.append(BF.balance_summary(rows))
    _csv(pd.DataFrame(per), out / '10_Fold_Balance_Per_Fold.csv')
    _csv(pd.DataFrame(summ), out / '10_Fold_Balance_Summary.csv')
    return pd.DataFrame(per), pd.DataFrame(summ)


def top10_per_fold(ranked, out, per_fold_csv, fold_comp):
    pf = pd.read_csv(per_fold_csv, encoding='utf-8-sig')
    top = ranked[ranked.Rank <= 10][['Task', 'Rank', 'Combination_ID']]
    x = pf.merge(top, on=['Task', 'Combination_ID'])
    comp = fold_comp[fold_comp.Scheme == 'v2_balanced'][['Task', 'Fold', 'ANI_groups', 'Species_composition']]
    x['Fold'] = x.Fold.astype(int)
    x = x.merge(comp, on=['Task', 'Fold'], how='left').sort_values(['Task', 'Rank', 'Fold'])
    cols = ['Task', 'Rank', 'Combination_ID', 'Route', 'Upstream', 'Middle', 'Downstream', 'Fold', 'N', 'Positive', 'Negative',
            'ANI_groups', 'ROC_AUC', 'PR_AUC', 'MCC', 'Balanced_Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1',
            'Specificity', 'TP', 'TN', 'FP', 'FN', 'Selection_Score', 'Species_composition']
    x = x[cols]
    _csv(x, out / '11_Top10_Per_Fold_Metrics_and_Confusion.csv')
    disp = (x.groupby(['Task', 'Rank', 'Combination_ID'])[['ROC_AUC', 'PR_AUC', 'MCC', 'Selection_Score']]
            .agg(['mean', 'std', 'min', 'max']))
    disp.columns = ['_'.join(c) for c in disp.columns]
    _csv(disp.reset_index(), out / '11_Top10_Per_Fold_Dispersion.csv')
    return x


def performance_change(ranked, v1_metrics_csv, out, strict=True):
    v1 = pd.read_csv(v1_metrics_csv, encoding='utf-8-sig')
    v1 = v1.sort_values(['Task', 'Selection_Score', 'Combination_ID'], ascending=[True, False, True])
    v1['Rank'] = v1.groupby('Task').cumcount() + 1
    keep = ID + ['Combination_ID', 'Rank'] + M4 + ['F1', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity']
    m = ranked[keep].merge(v1[keep], on=ID, suffixes=('_v2', '_v1'), how='outer' if strict else 'inner', indicator=True)
    if strict:
        P.need((m._merge == 'both').all() and len(m) == 16780, 'v1/v2 configuration identities do not match one-to-one')
    m = m.drop(columns='_merge')
    for k in M4 + ['F1', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity']:
        m[f'Delta_{k}'] = m[f'{k}_v2'] - m[f'{k}_v1']
    m['Delta_Rank'] = m['Rank_v2'] - m['Rank_v1']
    m = m.sort_values(['Task', 'Rank_v2'])
    _csv(m, out / '12_Performance_Change_v1_vs_v2_All_Configurations.csv')
    rows = []
    for (task, route), g in list(m.groupby(['Task', 'Route'])) + [((t, 'ALL'), g) for t, g in m.groupby('Task')]:
        for k in M4:
            d = g[f'Delta_{k}']
            rows.append({'Task': task, 'Route': route, 'Metric': k, 'Configurations': len(g),
                         'v1_median': g[f'{k}_v1'].median(), 'v2_median': g[f'{k}_v2'].median(),
                         'v1_max': g[f'{k}_v1'].max(), 'v2_max': g[f'{k}_v2'].max(),
                         'Delta_median': d.median(), 'Delta_Q1': d.quantile(.25), 'Delta_Q3': d.quantile(.75),
                         'Fraction_improved': float((d > 0).mean()),
                         'Spearman_v1_v2': g[f'{k}_v1'].rank().corr(g[f'{k}_v2'].rank())})
    s = pd.DataFrame(rows)
    _csv(s, out / '12_Performance_Change_Summary.csv')
    return m, s


def top10_rank_change(change, out):
    rows = []
    for task, g in change.groupby('Task'):
        new10 = g[g.Rank_v2 <= 10].assign(Set='v2 Top10')
        old10 = g[g.Rank_v1 <= 10].assign(Set='v1 Top10')
        rows.append(pd.concat([new10, old10]))
    x = pd.concat(rows)[['Set', 'Rank_v2', 'Rank_v1', 'Delta_Rank'] + ID +
                        ['Combination_ID_v2', 'Combination_ID_v1'] + [f'{k}_{v}' for k in M4 for v in ('v2', 'v1')]]
    x = x[['Task'] + [c for c in x.columns if c != 'Task']].sort_values(['Task', 'Set', 'Rank_v2'])
    _csv(x, out / '13_Top10_Rank_Change.csv')
    summ = []
    for task, g in change.groupby('Task'):
        s2 = set(g[g.Rank_v2 <= 10].Combination_ID_v2); s1 = set(g[g.Rank_v1 <= 10].Combination_ID_v2)
        top1_v2 = g.loc[g.Rank_v2.idxmin()]; top1_v1 = g.loc[g.Rank_v1.idxmin()]
        summ.append({'Task': task, 'Top10_overlap': len(s1 & s2),
                     'Top100_overlap': len(set(g[g.Rank_v2 <= 100].Combination_ID_v2) & set(g[g.Rank_v1 <= 100].Combination_ID_v2)),
                     'v1_Top1': f"{top1_v1.Upstream}; {top1_v1.Middle}; {top1_v1.Downstream}",
                     'v1_Top1_rank_in_v2': int(top1_v1.Rank_v2),
                     'v2_Top1': f"{top1_v2.Upstream}; {top1_v2.Middle}; {top1_v2.Downstream}",
                     'v2_Top1_rank_in_v1': int(top1_v2.Rank_v1),
                     'Spearman_rank_all_configurations': g.Rank_v1.corr(g.Rank_v2, method='spearman')})
    _csv(pd.DataFrame(summ), out / '13_Top10_Rank_Change_Summary.csv')
    return x, pd.DataFrame(summ)


# ------------------------------------------------------------------------------------------------
def stability(model_rows, partition_results, out):
    """partition_results: list of dict(Task, Partition, Combination_ID(v2 main), metrics..., Fold_SD_*)."""
    d = pd.DataFrame(partition_results)
    _csv(d, out / '14_Split_Stability_Per_Partition.csv')
    bal = d[d.Partition != 'v1_original']
    rows = []
    for (task, cid), g in bal.groupby(['Task', 'Combination_ID']):
        r = {'Task': task, 'Combination_ID': cid, 'Partitions': len(g)}
        r.update({k: g[k].iloc[0] for k in ('Route', 'Upstream', 'Middle', 'Downstream', 'Main_Rank', 'In_v2_Top10', 'In_v1_Top10')})
        for k in M4:
            r[f'{k}_main'] = float(g[g.Partition == 'v2_main'][k].iloc[0])
            r[f'{k}_mean'] = g[k].mean(); r[f'{k}_sd'] = g[k].std(ddof=1)
            r[f'{k}_min'] = g[k].min(); r[f'{k}_max'] = g[k].max()
            v1 = d[(d.Task == task) & (d.Combination_ID == cid) & (d.Partition == 'v1_original')][k]
            r[f'{k}_v1_folds'] = float(v1.iloc[0]) if len(v1) else np.nan
        r['Selection_Score_v1_published'] = float(g.Published_v1_Selection_Score.iloc[0])
        r['Within_partition_fold_SD_ROC_AUC_mean'] = g.Fold_SD_ROC_AUC.mean()
        v1r = d[(d.Task == task) & (d.Combination_ID == cid) & (d.Partition == 'v1_original')]
        r['Within_partition_fold_SD_ROC_AUC_v1'] = float(v1r.Fold_SD_ROC_AUC.iloc[0]) if len(v1r) else np.nan
        r['Rank_within_set_mean'] = g.Rank_within_set.mean(); r['Rank_within_set_sd'] = g.Rank_within_set.std(ddof=1)
        r['Times_rank1_within_set'] = int((g.Rank_within_set == 1).sum())
        r['Times_top5_within_set'] = int((g.Rank_within_set <= 5).sum())
        rows.append(r)
    s = pd.DataFrame(rows).sort_values(['Task', 'Main_Rank'])
    _csv(s, out / '14_Split_Stability_Summary.csv')
    agr = []
    for task, g in d.groupby('Task'):
        parts = sorted(g.Partition.unique())
        rk = {p: dict(zip(g[g.Partition == p].Combination_ID, g[g.Partition == p].Rank_within_set)) for p in parts}
        taus = [P.kendall_tau(rk['v2_main'], rk[p]) for p in parts if p not in ('v2_main', 'v1_original')]
        agr.append({'Task': task, 'Models_in_set': g.Combination_ID.nunique(),
                    'Alternative_partitions': len(taus),
                    'Kendall_tau_main_vs_alternatives_mean': float(np.nanmean(taus)),
                    'Kendall_tau_main_vs_alternatives_min': float(np.nanmin(taus)),
                    'Kendall_tau_main_vs_v1_folds': P.kendall_tau(rk['v2_main'], rk['v1_original']),
                    'Top1_main_is_rank1_in_fraction_of_alternatives':
                        float(np.mean([min(rk[p], key=rk[p].get) == min(rk['v2_main'], key=rk['v2_main'].get)
                                       for p in parts if p not in ('v2_main', 'v1_original')])),
                    'Score_SD_across_partitions_median': float(s[s.Task == task].Selection_Score_sd.median()),
                    'ROC_AUC_SD_across_partitions_median': float(s[s.Task == task].ROC_AUC_sd.median()),
                    'Within_partition_fold_SD_ROC_AUC_v2_median': float(s[s.Task == task].Within_partition_fold_SD_ROC_AUC_mean.median()),
                    'Within_partition_fold_SD_ROC_AUC_v1_median': float(s[s.Task == task].Within_partition_fold_SD_ROC_AUC_v1.median()),
                    'Max_abs_diff_v1_folds_rerun_vs_published': float((s[s.Task == task].Selection_Score_v1_folds -
                                                                       s[s.Task == task].Selection_Score_v1_published).abs().max())})
    a = pd.DataFrame(agr)
    _csv(a, out / '14_Split_Stability_Ranking_Agreement.csv')
    return s, a


# ------------------------------------------------------------------------------------------------
def taxonomy_check(ranked, records, folds, results_by_id, out):
    """Pooled vs within-species discrimination of the Top10 models and a fold-local species-rate baseline."""
    species = np.array([r['species'] for r in records])
    tax, per_sp = [], []
    for task in BF.TASKS:
        idx, ff = folds[task]
        y = np.array([records[i][task] for i in idx], dtype=int)
        sp = species[idx]
        base = np.zeros(len(records))
        for tr, te in ff:
            rate = defaultdict(list)
            for i in tr:
                rate[species[i]].append(records[i][task])
            prev = np.mean([records[i][task] for i in tr])
            for i in te:
                base[i] = np.mean(rate[species[i]]) if species[i] in rate else prev
        b = base[idx]
        m = engine.metrics(y, b, (b >= .5).astype(int))
        w, n = P.within_species_auc(y, b, sp)
        tax.append(dict(Task=task, Rank='Baseline', Model='Species-only baseline (fold-local species tolerance rate)',
                        ROC_AUC=m['ROC_AUC'], PR_AUC=m['PR_AUC'], MCC=m['MCC'], Selection_Score=m['Selection_Score'],
                        Accuracy=m['Accuracy'], Within_species_ROC_AUC=w, Within_species_pairs=n))
        for _, r in ranked[(ranked.Task == task) & (ranked.Rank <= 10)].iterrows():
            res = results_by_id[r.Combination_ID]
            with np.load(res['path'], allow_pickle=False) as z:
                s = z['score'][idx]
            w, n = P.within_species_auc(y, s, sp)
            tax.append(dict(Task=task, Rank=int(r.Rank), Model=f'{r.Route}: {r.Upstream} | {r.Middle} | {r.Downstream}',
                            ROC_AUC=r.ROC_AUC, PR_AUC=r.PR_AUC, MCC=r.MCC, Selection_Score=r.Selection_Score,
                            Accuracy=r.Accuracy, Within_species_ROC_AUC=w, Within_species_pairs=n))
            if r.Rank == 1:
                for g in sorted(set(sp)):
                    msk = sp == g; yy = y[msk]; ss = s[msk]; pp = (ss >= .5).astype(int)
                    auc = (float(np.mean([(a > b_) + .5 * (a == b_) for a in ss[yy == 1] for b_ in ss[yy == 0]]))
                           if 0 < yy.sum() < len(yy) else np.nan)
                    per_sp.append(dict(Task=task, Species=g, n=int(msk.sum()), Tolerant=int(yy.sum()),
                                       Non_tolerant=int(len(yy) - yy.sum()), Accuracy=float((pp == yy).mean()),
                                       Mean_score_tolerant=float(ss[yy == 1].mean()) if yy.sum() else np.nan,
                                       Mean_score_non_tolerant=float(ss[yy == 0].mean()) if (yy == 0).sum() else np.nan,
                                       ROC_AUC=auc))
    t = pd.DataFrame(tax); s = pd.DataFrame(per_sp)
    _csv(t, out / '16_Taxonomy_Check_Top10.csv'); _csv(s, out / '16_Top1_Per_Species.csv')
    return t, s


def component_summary(ranked, out):
    g = ranked.copy()
    enc = set(engine.ENCODERS)
    g['n_encoders'] = g.Upstream.str.split('+').apply(lambda u: sum(x in enc for x in u))
    g['n_kmers'] = g.Upstream.str.split('+').apply(lambda u: sum(x.endswith('mer') for x in u))
    parts = []

    def summ(df, by, factor):
        h = df.groupby(['Task', by])
        r = pd.DataFrame({'Configurations': h.size(), 'Score_median': h.Selection_Score.median(),
                          'Score_Q1': h.Selection_Score.quantile(.25), 'Score_Q3': h.Selection_Score.quantile(.75),
                          'Score_max': h.Selection_Score.max(), 'PR_AUC_median': h.PR_AUC.median(),
                          'ROC_AUC_median': h.ROC_AUC.median(), 'ROC_AUC_max': h.ROC_AUC.max(),
                          'MCC_median': h.MCC.median(), 'F1_median': h.F1.median(),
                          'Balanced_Accuracy_median': h.Balanced_Accuracy.median()}).reset_index()
        r = r.rename(columns={by: 'Level'}); r.insert(1, 'Factor', factor); r['Level'] = r.Level.astype(str)
        parts.append(r)
    summ(g, 'Route', 'Route'); summ(g, 'Middle', 'Fusion'); summ(g, 'Downstream', 'Classifier')
    a = g[g.Route == 'A_ENCODER']; b = g[g.Route == 'B_KMER']
    summ(a[a.n_encoders == 1], 'Upstream', 'Single encoder (Route A)'); summ(a, 'n_encoders', 'Number of encoders (Route A)')
    summ(b[b.n_kmers == 1], 'Upstream', 'Single k-mer scale (Route B)'); summ(b, 'n_kmers', 'Number of k-mer scales (Route B)')
    x = pd.concat(parts)
    _csv(x, out / '17_Component_Summary.csv')
    return x


def markdown_report(out, bal_summary, change_summary, rank_summary, agreement, prov, extra_summary=None):
    def tbl(df, cols=None):
        df = df[cols] if cols else df
        h = '| ' + ' | '.join(df.columns) + ' |\n|' + '---|' * len(df.columns) + '\n'
        for _, r in df.iterrows():
            h += '| ' + ' | '.join((f'{v:.4f}' if isinstance(v, float) else str(v)) for v in r.values) + ' |\n'
        return h
    txt = ['# iLAB-G2P v2 - balanced ANI-group cross-validation: re-evaluation report', '',
           'Fold assignment: `' + prov['scheme'] + '` (frozen; SHA-256 `' + prov['folds_sha256'] + '`). '
           'The assignment was optimised from ANI-group sizes and labels only (no features, predictions or metrics).', '',
           '## 1. Fold balance before (v1, StratifiedGroupKFold) and after (v2, balanced MILP)', '',
           tbl(bal_summary, ['Scheme', 'Task', 'Strains_per_fold', 'Strain_range', 'Groups_per_fold', 'Group_range',
                             'Tolerant_fraction_per_fold', 'Max_abs_tolerant_fraction_deviation']), '',
           'The largest ANI groups (acid: 40 strains; bile: 53 strains, both ANI995_A6) exceed the per-fold target and '
           'cannot be split; the fold that holds such a group is necessarily larger and has fewer groups.', '',
           '## 2. Performance change of all 16,780 configurations (v2 - v1, pooled OOF)', '',
           tbl(change_summary[change_summary.Route == 'ALL'],
               ['Task', 'Metric', 'v1_median', 'v2_median', 'v1_max', 'v2_max', 'Delta_median', 'Fraction_improved', 'Spearman_v1_v2']), '',
           '## 3. Top10 ranking change', '', tbl(rank_summary), '']
    if agreement is not None:
        txt += ['## 4. Split stability of the Top10 models (v2 Top10 and v1 Top10, re-scored on 20 alternative balanced '
                'partitions and on the v1 folds)', '', tbl(agreement), '']
    if extra_summary is not None:
        txt += ['## 5. Extra exploration (reported separately)', '',
                'Route C candidates re-derived from the v2 Route A/B Top5; only combinations absent from the frozen grid '
                'were evaluated (see `extra_exploration_routeC/`). They are not part of the 16,780 configurations.', '',
                tbl(extra_summary), '']
    txt += ['## Interpretation rules', '',
            '- A higher score under v2 is not, by itself, evidence that v2 is the better scheme. v2 is preferred because '
            'it keeps every ANI group intact (no similarity leakage) while equalising fold size and class balance, which '
            'reduces the variance of the per-fold estimates.',
            '- All values are internal OOF estimates; the Top10 models are selected on the same folds and are therefore '
            'optimistically biased. They are not independent external-test results.',
            '- v1 results are archived separately in `archive/v1_StratifiedGroupKFold_20261114/` and are never mixed '
            'into the v2 tables except in the explicit v1-vs-v2 comparison files (12_*, 13_*, 14_*).']
    (out / '15_CV_Reevaluation_Report.md').write_text('\n'.join(txt), encoding='utf-8')
