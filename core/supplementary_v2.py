"""Export the Supplementary Material (v2 balanced ANI-group cross-validation).

Same file types and naming as the previous release:
  Data_S1/S2  all configurations with pooled OOF metrics (CSV, one per task, ranked)
  Data_S3-S8  strain-level OOF predictions per task x route (CSV)
  Supplementary_Tables_S1-Sxx.xlsx  README sheet + one sheet per table
  README.md, MANIFEST.json
New in v2: Data_S9 (per-fold metrics and confusion matrices of every configuration), Data_S10 (extra Route C
exploration, reported separately) and Tables S15-S20 (fold design, per-fold Top10, v1-vs-v2 change, Top10 rank
change, split stability).

Tables S6-S14 (pangenome and enrichment analyses) do not depend on the folds and are copied unchanged from
supplement_template/.
"""
from __future__ import annotations
import csv, hashlib, json, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

import settings as S
from core import pipeline_v2 as P

CJK = re.compile(r'[　-〿㐀-䶿一-鿿＀-￯]')
TASK_NAME = {'acid': 'Acid', 'bile': 'Bile'}
ROUTE_FILE = {'A_ENCODER': 'RouteA_encoder', 'B_KMER': 'RouteB_kmer', 'C_CROSS': 'RouteC_cross'}
ROUTE_TEXT = {'A_ENCODER': 'Route A (pretrained encoder embeddings)', 'B_KMER': 'Route B (multiscale k-mer profiles)',
              'C_CROSS': 'Route C (cross-route combinations)'}
PERF_COLS = {'Upstream': 'Upstream_features', 'Middle': 'Fusion', 'Downstream': 'Classifier', 'Selection_Score': 'Score',
             'PR_AUC': 'PR-AUC', 'ROC_AUC': 'ROC-AUC', 'Balanced_Accuracy': 'Balanced-Acc'}
NICE = {'Upstream': 'Upstream features', 'Middle': 'Fusion', 'Downstream': 'Classifier', 'Selection_Score': 'Score',
        'PR_AUC': 'PR-AUC', 'ROC_AUC': 'ROC-AUC', 'Balanced_Accuracy': 'Balanced accuracy'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def scheme_words(c):
    """Manuscript wording: v2 -> current fold assignment, v1 -> previous fold assignment."""
    for a, b in (('main_vs_v1_folds', 'current vs previous folds'), ('main_vs_alternatives', 'current vs alternatives'),
                 ('Top1_main_is', 'Current Top1 is'), ('v1_folds_rerun_vs_published', 'rerun on previous folds vs published'),
                 ('_v1_published', ' published (previous folds)'), ('In_v2_Top10', 'In current Top10'),
                 ('In_v1_Top10', 'In previous Top10'), ('_v1_folds', ' (previous folds)'),
                 ('_v2_median', ' current median'), ('_v1_median', ' previous median'), ('_rank_in_v2', ' rank (current)'),
                 ('_rank_in_v1', ' rank (previous)'), ('Rank_v2', 'Rank (current)'), ('Rank_v1', 'Rank (previous)'),
                 ('Combination_ID_v2', 'Combination ID (current)'), ('Combination_ID_v1', 'Combination ID (previous)'),
                 ('v1_Top1', 'Previous Top1'), ('v2_Top1', 'Current Top1'), ('Spearman_v1_v2', 'Spearman previous vs current'),
                 ('v1_median', 'Previous median'), ('v2_median', 'Current median'), ('v1_max', 'Previous max'),
                 ('v2_max', 'Current max'), ('Delta_Rank', 'Rank change'), ('Delta_', 'Change '), ('_v2', ' (current)'),
                 ('_v1', ' (previous)'), ('Main_Rank', 'Rank (current)'), ('_main', ' (current partition)'),
                 ('Within_partition_', 'Within-partition '), ('Selection_Score', 'Score'), ('PR_AUC', 'PR-AUC'),
                 ('ROC_AUC', 'ROC-AUC'), ('Balanced_Accuracy', 'Balanced accuracy')):
        c = c.replace(a, b)
    return c


def nice(df):
    out = df.copy()
    ren = {}
    for c in out.columns:
        n = NICE.get(c, c)
        n = scheme_words(n) if n == c else n
        for a, b in (('_PR_AUC', ' PR-AUC'), ('_ROC_AUC', ' ROC-AUC'), ('PR_AUC_', 'PR-AUC '), ('ROC_AUC_', 'ROC-AUC '),
                     ('Selection_Score', 'Score')):
            n = n.replace(a, b)
        ren[c] = n.replace('_', ' ')
    return out.rename(columns=ren)


# ------------------------------------------------------------------------------------------------
def data_performance(ranked, supp):
    files = []
    for i, task in enumerate(('acid', 'bile'), 1):
        d = ranked[ranked.Task == task].sort_values('Rank')
        d = d[['Rank', 'Upstream', 'Middle', 'Downstream', 'Selection_Score', 'PR_AUC', 'ROC_AUC', 'MCC', 'F1',
               'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'Route', 'Combination_ID']].rename(columns=PERF_COLS)
        f = supp / f'Data_S{i}_{TASK_NAME[task]}_all_{len(d)}_models_OOF_performance.csv'
        d.to_csv(f, index=False, encoding='utf-8')
        files.append((f, len(d), f'All {len(d):,} {"acid" if task == "acid" else "bile-salt"}-tolerance configurations: '
                                 'specification and pooled OOF metrics, ranked by selection score'))
    return files


def data_predictions(out, supp):
    """Split 01_All_Models_All_Folds_All_Strains_OOF.csv into Data S3-S8 (values passed through as text)."""
    src = out / '01_All_Models_All_Folds_All_Strains_OOF.csv'
    order = [(t, r) for t in ('acid', 'bile') for r in ('A_ENCODER', 'B_KMER', 'C_CROSS')]
    names = {k: supp / f'Data_S{3 + n}_{TASK_NAME[k[0]]}_{ROUTE_FILE[k[1]]}_OOF_predictions.csv' for n, k in enumerate(order)}
    handles, writers, counts = {}, {}, {k: 0 for k in order}
    try:
        with src.open(encoding='utf-8-sig', newline='') as f:
            rd = csv.reader(f)
            head = next(rd)
            ix = {c: i for i, c in enumerate(head)}
            new_head = [{'Upstream': 'Upstream_features', 'Middle': 'Fusion', 'Downstream': 'Classifier'}.get(c, c) for c in head]
            ti, ri, pi = ix['Task'], ix['Route'], ix['OOF_Protocol']
            for k, p in names.items():
                handles[k] = p.open('w', encoding='utf-8', newline='')
                writers[k] = csv.writer(handles[k]); writers[k].writerow(new_head)
            for row in rd:
                k = (row[ti], row[ri])
                row[pi] = P.OOF_PROTOCOL_TEXT
                writers[k].writerow(row); counts[k] += 1
    finally:
        for h in handles.values():
            h.close()
    files = []
    for k in order:
        files.append((names[k], counts[k], f'Strain-level OOF predictions, {k[0]} task, {ROUTE_TEXT[k[1]]}'))
    return files


def data_per_fold(out, supp):
    d = pd.read_csv(out / '02_Per_Fold_Metrics.csv', encoding='utf-8-sig')
    d = d[['Task', 'Route', 'Combination_ID', 'Upstream', 'Middle', 'Downstream', 'Fold', 'N', 'Positive', 'Negative',
           'ROC_AUC', 'PR_AUC', 'MCC', 'Balanced_Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity',
           'TP', 'TN', 'FP', 'FN', 'Selection_Score']].rename(columns=PERF_COLS)
    d = d.sort_values(['Task', 'Route', 'Combination_ID', 'Fold'])
    f = supp / 'Data_S9_All_models_per_fold_metrics_and_confusion.csv'
    d.to_csv(f, index=False, encoding='utf-8')
    return [(f, len(d), 'Per-fold metrics and confusion matrices (TP, TN, FP, FN) of every configuration in each of the five validation folds')]


def data_extra(extra, supp):
    if extra is None or not len(extra):
        return []
    d = extra.rename(columns=PERF_COLS)
    f = supp / 'Data_S10_Extra_RouteC_exploration_not_in_grid.csv'
    d.to_csv(f, index=False, encoding='utf-8')
    return [(f, len(d), 'Extra exploration only: Route C combinations re-derived from the v2 Route A/B Top5 that are not in the '
                        'frozen grid (reported separately; excluded from Data S1-S9 and from model selection)')]


# ------------------------------------------------------------------------------------------------
def tables(ctx):
    tpl = pd.read_excel(S.SUPPLEMENT_TEMPLATE, sheet_name=None)
    records, fold_of, ranked = ctx['records'], ctx['fold_of'], ctx['ranked']
    # S1: cohort + v2 fold of each labeled strain
    s1 = tpl['Table_S1_Cohort'].copy()
    sp = {r['id']: r['species'] for r in records}
    s1['Species'] = s1['Strain ID'].map(sp).fillna(s1['Species'])
    pos = list(s1.columns).index('ANI >= 99.5% group') + 1
    s1.insert(pos, 'Acid CV fold', s1['Strain ID'].map(fold_of['acid']).astype('Int64'))
    s1.insert(pos + 1, 'Bile-salt CV fold', s1['Strain ID'].map(fold_of['bile']).astype('Int64'))
    # S2: Top100 per task
    s2 = ranked[ranked.Rank <= 100][['Task', 'Rank', 'Route', 'Upstream', 'Middle', 'Downstream', 'Selection_Score', 'PR_AUC',
                                     'ROC_AUC', 'MCC', 'F1', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity', 'Accuracy']]
    s2 = nice(s2.sort_values(['Task', 'Rank']))
    # S3: component summary
    c = ctx['comp'].rename(columns={'Score_median': 'Score median', 'Score_Q1': 'Score Q1', 'Score_Q3': 'Score Q3',
                                    'Score_max': 'Score max', 'PR_AUC_median': 'PR-AUC median', 'ROC_AUC_median': 'ROC-AUC median',
                                    'ROC_AUC_max': 'ROC-AUC max', 'MCC_median': 'MCC median', 'F1_median': 'F1 median',
                                    'Balanced_Accuracy_median': 'Balanced accuracy median'})
    # S4/S5: taxonomy-aware checks
    s4 = ctx['tax'].rename(columns={'ROC_AUC': 'ROC-AUC', 'PR_AUC': 'PR-AUC', 'Selection_Score': 'Score',
                                    'Within_species_ROC_AUC': 'Within-species ROC-AUC', 'Within_species_pairs': 'Within-species pairs'})
    s5 = ctx['per_sp'].rename(columns={'Non_tolerant': 'Non-tolerant', 'Mean_score_tolerant': 'Mean score (tolerant)',
                                       'Mean_score_non_tolerant': 'Mean score (non-tolerant)', 'ROC_AUC': 'ROC-AUC'})
    T = [('Table_S1_Cohort', s1, 'Cohort of 256 genomes: species, B. longum subspecies, strain and phenotype source (in-house; NCBI; '
                                 'literature), tolerance labels, ANI >= 99.5% group, validation fold of each labeled strain in the '
                                 'balanced ANI-group five-fold cross-validation, assembly statistics and QC flag, median ANI to own '
                                 'species, and NCBI accessions of the 45 NCBI genomes'),
         ('Table_S2_Top100_models', s2, 'Top 100 configurations per task ranked by the selection score (first 100 rows of '
                                        'Supplementary Data S1 and S2)'),
         ('Table_S3_Component_summary', c, 'OOF performance summarized by route, fusion, classifier, single encoder, number of '
                                           'encoders, single k-mer scale and number of k-mer scales'),
         ('Table_S4_Taxonomy_check', s4, 'Pooled and within-species ROC-AUC of the ten best models of each task and the '
                                         'fold-local species-rate baseline'),
         ('Table_S5_Top1_by_species', s5, 'Per-species OOF performance of the Top1 model of each task')]
    keep = ['Table_S6_Pangenome_summary', 'Table_S7_Population_structure', 'Table_S8_Profiles_CMH', 'Table_S9_Candidate_modules',
            'Table_S10_Top5_evidence', 'Table_S11_Top5_members', 'Table_S12_CMH_sensitivity', 'Table_S13_Enriched_terms',
            'Table_S14_Direction_enrichment']
    desc = dict(zip(tpl['README'].Sheet, tpl['README'].Content))
    T += [(k, tpl[k], desc[k]) for k in keep]
    # New v2 tables
    fp = ctx['fold_per'].copy()
    fp['Scheme'] = fp.Scheme.map({'v1_StratifiedGroupKFold': 'Previous: StratifiedGroupKFold (seed 20261114)',
                                  'v2_balanced': 'Current: balanced ANI-group MILP'})
    fp = fp.rename(columns={'Training_strains': 'Training strains', 'Training_tolerant': 'Training tolerant',
                            'ANI_groups': 'ANI groups', 'Tolerant_fraction': 'Tolerant fraction', 'Target_strains': 'Target strains',
                            'Target_groups': 'Target ANI groups', 'Target_tolerant_fraction': 'Target tolerant fraction',
                            'Non_tolerant': 'Non-tolerant', 'Species_composition': 'Species composition'})
    T.append(('Table_S15_CV_fold_design', fp, 'Composition of the five validation folds (strains, ANI >= 99.5% groups, class '
                                             'balance, training-fold size and species composition) under the current balanced '
                                             'ANI-group assignment and the previous StratifiedGroupKFold assignment'))
    T.append(('Table_S16_Top10_per_fold', nice(ctx['top_pf']), 'Per-fold metrics, confusion matrices and species composition of '
                                                              'the validation folds for the Top10 models of each task'))
    ch = ctx['change_sum'].copy()
    T.append(('Table_S17_Fold_scheme_change', nice(ch), 'Change of pooled OOF metrics of all configurations between the previous and '
                                                 'the current fold assignment, by task and route (identical configurations, '
                                                 'encoders, k-mers, fusions, classifiers and hyper-parameters)'))
    rx = ctx['rank_x'].copy(); rx['Set'] = rx.Set.map({'v2 Top10': 'Current Top10', 'v1 Top10': 'Previous Top10'})
    T.append(('Table_S18_Top10_rank_change', nice(rx), 'Top10 models of each task under the current and previous fold '
                                                                 'assignments with their rank under the other assignment'))
    if ctx['stab_sum'] is not None:
        st = ctx['stab_sum'].copy()
        T.append(('Table_S19_Split_stability', nice(st), 'Split stability of the Top10 models (current and previous Top10 '
                                                        'of each task) re-evaluated on 20 alternative balanced ANI-group '
                                                        'partitions: mean, SD and range of pooled metrics, rank within the '
                                                        'model set and within-partition fold SD'))
        ag = ctx['agreement'].rename(columns=lambda x: x.replace('_ROC_AUC_v2_', '_ROC_AUC_current_').replace('_ROC_AUC_v1_', '_ROC_AUC_previous_'))
        T.append(('Table_S20_Ranking_agreement', nice(ag), 'Agreement of the Top10 model ranking between the current partition, '
                                                         'the 20 alternative balanced partitions and the previous folds'))
    return T


def write_workbook(T, path):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    readme = pd.DataFrame({'Sheet': [n for n, _, _ in T], 'Content': [c for _, _, c in T]})
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        readme.to_excel(w, sheet_name='README', index=False)
        for n, d, _ in T:
            d.to_excel(w, sheet_name=n[:31], index=False)
    wb = load_workbook(path)
    thin = Side(style='thin', color='000000'); fill = PatternFill('solid', start_color='E9EEF2')
    pcols = {'Fisher P', 'BH q', 'P', 'CMH P', 'CMH BH q', 'Pooled_Fisher_P', 'Pooled_Profile_BH_FDR', 'Pooled_FDR', 'q',
             'Profile_BH_FDR', 'p'}
    for ws in wb.worksheets:
        heads = {c.column: c.value for c in ws[1]}
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name='Arial', size=9, bold=(c.row == 1))
                if isinstance(c.value, float):
                    c.number_format = '0.00E+00' if heads.get(c.column) in pcols else '0.0000'
        for c in ws[1]:
            c.fill = fill; c.alignment = Alignment(wrap_text=True, vertical='center'); c.border = Border(top=thin, bottom=thin)
        ws.freeze_panes = 'A2'
        for col in ws.columns:
            L = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, L * 0.9), 60)
    wb.save(path)
    wb = load_workbook(path, read_only=True)
    bad = [(ws.title, str(v)[:40]) for ws in wb.worksheets for row in ws.iter_rows(values_only=True) for v in row
           if isinstance(v, str) and CJK.search(v)]
    P.need(not bad, 'Non-English text found in the supplementary workbook: ' + repr(bad[:5]))


# ------------------------------------------------------------------------------------------------
def readme(ctx, T, data_files, wb_name, supp):
    prov, fs = ctx['prov'], ctx['fold_sum']
    v2 = fs[fs.Scheme == 'v2_balanced']
    fold_lines = '\n'.join(f'  - {r.Task}: strains per fold {r.Strains_per_fold}; ANI groups per fold {r.Groups_per_fold}; '
                           f'tolerant fraction per fold {r.Tolerant_fraction_per_fold}.' for _, r in v2.iterrows())
    n_tab = len(T)
    lines = ['# Supplementary Material for iLAB-G2P', '',
             'Genome-to-phenotype prediction and population-aware genomic analysis of acid and bile-salt tolerance in '
             '*Bifidobacterium*.', '', '## Contents', '', f'### {wb_name}', '',
             'One workbook with a README sheet and one sheet per table.', '',
             '| Sheet | Rows | Content |', '|---|---:|---|']
    lines += [f'| {n} | {len(d)} | {c} |' for n, d, c in T]
    lines += ['', '### Supplementary Data (CSV, UTF-8)', '', '| File | Rows | Content |', '|---|---:|---|']
    lines += [f'| {f.name} | {n:,} | {c} |' for f, n, c in data_files]
    lines += ['', '## Conventions', '',
              '- Tasks: acid tolerance (survival >= 50% after 3 h at pH 3.0 for in-house strains) and bile-salt tolerance '
              '(bile-induced growth delay LT < 4 h for in-house strains). Labels: 1 = tolerant, 0 = non-tolerant; strains '
              'without a label were not used for the corresponding task.',
              '- Sources: In-house (this study); NCBI (genomes downloaded from NCBI; accessions in Table S1); Literature '
              '(62 *B. adolescentis* strains provided by the corresponding author of Li et al., 2026, China Dairy Industry '
              '54(8):5-12, doi:10.19827/j.issn1001-2230.2026.08.001).',
              '- Cross-validation: five folds built separately for each task from the connected components of the skani '
              'ANI >= 99.5% graph (89 groups). All labeled strains of a group are assigned to the same validation fold. The '
              'assignment was obtained by lexicographic mixed-integer optimisation (scipy/HiGHS) that uses only group sizes '
              'and labels: (1) balance of strain numbers and of the tolerant fraction across folds, with at least two '
              'tolerant and two non-tolerant strains in every validation fold; (2) balance of the number of ANI groups per '
              'fold; (3) a fixed, data-independent tie-break. Largest groups cannot be split, so the fold that contains '
              'them is larger and holds fewer groups. Resulting folds:',
              fold_lines,
              f'  - Fold file SHA-256: {prov["folds_sha256"]}.',
              '- All feature selection, scaling and classifier fitting are performed within the training folds. OOF = out-of-fold.',
              '- Selection score = 0.6 x PR-AUC (average precision) + 0.4 x MCC; binary calls at threshold 0.5.',
              '- Encoders: DNABERT-S; prokbert-mini; NTv2-100M; GENERanno-0.5B; JEPA-DNA-NTv3-100M. k-mers: canonical, '
              'hashed to 4,096 bins per k (k = 4-10).',
              '- Route C: the cross-route candidate set is identical to the previously reported grid (acid 380, bile 600 '
              'configurations). Route C candidates re-derived from the current Route A/B ranking are reported separately '
              '(Data S10) and are not part of the 16,780 configurations.',
              '- Tables S15-S20 compare the current fold assignment with the previous StratifiedGroupKFold assignment. A '
              'higher metric under one assignment is not by itself evidence of a better scheme; the current assignment '
              'was chosen for group integrity and balance before any model was evaluated.',
              '- Pangenome statistics (Tables S6-S14): Panaroo families collapsed to exact presence/absence profiles; '
              "Fisher's exact test with Benjamini-Hochberg FDR; species-stratified Cochran-Mantel-Haenszel re-test. All gene "
              'associations are exploratory.',
              '- File integrity: SHA-256 checksums of all files are listed in MANIFEST.json.', '']
    (supp / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build(ctx):
    supp = Path(ctx['supp'])
    if supp.exists():
        old = supp.parent / (supp.name + '_previous')
        if old.exists():
            shutil.rmtree(old)
        supp.rename(old)
        P.log('[SUPPLEMENT] previous export moved to', old.name)
    supp.mkdir(parents=True)
    P.log('[SUPPLEMENT] writing Data S1-S10')
    files = data_performance(ctx['ranked'], supp)
    files += data_predictions(ctx['out'], supp)
    files += data_per_fold(ctx['out'], supp)
    files += data_extra(ctx['extra'], supp)
    T = tables(ctx)
    wb_name = f'Supplementary_Tables_S1-S{len(T)}.xlsx'
    P.log('[SUPPLEMENT] writing', wb_name)
    write_workbook(T, supp / wb_name)
    readme(ctx, T, files, wb_name, supp)
    for f, _, _ in files:                                         # header / text sanity
        with open(f, encoding='utf-8') as h:
            head = h.readline() + h.readline()
        P.need(not CJK.search(head), 'Non-English text in ' + f.name)
    man = {'cohort_genomes': 256, 'model_configurations': int(len(ctx['ranked'])),
           'oof_prediction_rows': int(sum(n for f, n, _ in files if 'OOF_predictions' in f.name)),
           'cv_scheme': ctx['prov']['scheme'], 'folds_sha256': ctx['prov']['folds_sha256'],
           'pipeline_version': P.VERSION, 'smoke_test': bool(ctx['smoke']), 'files': []}
    for f in sorted(supp.iterdir()):
        if f.name == 'MANIFEST.json':
            continue
        rows = next((n for g, n, _ in files if g == f), None)
        if rows is None and f.suffix == '.xlsx':
            rows = {n: int(len(d)) for n, d, _ in T}
        man['files'].append({'file': f.name, 'rows': rows, 'size_bytes': f.stat().st_size, 'sha256': sha(f)})
    (supp / 'MANIFEST.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    P.log('[SUPPLEMENT] done:', supp, f'({len(man["files"])} files)')
