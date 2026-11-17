"""Balanced, ANI-group-preserving five-fold assignment (iLAB-G2P v2).

Design (expert recommendation, 2026-11):
  * HARD    the frozen ANI >= 99.5% connected components (89 groups) are never split: every labeled strain
            of a group is placed in the same validation fold. Each task is partitioned separately.
  * HARD    every validation fold contains >= MIN_CLASS tolerant and >= MIN_CLASS non-tolerant strains
            (hence every training fold contains both classes as well).
  * STAGE 1 (primary: strain and class balance) minimise
                sum_f Q(|n_f - N/5|)  +  W_POS * sum_f Q(|p_f - pi*n_f|)
            n_f = labeled strains in validation fold f, p_f = tolerant strains in fold f,
            pi = overall tolerant fraction of the task, Q(d) = sum_{j>=0} max(0, d - j) is a convex
            piecewise-linear approximation of d(d+1)/2 (a squared-deviation criterion: large deviations
            are penalised progressively and unavoidable imbalances are spread over folds).
  * STAGE 2 (secondary: ANI-group-count balance) keep the stage-1 objective <= optimum + STAGE1_SLACK
            and minimise  sum_f Q(|k_f - G/5|)  (k_f = ANI groups in fold f).
  * STAGE 3 (tie-break) keep both objectives at their attained values and minimise a fixed,
            data-independent canonical cost, so that the returned partition is unique and reproducible.

Groups with identical (size, tolerant count) are interchangeable for every objective, so the MILP is written
over group *types* with integer counts per fold (this removes the solver symmetry created by dozens of
singleton groups); groups of one type are then dealt to folds in a canonical order (sorted group ID).

The optimisation uses only group sizes and labels. It never sees genome features, model predictions or
performance metrics, so the partition cannot be tuned towards favourable results.

Alternative near-balanced partitions for the split-stability analysis are produced by the same model with
seeded random canonical costs, relaxed tolerances and seeded random dealing of interchangeable groups.

Solver: scipy.optimize.milp (HiGHS) with fixed options.
"""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

K = 5
MIN_CLASS = 2
W_POS = 2.0
Q_BREAKS = 60            # breakpoints of the convex squared-deviation approximation
STAGE1_SLACK = 1.0      # stage 2 may give up at most one unit of the stage-1 criterion for group balance
TIME_LIMIT = 120.0
SCHEME = 'BalancedANI995_MILP_v2'
TASKS = ('acid', 'bile')


def task_groups(records, groups, task):
    """Ordered list of ANI groups with labeled strains for one task: [(group, size, tolerant)]."""
    agg = defaultdict(lambda: [0, 0])
    for i, r in enumerate(records):
        if r[task] is None:
            continue
        agg[groups[i]][0] += 1
        agg[groups[i]][1] += int(r[task])
    return sorted(((g, a[0], a[1]) for g, a in agg.items()), key=lambda z: (-z[1], -z[2], z[0]))


class _TypeModel:
    """Variables: y[t,f] (integer counts of groups of type t in fold f) | dn_f | dp_f | dk_f |
    nmax nmin kmax kmin dpmax."""

    def __init__(self, types, counts):
        self.T = T = len(types)
        self.size = np.array([t[0] for t in types], float)
        self.pos = np.array([t[1] for t in types], float)
        self.cnt = np.array(counts, float)
        N, P, G = (self.size * self.cnt).sum(), (self.pos * self.cnt).sum(), self.cnt.sum()
        self.pi = P / N
        self.ny = T * K
        self.o_dn, self.o_dp, self.o_dk, self.o_ext = self.ny, self.ny + K, self.ny + 2 * K, self.ny + 3 * K
        self.o_q = self.o_ext + 5                      # Q-excess variables: 3 measures x K folds x Q_BREAKS
        self.nv = self.o_q + 3 * K * Q_BREAKS
        rows, lo, hi = [], [], []

        def add(c, l, h):
            rows.append(c); lo.append(l); hi.append(h)
        Y = self.Y = lambda t, f: t * K + f
        for t in range(T):
            c = np.zeros(self.nv); c[[Y(t, f) for f in range(K)]] = 1; add(c, self.cnt[t], self.cnt[t])
        nt, kt = N / K, G / K
        nvec = []
        for f in range(K):
            n = np.zeros(self.nv); p = np.zeros(self.nv); k = np.zeros(self.nv)
            for t in range(T):
                n[Y(t, f)] = self.size[t]; p[Y(t, f)] = self.pos[t]; k[Y(t, f)] = 1
            nvec.append(n)
            add(p.copy(), MIN_CLASS, np.inf)
            add(n - p, MIN_CLASS, np.inf)
            c = n.copy(); c[self.o_dn + f] = -1; add(c, -np.inf, nt)
            c = -n; c[self.o_dn + f] = -1; add(c, -np.inf, -nt)
            d = p - self.pi * n
            c = d.copy(); c[self.o_dp + f] = -1; add(c, -np.inf, 0)
            c = -d; c[self.o_dp + f] = -1; add(c, -np.inf, 0)
            c = np.zeros(self.nv); c[self.o_dp + f] = 1; c[self.o_ext + 4] = -1; add(c, -np.inf, 0)
            c = k.copy(); c[self.o_dk + f] = -1; add(c, -np.inf, kt)
            c = -k; c[self.o_dk + f] = -1; add(c, -np.inf, -kt)
            c = n.copy(); c[self.o_ext + 0] = -1; add(c, -np.inf, 0)
            c = -n; c[self.o_ext + 1] = 1; add(c, -np.inf, 0)
            c = k.copy(); c[self.o_ext + 2] = -1; add(c, -np.inf, 0)
            c = -k; c[self.o_ext + 3] = 1; add(c, -np.inf, 0)
        for m_, base in enumerate((self.o_dn, self.o_dp, self.o_dk)):   # e_{m,f,j} >= d_{m,f} - j
            for f in range(K):
                for j in range(Q_BREAKS):
                    c = np.zeros(self.nv); c[base + f] = 1; c[self.q(m_, f, j)] = -1; add(c, -np.inf, j)
        for f in range(K - 1):                                   # symmetry breaking: folds ordered by size
            add(nvec[f] - nvec[f + 1], 0, np.inf)
        self.A, self.lo, self.hi = np.array(rows), np.array(lo), np.array(hi)
        self.integrality = np.r_[np.ones(self.ny), np.zeros(self.nv - self.ny)]
        ub = np.r_[np.repeat(self.cnt, K), np.full(self.nv - self.ny, np.inf)]
        self.bounds = Bounds(np.zeros(self.nv), ub)

    def q(self, m_, f, j):
        return self.o_q + (m_ * K + f) * Q_BREAKS + j

    def obj1(self):
        c = np.zeros(self.nv)
        for f in range(K):
            for j in range(Q_BREAKS):
                c[self.q(0, f, j)] = 1; c[self.q(1, f, j)] = W_POS
        return c

    def obj2(self):
        c = np.zeros(self.nv)
        for f in range(K):
            for j in range(Q_BREAKS):
                c[self.q(2, f, j)] = 1
        return c

    def canonical(self, rng=None):
        c = np.zeros(self.nv)
        for t in range(self.T):
            for f in range(K):
                c[self.Y(t, f)] = rng.random() if rng is not None else (f + 1) * (self.T - t) / (self.T * K)
        return c

    def solve(self, c, extra=()):
        A, lo, hi = self.A, self.lo, self.hi
        if extra:
            A = np.vstack([A] + [e[0][None, :] for e in extra])
            lo = np.r_[lo, [e[1] for e in extra]]; hi = np.r_[hi, [e[2] for e in extra]]
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=self.integrality, bounds=self.bounds,
                   options={'time_limit': TIME_LIMIT, 'mip_rel_gap': 0.0, 'presolve': True, 'disp': False})
        if res.x is None:
            raise RuntimeError('Balanced ANI fold MILP failed: ' + str(res.message))
        return res


def optimise_task(glist, alt_seed=None, tol1=0.0, tol2=0.0):
    """glist from task_groups(). Returns ({group: fold 1..5}, solver report)."""
    tkey = sorted({(s, p) for _, s, p in glist}, key=lambda z: (-z[0], -z[1]))
    members = {k: sorted(g for g, s, p in glist if (s, p) == k) for k in tkey}
    m = _TypeModel(tkey, [len(members[k]) for k in tkey])
    r1 = m.solve(m.obj1()); z1 = float(m.obj1() @ r1.x)
    lim1 = z1 * (1 + tol1) + STAGE1_SLACK
    r2 = m.solve(m.obj2(), extra=[(m.obj1(), -np.inf, lim1)]); z2 = float(m.obj2() @ r2.x)
    lim2 = z2 * (1 + tol2) + 1e-6
    rng = np.random.default_rng(alt_seed) if alt_seed is not None else None
    r3 = m.solve(m.canonical(rng), extra=[(m.obj1(), -np.inf, lim1), (m.obj2(), -np.inf, lim2)])
    y = np.round(r3.x[:m.ny]).astype(int).reshape(m.T, K)
    if not np.array_equal(y.sum(1), m.cnt.astype(int)):
        raise RuntimeError('Non-integral fold assignment')
    gfold = {}
    for t, k in enumerate(tkey):
        gs = list(members[k])
        if rng is not None:
            gs = [gs[i] for i in rng.permutation(len(gs))]
        slots = [f for f in range(K) for _ in range(y[t, f])]
        for g, f in zip(gs, slots):
            gfold[g] = f
    # fold labels: 1 = largest validation fold; ties broken by the first canonical group
    order_g = [g for g, _, _ in glist]
    sizes = {g: s for g, s, _ in glist}
    n = [sum(sizes[g] for g in gfold if gfold[g] == f) for f in range(K)]
    first = [min(order_g.index(g) for g in gfold if gfold[g] == f) for f in range(K)]
    remap = {old: new + 1 for new, old in enumerate(sorted(range(K), key=lambda f: (-n[f], first[f])))}
    gfold = {g: remap[f] for g, f in gfold.items()}
    rep = {'group_types': m.T, 'stage1_objective': z1, 'stage2_objective': z2,
           'stage3_objective': float(r3.fun),
           'final_stage1_value': float(m.obj1() @ r3.x), 'final_stage2_value': float(m.obj2() @ r3.x),
           'stage1_optimal': bool(r1.status == 0), 'stage2_optimal': bool(r2.status == 0),
           'stage3_optimal': bool(r3.status == 0), 'solver_messages': [r1.message, r2.message, r3.message]}
    return gfold, rep


def build_partition(records, groups, cohort_digest, alt_seed=None, tol1=0.0, tol2=0.0):
    """Return ({task: {strain_id: fold}}, fold rows, solver report)."""
    out, rows, report = {}, [], {}
    for task in TASKS:
        gl = task_groups(records, groups, task)
        gfold, rep = optimise_task(gl, alt_seed=alt_seed, tol1=tol1, tol2=tol2)
        out[task] = {}
        for i, r in enumerate(records):
            if r[task] is None:
                continue
            f = gfold[groups[i]]
            out[task][r['id']] = f
            rows.append({'Task': task, 'Fold': f, 'Strain_ID': r['id'], 'True_Label': r[task],
                         'ANI_Group': groups[i], 'Group_Protocol': SCHEME, 'Cohort': cohort_digest})
        report[task] = dict(rep, groups=len(gl), strains=int(sum(s for _, s, _ in gl)),
                            tolerant=int(sum(p for _, _, p in gl)))
    rows.sort(key=lambda r: (TASKS.index(r['Task']), r['Fold'], r['ANI_Group'], r['Strain_ID']))
    return out, rows, report


def fold_table(records, groups, task, fold_of_strain, scheme=''):
    rows = []
    lab = [i for i, r in enumerate(records) if r[task] is not None]
    N = len(lab); P = sum(records[i][task] for i in lab); G = len({groups[i] for i in lab})
    for f in range(1, K + 1):
        val = [i for i in lab if fold_of_strain[records[i]['id']] == f]
        p = sum(records[i][task] for i in val)
        sp = Counter(records[i]['species'] for i in val)
        rows.append({'Scheme': scheme, 'Task': task, 'Fold': f, 'Strains': len(val),
                     'ANI_groups': len({groups[i] for i in val}), 'Tolerant': p, 'Non_tolerant': len(val) - p,
                     'Tolerant_fraction': p / len(val), 'Training_strains': N - len(val),
                     'Training_tolerant': P - p, 'Target_strains': N / K, 'Target_groups': G / K,
                     'Target_tolerant_fraction': P / N,
                     'Species_composition': '; '.join(f'{s} {c}' for s, c in sorted(sp.items(), key=lambda x: (-x[1], x[0])))})
    return rows


def balance_summary(rows):
    n = np.array([r['Strains'] for r in rows]); k = np.array([r['ANI_groups'] for r in rows])
    q = np.array([r['Tolerant_fraction'] for r in rows]); pi = rows[0]['Target_tolerant_fraction']
    return {'Scheme': rows[0]['Scheme'], 'Task': rows[0]['Task'],
            'Strains_per_fold': '/'.join(map(str, n)), 'Strain_range': int(n.max() - n.min()),
            'Strain_CV': float(n.std() / n.mean()), 'Sum_abs_strain_deviation': float(np.abs(n - n.mean()).sum()),
            'Groups_per_fold': '/'.join(map(str, k)), 'Group_range': int(k.max() - k.min()),
            'Sum_abs_group_deviation': float(np.abs(k - k.mean()).sum()),
            'Tolerant_fraction_per_fold': '/'.join(f'{x:.3f}' for x in q),
            'Max_abs_tolerant_fraction_deviation': float(np.abs(q - pi).max()),
            'Mean_abs_tolerant_fraction_deviation': float(np.abs(q - pi).mean())}


def to_index_folds(records, groups, fold_of):
    """{task: (labeled indices, [(train_idx, val_idx) x 5])} with validation of all hard constraints."""
    folds = {}
    for task in TASKS:
        lab = [i for i, r in enumerate(records) if r[task] is not None]
        if set(fold_of[task]) != {records[i]['id'] for i in lab}:
            raise RuntimeError('[FOLDS BLOCKED] fold table does not cover exactly the labeled ' + task + ' strains')
        gf = {}
        for i in lab:
            f = fold_of[task][records[i]['id']]
            if not 1 <= f <= K:
                raise RuntimeError('[FOLDS BLOCKED] invalid fold number')
            if gf.setdefault(groups[i], f) != f:
                raise RuntimeError('[FOLDS BLOCKED] ANI group split across folds: ' + groups[i])
        five = []
        for f in range(1, K + 1):
            val = np.array([i for i in lab if fold_of[task][records[i]['id']] == f], dtype=int)
            tr = np.array([i for i in lab if fold_of[task][records[i]['id']] != f], dtype=int)
            if not len(val) or not len(tr):
                raise RuntimeError('[FOLDS BLOCKED] empty fold')
            for part in (val, tr):
                yy = [records[i][task] for i in part]
                if min(sum(yy), len(yy) - sum(yy)) < 1:
                    raise RuntimeError(f'[FOLDS BLOCKED] fold {f} of {task} lacks a class')
            if {groups[i] for i in tr} & {groups[i] for i in val}:
                raise RuntimeError('[FOLDS BLOCKED] ANI group leakage')
            five.append((tr, val))
        folds[task] = (np.array(lab, dtype=int), five)
    return folds


def write_csv(p, rows, fields=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    tmp.replace(p)


def read_fold_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = {t: {} for t in TASKS}
    for r in rows:
        out[r['Task']][r['Strain_ID']] = int(r['Fold'])
    return out, rows
