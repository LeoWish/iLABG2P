"""Load exact frozen 2026 ANI>=99.5% OOF folds (no ANI recomputation)."""
from __future__ import annotations
import csv, hashlib, json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

def _read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def _hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def _need(ok,msg):
    if not ok: raise RuntimeError('[FROZEN ANI BLOCKED] '+msg)

def load_frozen(records,cohort,root,out):
    """Return (all-256 groups, {'acid':(inds,5 folds),...}, frozen metadata)."""
    root,out=Path(root),Path(out)
    p=root/'plan'
    gf=p/'frozen_256_ANI995_groups.csv'
    ff=p/'00_Frozen_ANI995_Folds.csv'
    provf=p/'00_Fold_Provenance.json'
    for f in (gf,ff,provf):_need(f.is_file(),'Missing '+str(f))
    provenance=json.loads(provf.read_text(encoding='utf-8'))
    _need(provenance.get('cohort_digest')==cohort['cohort_digest'],
          'Genome cohort SHA/identity differs from frozen original')
    _need(provenance.get('folds_sha256')==_hash(ff),
          'Frozen fold CSV SHA256 differs from original lock')
    _need('ANI' in str(provenance.get('ANI_method','')).upper(),
          'Frozen provenance is not an ANI assignment')
    index={r['id']:i for i,r in enumerate(records)}
    _need(len(index)==len(records)==256,'Expected 256 unique genome IDs')
    groups_by_id={}
    for row in _read(gf):
        sid=row.get('Strain_ID','')
        _need(sid in index and sid not in groups_by_id,'Unexpected/repeated group ID '+sid)
        record=records[index[sid]]
        _need(row.get('Genome_SHA256')==record['genome_sha'],
              'Genome SHA256 differs for '+sid)
        group=row.get('ANI995_Group','')
        _need(bool(group),'Empty ANI group '+sid)
        groups_by_id[sid]=group
    _need(len(groups_by_id)==256,'Frozen group CSV must cover all 256 strains')
    groups=[groups_by_id[r['id']] for r in records]
    _need(len(set(groups))==89,'Frozen ANI group count differs from verified 89')
    original=_read(ff)
    _need(len(original)==394,'Expected 175 acid + 219 bile fold assignments')
    task_rows={'acid':{},'bile':{}}
    group_fold={'acid':{},'bile':{}}
    for row in original:
        task=row.get('Task','')
        _need(task in task_rows,'Unexpected task in frozen fold CSV')
        sid=row.get('Strain_ID','')
        _need(sid in index and sid not in task_rows[task],
              'Unexpected/duplicated frozen strain '+task+'/'+sid)
        group=groups_by_id[sid]
        _need(row.get('ANI_Group')==group,
              'Frozen fold ANI group differs: '+sid)
        _need(row.get('Cohort')==cohort['cohort_digest'],
              'Frozen fold cohort digest differs: '+sid)
        _need(str(records[index[sid]][task])==str(row.get('True_Label')),
              'Label differs from frozen fold: '+task+'/'+sid)
        fold=int(row.get('Fold','0'))
        _need(1<=fold<=5,'Invalid fold for '+sid)
        other=group_fold[task].setdefault(group,fold)
        _need(other==fold,'ANI group appears in multiple folds: '+group)
        task_rows[task][sid]=fold
    folds={}
    for task,n in (('acid',175),('bile',219)):
        labeled=[i for i,r in enumerate(records) if r[task] is not None]
        expected_ids={records[i]['id'] for i in labeled}
        _need(len(labeled)==n and set(task_rows[task])==expected_ids,
              'Incomplete '+task+' frozen validation set')
        inds=np.array(labeled,dtype=int)
        ffive=[]
        for fold in range(1,6):
            val=np.array([i for i in labeled if task_rows[task][records[i]['id']]==fold],dtype=int)
            tr=np.array([i for i in labeled if task_rows[task][records[i]['id']]!=fold],dtype=int)
            _need(len(val)>0 and len(tr)>0,'Empty fold')
            _need(len({records[i][task] for i in val})==2 and
                  len({records[i][task] for i in tr})==2,
                  'Frozen split is missing a class: '+task+'/'+str(fold))
            _need(not {groups[i] for i in tr}&{groups[i] for i in val},
                  'ANI group leakage: '+task+'/'+str(fold))
            ffive.append((tr,val))
        folds[task]=(inds,ffive)
    out.mkdir(parents=True,exist_ok=True)
    for src in (ff,provf):
        target=out/src.name
        if target.exists():
            _need(_hash(src)==_hash(target),'Conflicting existing output: '+str(target))
        else:shutil.copy2(src,target)
    print('[FROZEN ANI PASS] 256 genomes; 89 ANI>=99.5% groups; exact original 5 folds; no skani needed',flush=True)
    return groups,folds,provenance
