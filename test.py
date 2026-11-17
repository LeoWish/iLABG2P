"""Predict unlabeled FASTA files from test_data using saved final classifiers."""
import argparse,csv,hashlib,os
from pathlib import Path
from types import SimpleNamespace
import joblib
from settings import ROOT
from core import legacy_engine as engine

def get_records():
    d=ROOT/'test_data'
    files=sorted(p for p in d.rglob('*') if p.is_file() and engine.fasta_stem(p) is not None)
    if not files:raise RuntimeError('Place new .fa/.fna/.fasta[.gz] genomes in test_data/')
    records=[];seen=set()
    for f in files:
        rid=engine.fasta_stem(f)
        if rid in seen:raise RuntimeError('Duplicate new Strain_ID: '+rid)
        seen.add(rid)
        records.append({'id':rid,'fasta':str(f.resolve()),'genome_sha':engine.sha(f)})
    return records

def load_features(records,necessary,protocol):
    k=[x for x in necessary if x.endswith('mer')]
    enc=[x for x in necessary if not x.endswith('mer')]
    mats={}
    if k:
        args=SimpleNamespace(windows=protocol['windows'],window_bp=protocol['window_bp'],
                             kmer_bins=protocol['kmer_bins'])
        km=engine.kmer_features(records,ROOT/'cacher',args)
        mats.update({name:km[name] for name in k})
    for name in enc:
        path=ROOT/'test_data'/'embeddings'/(name+'.npz')
        if not path.is_file():
            raise RuntimeError('Model uses '+name+'. Authentic embedding missing: '+str(path)
                              +'\nGenerate embeddings with the authentic encoder first; no fake features allowed.')
        mats[name]=engine.load_npz_matrix(path,records)
    return mats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trained-mode',choices=('auto','full','smoke'),default='auto')
    ap.add_argument('--rank',type=int,default=1)
    ap.add_argument('--task',choices=('acid','bile','both'),default='both')
    opt=ap.parse_args()
    mode=next((m for m in ('full','smoke') if (ROOT/'save_models'/m/'acid/Top01.joblib').is_file()),None) if opt.trained_mode=='auto' else opt.trained_mode
    if mode is None:raise RuntimeError('No saved final models. Run python run.py first.')
    records=get_records()
    tasks=('acid','bile') if opt.task=='both' else (opt.task,)
    rows=[]
    for task in tasks:
        file=ROOT/'save_models'/mode/task/f'Top{opt.rank:02d}.joblib'
        if not file.is_file():raise RuntimeError('Trained model missing: '+str(file))
        obj=joblib.load(file)  # use only locally produced trusted artifacts
        features=load_features(records,obj['upstream'],obj)
        X=obj['fusion_pipeline'].transform(features)
        prob=obj['classifier'].predict_proba(X)[:,1]
        for r,p in zip(records,prob):
            rows.append({'Strain_ID':r['id'],'Genome_SHA256':r['genome_sha'],
                         'Task':task,'Model_Rank':opt.rank,'Combination_ID':obj['combination_id'],
                         'Predicted_Probability':format(float(p),'.10g'),
                         'Predicted_Label':int(p>=obj['threshold']),
                         'Threshold':obj['threshold']})
    out=ROOT/'results'/'predictions'
    out.mkdir(parents=True,exist_ok=True)
    dest=out/'New_Genome_Predictions.csv'
    engine.write_csv(dest,rows,list(rows[0]))
    print('[PREDICTION PASS]',len(records),'new genomes;',len(rows),'predictions',dest)

if __name__=='__main__':main()
