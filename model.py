"""Fold-identical fitted preprocessing for deployable final classifiers."""
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer

class FittedFusion:
    def __init__(self, upstream, method):
        self.upstream=tuple(upstream)
        self.method=method
        self.stages=[]
        self.final_indices=None
    def fit(self, blocks, y):
        y=np.asarray(y,dtype=np.int32)
        yc=y.astype(np.float32)-float(y.mean())
        xparts=[]
        self.stages=[]
        for name in self.upstream:
            a=np.asarray(blocks[name],dtype=np.float32)
            mu=a.mean(axis=0,dtype=np.float64).astype(np.float32)
            var=np.maximum(np.mean(a*a,axis=0)-mu*mu,0)
            cov=yc @ a
            strength=(cov*cov)/(np.maximum(var,1e-16)*max(float(yc@yc),1e-16))
            strength[~np.isfinite(strength)]=0
            nz=np.flatnonzero(var>1e-14)
            if self.method=='concat_z': chosen=nz
            else:
                cnt=256 if self.method in ('block256_z','select128') else 128
                if len(nz)>cnt:
                    ksel=np.argpartition(-strength[nz],cnt-1)[:cnt]
                    chosen=np.sort(nz[ksel])
                else:chosen=nz
            if not len(chosen): continue
            scaler=Normalizer() if self.method=='block128_l2' else StandardScaler()
            transformed=scaler.fit_transform(a[:,chosen])
            if self.method!='block128_l2': transformed=np.clip(transformed,-10,10)
            xparts.append(np.asarray(transformed,dtype=np.float32))
            self.stages.append((name,np.asarray(chosen,dtype=np.int64),scaler))
        if not xparts:raise ValueError('All selected features have zero variance')
        x=np.concatenate(xparts,axis=1)
        if self.method=='select128' and x.shape[1]>128:
            strength=np.abs(yc @ x)
            self.final_indices=np.sort(np.argpartition(-strength,127)[:128])
            x=x[:,self.final_indices]
        return x
    def transform(self, blocks):
        parts=[]
        for name,chosen,scaler in self.stages:
            a=np.asarray(blocks[name],dtype=np.float32)
            x=scaler.transform(a[:,chosen])
            if self.method!='block128_l2':x=np.clip(x,-10,10)
            parts.append(np.asarray(x,dtype=np.float32))
        if not parts:raise ValueError('Missing feature blocks')
        x=np.concatenate(parts,axis=1)
        return x[:,self.final_indices] if self.final_indices is not None else x
