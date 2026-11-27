"""Synthetic exploratory benchmark only (NumPy optional)."""
import json, math
import numpy as np
from contextual_folded import ContextualFolded

N=160; REPS=3000; SEED=20261127

def run(name):
    rng=np.random.default_rng(SEED)
    m=np.resize([8,12,8,12],N)[None,:]
    p=np.resize([.1,.2,.1,.2],N)[None,:]
    if name=='null': q=np.resize([.9,.2,.1,.8],N)[None,:]
    elif name=='modest': q=p+.1
    elif name=='strong': q=np.full((1,N),.5)
    elif name=='gap_cancel':
        large=rng.random((REPS,N))<.5
        plus=rng.random((REPS,N))<np.where(large,.85,.15)
        gap=np.where(large,m//2,1)
        a=np.where(plus,0,gap);b=np.where(plus,gap,0)
    else: raise ValueError(name)
    if name!='gap_cancel':
        a=rng.binomial(m,p,(REPS,N));b=rng.binomial(m,q,(REPS,N))
    fa=np.minimum(a,m-a);fb=np.minimum(b,m-b);s=np.sign(fb-fa)
    gap=(4*np.abs(fb-fa)>m);size=np.broadcast_to(m>8,s.shape)
    lookup=np.zeros((N+1,N+1))
    for w in range(N+1):
        for l in range(N+1-w):
            lookup[w,l]=2**(w+l)/((w+l+1)*math.comb(w+l,w))
    def wealth(mask):
        w=np.cumsum(mask&(s>0),axis=1);l=np.cumsum(mask&(s<0),axis=1)
        return lookup[w,l]
    base=wealth(np.ones_like(s,dtype=bool));new=(base+wealth(gap)*wealth(~gap)+wealth(size)*wealth(~size))/3
    # Verify the vectorized benchmark against actual exact production implementation.
    for i in range(4):
        t=ContextualFolded()
        for j in range(N):
            r=t.update(str(j),int(a[i,j]),int(b[i,j]),int(m[0,j]))
            assert math.isclose(float(r['e']),new[i,j],rel_tol=1e-12)
    def stats(e):
        hit=e>=20;yes=hit.any(axis=1);stop=np.where(yes,hit.argmax(axis=1)+1,N)
        return {'reject':int(yes.sum()),'rejection_rate':float(yes.mean()),'mean_tasks':float(stop.mean())}
    return {'scenario':name,'sign':stats(base),'contextual':stats(new)}

if __name__=='__main__':
    print(json.dumps({'seed':SEED,'reps':REPS,'max_tasks':N,'results':[run(x) for x in ('null','modest','strong','gap_cancel')]},indent=2))
