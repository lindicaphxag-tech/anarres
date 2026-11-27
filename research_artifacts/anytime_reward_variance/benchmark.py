"""Seeded synthetic comparisons; not real model or CyberGym experiments."""
import json
import numpy as np
from scipy.special import gammaln
from scipy.stats import hypergeom
from anytime import UnionVarianceTest

def table(n):
    v=np.ones((n+1,n+1))
    for total in range(2*n+1):
        a=np.arange(max(0,total-n),min(n,total)+1)
        pm=hypergeom.pmf(a,2*n,total,n)
        # inclusive two-sided probability ordering
        v[a,total-a]=np.array([pm[pm<=x*(1+1e-10)].sum() for x in pm])
    return np.maximum(v,v[:,::-1])

def fair_crossing(n,alpha=.05):
    # all signs informative; crossing probability is a universal upper bound
    active=np.ones(1); crossed=0.
    for m in range(1,n+1):
        tmp=np.zeros(m+1); tmp[:-1]+=active/2; tmp[1:]+=active/2
        x=np.arange(m+1)
        e=m*np.log(2)-np.log(m+1)-gammaln(m+1)+gammaln(x+1)+gammaln(m-x+1)
        hit=e>=-np.log(alpha); crossed+=tmp[hit].sum(); tmp[hit]=0; active=tmp
    assert abs(crossed+active.sum()-1)<1e-12
    return float(crossed)

def run():
    rng=np.random.default_rng(20261127); R=10000; N=480; looks=(160,320,480); tables={n:table(n) for n in looks}; rows=[]
    for p,q in [(0.035,0.035),(0.035,0.965),(.5,.5),(.035,.0525),(.035,.0805),(.1,.4),(.3,.5)]:
        counts=np.zeros((2,2,R),dtype=int); crossed=np.zeros((2,R),bool); t1=np.full(R,N+1); t2=t1.copy(); t3=t1.copy(); sa=np.zeros(R,int); sb=sa.copy()
        for n in range(1,N+1):
            a=rng.random(R)<p; b=rng.random(R)<q; sa+=a; sb+=b
            for j,mask in enumerate([a!=b,a==b]):
                counts[j,0]+=mask&~a; counts[j,1]+=mask&a
                x,y=counts[j]; m=x+y
                loge=m*np.log(2)-np.log(m+1)-gammaln(m+1)+gammaln(x+1)+gammaln(y+1)
                crossed[j]|=loge>=np.log(20)
            t1[(t1==N+1)&crossed.all(axis=0)]=n
            if n in tables:
                pv=tables[n][sa,sb]; t2[(t2==N+1)&(pv<=.05/3)]=n; t3[(t3==N+1)&(pv<=.05)]=n
        row={'p':p,'q':q}
        for name,t in [('anytime',t1),('three_look_spending',t2),('unadjusted_peeking',t3)]:
            hits=int((t<=N).sum()); row[name]={'rejections':hits,'rate':hits/R,'mean_n':float(np.minimum(t,N).mean())}
        rows.append(row)
    # exact-rational streaming oracle vs vectorized benchmark arithmetic
    for _ in range(100):
        a=rng.integers(0,2,size=(100,2)); t=UnionVarianceTest(); c=np.zeros((2,2),int); hits=np.zeros(2,bool)
        for x,y in a:
            j=int(x==y); c[j,x]+=1; m=c[j].sum(); s=c[j,0]
            hits[j]|=m*np.log(2)-np.log(m+1)-gammaln(m+1)+gammaln(s+1)+gammaln(m-s+1)>=np.log(20)
            assert t.update(int(x),int(y))==bool(hits.all())
    out={'seed':20261127,'replicates_per_scenario':R,'max_pairs':N,'looks':looks,'fair_sign_crossing_bound_at_480':fair_crossing(N),'exact_reference_trace_checks':10000,'results':rows}
    print(json.dumps(out,indent=2)); open('results.json','w').write(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
