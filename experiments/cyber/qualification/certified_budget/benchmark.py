"""Reproduce deterministic planning cases; no observed training data."""
import json,time
from certified import Planner
from reference import exact_oracle

CASES=[('small', (1,2), ((.75,.25,0),(.9,.1,0)),12,.2,12),('null',(1,2),((.5,.5,0),(.2,.2,.6)),24,.05,4),('mixed', (1,2,3), ((.65,.25,.1),(.75,.15,.1),(.85,.05,.1)),18,.05,8),('large',(2,3,5),((.65,.25,.1),(.8,.1,.1),(.9,.05,.05)),42,.05,6),('rare',(2,3,5),((.52,.43,.05),(.55,.35,.1),(.65,.2,.15)),60,.05,6)]

def run():
    out=[]
    for name,c,l,b,a,limit in CASES:
        p=Planner(c,l,b,a); rows=[]
        for d in sorted({0, min(4,limit), limit}):
            p.clear()
            t=time.perf_counter();lo,hi,j=p.solve(d)
            rows.append(dict(depth=d,lower=float(lo),upper=float(hi),gap=float(hi-lo),action=j,seconds=time.perf_counter()-t,bellman_states=p.bounds.cache_info().currsize))
        p.clear()
        exact=None
        if name=='small':exact=float(exact_oracle(c,l,b,a)['power'])
        out.append(dict(name=name,costs=c,laws=l,budget=b,alpha=a,exact=exact,rows=rows))
    return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
