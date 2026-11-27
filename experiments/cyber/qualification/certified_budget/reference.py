"""Budget-aware folded-sign testing. Costs and actions are fixed before each observation."""
from fractions import Fraction as F
from functools import lru_cache
from math import comb, log, sqrt
from random import Random


@lru_cache(maxsize=65536)
def wealth(w, l):
    return F(2**(w+l), (w+l+1)*comb(w+l,w))


def sign_law(m,p,q):
    p,q=F(str(p)),F(str(q))
    if type(m) is not int or m<2 or not 0<=p<=1 or not 0<=q<=1:
        raise ValueError('invalid model')
    def fold(r):
        a=[F(0)]*(m//2+1)
        for k in range(m+1): a[min(k,m-k)]+=comb(m,k)*r**k*(1-r)**(m-k)
        return a
    a,b=fold(p),fold(q)
    plus=sum(x*y for i,x in enumerate(a) for j,y in enumerate(b) if j>i)
    minus=sum(x*y for i,x in enumerate(a) for j,y in enumerate(b) if j<i)
    return plus,minus,1-plus-minus


def info(law):
    a,b,_=map(float,law); s=a+b
    return sum(x*log(2*x/s) for x in (a,b) if x) if s else 0.


class BudgetTest:
    def __init__(self,costs,budget,alpha=F(1,20),seed=0):
        self.costs=tuple(costs); self.alpha=F(str(alpha))
        if not self.costs or any(type(x) is not int or x<1 for x in self.costs) or type(budget) is not int or budget<0 or not 0<self.alpha<1:
            raise ValueError('invalid budget contract')
        self.remaining=budget; self.spent=0; self.counts=[[0,0,0] for _ in self.costs]
        self.pending=None; self.seen=set(); self.e=F(1); self.hit=False; self.rng=Random(seed)
    def choose(self,policy='adaptive',laws=None):
        if self.pending is not None or self.hit: raise ValueError('outstanding action or stopped')
        feasible=[i for i,c in enumerate(self.costs) if c<=self.remaining]
        if not feasible: return None
        if policy=='uniform': i=min(feasible,key=lambda j:(sum(self.counts[j]),j))
        elif policy=='oracle':
            if laws is None or len(laws)!=len(self.costs): raise ValueError('oracle requires laws')
            i=max(feasible,key=lambda j:info(laws[j])/self.costs[j])
        elif policy=='posterior':
            # Explore every affordable action at least three times, then sample
            # a Dirichlet(1,1,1) posterior and maximize KL per rollout.
            under=[j for j in feasible if sum(self.counts[j])<3]
            if under: i=min(under,key=lambda j:(sum(self.counts[j]),j))
            else:
                def score(j):
                    draw=[self.rng.gammavariate(x+1,1) for x in self.counts[j]]
                    return info(draw)/(sum(draw)*self.costs[j])
                i=max(feasible,key=score)
        elif policy=='adaptive':
            total=sum(map(sum,self.counts))
            def score(j):
                w,l,t=self.counts[j]; n=w+l+t
                # Smoothed sign-frequency estimate and explicit exploration bonus.
                # Heuristic, not a proved optimal index or confidence bound.
                return (info((w+1,l+1,t+1))/(n+3)+sqrt(log(total+2)/(n+1))*.15)/self.costs[j]
            i=max(feasible,key=score)
        else: raise ValueError('unknown policy')
        self.pending=i; return i
    def observe(self,task_id,sign):
        if self.pending is None or not isinstance(task_id,str) or not task_id or task_id in self.seen or type(sign) is not int or sign not in (-1,0,1):
            raise ValueError('invalid observation')
        i=self.pending; w,l,t=self.counts[i]; before=wealth(w,l)
        w+=sign==1;l+=sign==-1;t+=sign==0
        self.e*=wealth(w,l)/before; self.counts[i]=[w,l,t]
        self.spent+=self.costs[i];self.remaining-=self.costs[i];self.pending=None;self.seen.add(task_id)
        self.hit=self.e*self.alpha>=1
        return self.hit


def exact_oracle(costs,laws,budget,alpha=F(1,20),greedy=False):
    """Exact finite-budget optimum for the specified wealth rule and known independent laws.

    Exponential state space: intended only as a small-instance reference.
    """
    costs=tuple(costs); laws=tuple(tuple(F(str(x)) for x in row) for row in laws);alpha=F(str(alpha))
    if not costs or len(costs)!=len(laws) or any(type(c) is not int or c<1 for c in costs) or type(budget) is not int or not 0<=budget<=30 or not 0<alpha<1 or any(len(r)!=3 or min(r)<0 or sum(r)!=1 for r in laws): raise ValueError('invalid oracle contract')
    @lru_cache(None)
    def solve(left,state):
        e=F(1)
        for w,l in state:e*=wealth(w,l)
        if e*alpha>=1:return F(1),None
        best=F(0); action=None
        feasible=[i for i,c in enumerate(costs) if c<=left]
        if greedy and feasible: feasible=[max(feasible,key=lambda j:info(laws[j])/costs[j])]
        for i in feasible:
            c=costs[i]
            v=F(0)
            for (dw,dl),p in zip(((1,0),(0,1),(0,0)),laws[i]):
                if not p:continue
                nxt=list(state); w,l=nxt[i];nxt[i]=(w+dw,l+dl)
                v+=p*solve(left-c,tuple(nxt))[0]
            if action is None or v>best:best,action=v,i
        return best,action
    p,a=solve(budget,tuple((0,0) for _ in costs))
    return {'power':p,'first_action':a,'states':solve.cache_info().currsize}
