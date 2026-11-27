"""Anytime reward-variance tests, independent prototype, stdlib only."""
from fractions import Fraction
from math import comb, isfinite

class UnionVarianceTest:
    """Paired independent binary observations: global p=q OR p+q=1.

    Under a null branch its selected informative signs are conditionally fair.
    Uniform beta mixture e-value is computed exactly, with absorbing crossings.
    No claim for arbitrary mean-variance equality across heterogeneous tasks.
    """
    def __init__(self, alpha=Fraction(1,20)):
        self.alpha=Fraction(alpha)
        if not 0<self.alpha<1: raise ValueError('alpha')
        self.counts=[[0,0],[0,0]]
        self.crossed=[False,False]
        self.n=0
    def update(self,a,b):
        if a not in (0,1) or b not in (0,1): raise ValueError('binary')
        j=int(a==b)
        self.counts[j][a]+=1
        self.n+=1
        x,y=self.counts[j]; m=x+y
        self.crossed[j] |= (2**m*self.alpha.numerator >= (m+1)*comb(m,x)*self.alpha.denominator)
        return all(self.crossed)
    @property
    def evalues(self):
        return tuple(Fraction(2**(x+y),(x+y+1)*comb(x+y,x)) for x,y in self.counts)

class TaskMeanTest:
    """Two-sided betting test of zero conditional mean of bounded task differences.

    Each task supplies independent conditional Bernoulli draws within each arm
    for interpretation as raw variance. Arbitrary within-pair dependence across
    arms is allowed. Across tasks the null requires conditional mean zero.
    Fixed mixture over +/- bets; exact rational arithmetic for audit.
    """
    def __init__(self,alpha=Fraction(1,20),bets=None):
        self.alpha=Fraction(alpha)
        if not 0<self.alpha<1: raise ValueError('alpha')
        self.bets=tuple(Fraction(v) for v in (bets if bets is not None else [Fraction(i,16) for i in range(-16,17) if i]))
        if not self.bets or any(abs(v)>1 for v in self.bets): raise ValueError('bets')
        self.wealth=[Fraction(1)]*len(self.bets)
        self.ids=set(); self.crossed=False
    def update(self,task_id,ka,na,kb,nb):
        if task_id in self.ids: raise ValueError('duplicate task')
        def u(k,n):
            if not isinstance(k,int) or not isinstance(n,int) or n<2 or not 0<=k<=n: raise ValueError('invalid counts')
            return Fraction(k*(n-k),n*(n-1))
        x=2*(u(ka,na)-u(kb,nb)) # U in [0,1/2], hence |x|<=1
        self.ids.add(task_id)
        self.wealth=[w*(1+l*x) for w,l in zip(self.wealth,self.bets)]
        self.crossed |= self.evalue*self.alpha>=1
        return self.crossed
    @property
    def evalue(self): return sum(self.wealth)/len(self.wealth)
