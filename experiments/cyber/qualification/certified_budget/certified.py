"""Rational Bellman bounds for a known-law budgeted folded-sign test."""
from fractions import Fraction as F
from functools import lru_cache
from math import comb


@lru_cache(maxsize=65536)
def wealth(w, l):
    return F(2**(w+l), (w+l+1)*comb(w+l,w))


class Planner:
    def __init__(self, costs, laws, budget, alpha=F(1,20)):
        self.costs = tuple(costs)
        self.laws = tuple(tuple(F(str(x)) for x in row) for row in laws)
        self.budget = budget
        self.alpha = F(str(alpha))
        if (not self.costs or len(self.costs)!=len(self.laws)
                or any(type(c) is not int or c<1 for c in self.costs)
                or type(budget) is not int or not 0<=budget<=150
                or not 0<self.alpha<1
                or any(len(r)!=3 or min(r)<0 or sum(r)!=1 for r in self.laws)):
            raise ValueError('invalid planning contract')
        # Uniform upper bound on expected one-step wealth multiplier.
        self.growth = [F(1)]*(budget+1)
        for b in range(1,budget+1):
            self.growth[b] = max([F(1)]+[(2*max(p,q)+t)*self.growth[b-c]
                for c,(p,q,t) in zip(self.costs,self.laws) if c<=b])

    @staticmethod
    def total(state):
        e=F(1)
        for w,l in state: e*=wealth(w,l)
        return e

    @lru_cache(maxsize=100000)
    def reachable(self,b,state):
        # Relax to perfect outcomes in each action; allow unused budget.
        dp=[F(1)]*(b+1)
        for c,(w,l) in zip(self.costs,state):
            vals=[max(wealth(w+k,l),wealth(w,l+k),wealth(w,l)) for k in range(b//c+1)]
            dp=[max(dp[r-k*c]*vals[k] for k in range(r//c+1)) for r in range(b+1)]
        return dp[b]*self.alpha>=1

    @lru_cache(maxsize=200000)
    def single(self,j,b,w,l,other):
        # Feasible continuation: spend all remaining affordable budget on j.
        if wealth(w,l)*other*self.alpha>=1: return F(1)
        c=self.costs[j]
        if b<c: return F(0)
        return sum((p*self.single(j,b-c,w+dw,l+dl,other)
                    for (dw,dl),p in zip(((1,0),(0,1),(0,0)),self.laws[j]) if p),F(0))

    @lru_cache(maxsize=200000)
    def bounds(self,b,state,depth):
        e=self.total(state)
        if e*self.alpha>=1: return F(1),F(1),None
        feasible=[j for j,c in enumerate(self.costs) if c<=b]
        if not feasible or not self.reachable(b,state): return F(0),F(0),None
        upper=min(F(1),e*self.growth[b]*self.alpha)
        if depth==0:
            lower,j=max((self.single(j,b,*state[j],e/wealth(*state[j])),j) for j in feasible)
            return lower,upper,j
        options=[]
        for j in feasible:
            lo=hi=F(0)
            for (dw,dl),p in zip(((1,0),(0,1),(0,0)),self.laws[j]):
                if not p: continue
                nxt=list(state);w,l=nxt[j];nxt[j]=(w+dw,l+dl)
                a,z,_=self.bounds(b-self.costs[j],tuple(nxt),depth-1)
                lo+=p*a;hi+=p*z
            options.append((lo,hi,j))
        lo,_,j=max(options,key=lambda x:(x[0],-x[2]))
        return lo,min(upper,max(x[1] for x in options)),j

    def solve(self,depth):
        if type(depth) is not int or not 0<=depth<=min(30,self.budget):
            raise ValueError('invalid lookahead depth')
        return self.bounds(self.budget,tuple((0,0) for _ in self.costs),depth)

    def clear(self):
        self.bounds.cache_clear(); self.single.cache_clear(); self.reachable.cache_clear()
