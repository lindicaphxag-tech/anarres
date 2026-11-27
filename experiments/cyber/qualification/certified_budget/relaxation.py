"""Perfect-information upper bound: exact pathwise knapsack, sampled expectation."""
from fractions import Fraction as F
from random import Random
from math import log, sqrt
from certified import wealth, Planner


def clairvoyant(costs, streams, budget, alpha):
    # For each action choose any prefix of its independent future stream.
    # This relaxes nonanticipativity, not the shared budget.
    if len(costs)!=len(streams) or any(len(s)<budget//c for c,s in zip(costs,streams)):
        raise ValueError('incomplete stream')
    dp=[F(1)]*(budget+1)
    for c,stream in zip(costs,streams):
        w=l=0; v=[F(1)]
        for z in stream[:budget//c]:
            if type(z) is not int or z not in (-1,0,1): raise ValueError('invalid sign')
            w+=z==1;l+=z==-1;v.append(max(v[-1],wealth(w,l)))
        dp=[max(dp[b-k*c]*v[k] for k in range(b//c+1)) for b in range(budget+1)]
    return dp[budget]*F(str(alpha))>=1


def estimate(costs,laws,budget,alpha=F(1,20),reps=1000,seed=20261127,delta=.05):
    p=Planner(costs,laws,budget,alpha)
    if type(reps) is not int or reps<1 or not 0<delta<1: raise ValueError('invalid simulation contract')
    rng=Random(seed); hits=0
    # Sample exact rational categorical laws with integer random draws.
    from math import lcm
    weights=[]
    for row in p.laws:
        d=lcm(*(v.denominator for v in row));weights.append((d,int(row[0]*d),int((row[0]+row[1])*d)))
    for _ in range(reps):
        streams=[]
        for c,(d,a,b) in zip(p.costs,weights):
            stream=[]
            for _ in range(budget//c):
                u=rng.randrange(d);stream.append(1 if u<a else -1 if u<b else 0)
            streams.append(stream)
        hits+=clairvoyant(p.costs,streams,budget,p.alpha)
    return dict(hits=hits,reps=reps,seed=seed,delta=delta,mean=hits/reps,
                confidence_upper=min(1.,hits/reps+sqrt(log(1/delta)/(2*reps))))
