"""Synthetic known-model comparison; each policy sees common per-action streams."""
import json, random
from budgeted import BudgetTest, sign_law, exact_oracle, info
from fractions import Fraction as F

SCENARIOS={
    'null':((4,.1,.9),(8,.2,.2),(12,.4,.6)),
    'cheap_signal':((4,.05,.45),(8,.2,.25),(12,.4,.6)),
    'costly_signal':((4,.1,.9),(8,.2,.25),(12,.05,.5)),
    'mixed':((4,.1,.25),(8,.45,.1),(12,.2,.45)),
}

def run(reps=1000,budget=480):
    out={}
    for name,models in SCENARIOS.items():
        costs=[2*m for m,_,_ in models]; laws=[sign_law(*model) for model in models]
        records={p:[] for p in ('uniform','adaptive','posterior','oracle')}
        for rep in range(reps):
            rng=random.Random(20261127+rep)
            streams=[rng.choices((1,-1,0),weights=tuple(map(float,law)),k=budget//c) for c,law in zip(costs,laws)]
            for policy in records:
                t=BudgetTest(costs,budget,seed=20261127+rep); pos=[0]*len(costs)
                while not t.hit:
                    a=t.choose(policy,laws)
                    if a is None:break
                    t.observe(f'{a}:{pos[a]}',streams[a][pos[a]]);pos[a]+=1
                records[policy].append((t.hit,t.spent))
        out[name]={'information_per_cost':[info(x)/c for x,c in zip(laws,costs)],'policies':{p:{'reject':sum(x for x,_ in r),'rate':sum(x for x,_ in r)/reps,'mean_cost':sum(c for _,c in r)/reps} for p,r in records.items()}}
    # small known-law DP reference, normalized integer costs and alpha .2
    laws=((F(3,4),F(1,4),F(0)),(F(9,10),F(1,10),F(0)))
    v=exact_oracle([1,2],laws,12,F(1,5)); g=exact_oracle([1,2],laws,12,F(1,5),greedy=True)
    return {'reps':reps,'budget_rollouts':budget,'scenarios':out,'small_oracle':{'power':str(v['power']),'power_float':float(v['power']),'states':v['states'],'first_action':v['first_action'],'greedy_power':str(g['power']),'greedy_power_float':float(g['power'])}}

if __name__=='__main__':print(json.dumps(run(),indent=2))
