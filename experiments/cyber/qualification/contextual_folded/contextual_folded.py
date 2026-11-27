"""Exact contextual sign mixtures under conditional folded-pair exchangeability."""
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from functools import lru_cache


@lru_cache(maxsize=32768)
def beta_wealth(w, l):
    if any(type(x) is not int or x < 0 for x in (w, l)):
        raise ValueError('invalid counts')
    return Fraction(2**(w+l), (w+l+1)*comb(w+l, w))


def context_keys(a, b, m):
    if type(m) is not int or m < 2 or any(type(k) is not int or not 0 <= k <= m for k in (a,b)):
        raise ValueError('invalid group')
    fa, fb = min(a,m-a), min(b,m-b)
    # Fixed before data: partition by normalized gap or group-size band.
    return ((fb > fa)-(fb < fa), (0, int(4*abs(fb-fa) > m), int(m > 8)))


@dataclass
class ContextualFolded:
    """Three equally weighted experts: global, gap, group size.

    Conditional fairness given the unordered pair and past is essential.
    A tie does not change wealth. Each expert is a product of per-context
    uniform-Beta sign mixtures. The final test is ONE convex mixture.
    """
    alpha: Fraction = Fraction(1,20)
    counts: list = field(default_factory=lambda: [{},{},{}], init=False)
    seen: set = field(default_factory=set, init=False)
    max_wealth: Fraction = field(default=Fraction(1), init=False)
    first_crossing: int | None = field(default=None, init=False)

    def __post_init__(self):
        self.alpha = Fraction(str(self.alpha))
        if not 0 < self.alpha < 1:
            raise ValueError('invalid alpha')

    def update(self, task_id, a, b, m):
        if not isinstance(task_id,str) or not task_id or task_id in self.seen:
            raise ValueError('invalid or repeated task id')
        sign, keys = context_keys(a,b,m)
        self.seen.add(task_id)
        experts = []
        for table,key in zip(self.counts,keys,strict=True):
            if sign:
                w,l = table.get(key,(0,0))
                table[key] = (w+int(sign>0),l+int(sign<0))
            e = Fraction(1)
            for w,l in table.values():
                e *= beta_wealth(w,l)
            experts.append(e)
        value = sum(experts)/len(experts)
        self.max_wealth = max(self.max_wealth,value)
        if self.first_crossing is None and value*self.alpha >= 1:
            self.first_crossing = len(self.seen)
        return {'tasks':len(self.seen),'e':value,'expert_e':tuple(experts),
                'anytime_p':1/self.max_wealth,'first_crossing':self.first_crossing}
