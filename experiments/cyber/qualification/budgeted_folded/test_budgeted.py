import itertools, unittest
from fractions import Fraction as F
from budgeted import BudgetTest, exact_oracle, sign_law, wealth, info

class Test(unittest.TestCase):
    def test_null_laws(self):
        for m,p in itertools.product(range(2,13),(F(i,10) for i in range(11))):
            for q in (p,1-p):
                a,b,t=sign_law(m,p,q);self.assertEqual(a,b);self.assertEqual(a+b+t,1)
    def test_martingale(self):
        for w,l in itertools.product(range(21),repeat=2):
            self.assertEqual((wealth(w+1,l)+wealth(w,l+1))/2,wealth(w,l))
    def test_budget_and_atomicity(self):
        t=BudgetTest([2,4],3);self.assertEqual(t.choose(),0)
        with self.assertRaises(ValueError):t.choose()
        with self.assertRaises(ValueError):t.observe('x',True)
        self.assertEqual(t.remaining,3);t.observe('x',0);self.assertEqual(t.remaining,1);self.assertIsNone(t.choose())
    def test_adaptive_exact_paths(self):
        # fair signs, action-dependent tie probabilities, predictable allocation
        costs=(1,2); laws=((F(1,2),F(1,2),F(0)),(F(1,4),F(1,4),F(1,2)))
        total=F(0);reject=F(0)
        def walk(history,mass):
            nonlocal total,reject
            t=BudgetTest(costs,10,F(1,5))
            for i,s in enumerate(history): t.choose();t.observe(str(i),s)
            if t.hit or (a:=t.choose()) is None:
                total+=mass;reject+=mass*t.hit;return
            for s,p in zip((1,-1,0),laws[a]):
                if p:walk(history+(s,),mass*p)
        walk((),F(1));self.assertEqual(total,1);self.assertLessEqual(reject,F(1,5))
    def test_oracle_single(self):
        law=(F(1),F(0),F(0));self.assertEqual(exact_oracle([1],[law],7)['power'],0);self.assertEqual(exact_oracle([1],[law],8)['power'],1)
    def test_oracle_exhaustive(self):
        laws=((F(3,4),F(1,4),F(0)),(F(1),F(0),F(0)))
        # Independent recursive action-tree calculation, no memoization.
        def brute(left,states):
            e=wealth(*states[0])*wealth(*states[1])
            if e>=5:return F(1)
            choices=[F(0)]
            for i,c in enumerate((1,2)):
                if c<=left:
                    v=F(0)
                    for z,p in enumerate(laws[i][:2]):
                        nxt=list(states);w,l=nxt[i];nxt[i]=(w+(z==0),l+(z==1));v+=p*brute(left-c,nxt)
                    choices.append(v)
            return max(choices)
        self.assertEqual(exact_oracle([1,2],laws,7,F(1,5))['power'],brute(7,[(0,0),(0,0)]))
    def test_oracle_dominates_greedy(self):
        laws=((F(3,4),F(1,4),F(0)),(F(9,10),F(1,10),F(0)))
        optimal=exact_oracle([1,2],laws,12,F(1,5))
        greedy=exact_oracle([1,2],laws,12,F(1,5),greedy=True)
        self.assertGreaterEqual(optimal["power"],greedy["power"])
    def test_posterior_budget(self):
        t=BudgetTest([2,4],37,seed=17)
        while not t.hit and t.choose("posterior") is not None:
            t.observe(str(len(t.seen)),0)
        self.assertEqual(t.spent,36)
        self.assertEqual(t.e,1)
    def test_info(self):
        self.assertEqual(info((F(1,4),F(1,4),F(1,2))),0)
        self.assertGreater(info((F(3,4),F(1,4),0)),0)

if __name__=='__main__':unittest.main()
