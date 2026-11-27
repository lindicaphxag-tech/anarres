import itertools
import unittest
from fractions import Fraction as F
from contextual_folded import ContextualFolded, beta_wealth, context_keys


class Tests(unittest.TestCase):
    def test_exact_conditional_identity(self):
        for w in range(21):
            for l in range(21):
                self.assertEqual((beta_wealth(w+1,l)+beta_wealth(w,l+1))/2,beta_wealth(w,l))

    def test_exhaustive_paths(self):
        # Vary the unordered pair and group size. Even adaptive context is allowed.
        crossing=F(0); expectation=F(0)
        for signs in itertools.product((-1,1),repeat=12):
            t=ContextualFolded(F(1,5)); previous=1
            for i,s in enumerate(signs):
                m=8 if previous>0 else 12
                a,b=(0,1) if i%2 else (0,m//2)
                if s<0: a,b=b,a
                r=t.update(str(i),a,b,m);previous=s
            crossing+=F(t.first_crossing is not None,2**12)
            expectation+=r['e']/2**12
        self.assertEqual(expectation,1)
        self.assertLessEqual(crossing,F(1,5))

    def test_reflection_and_swap(self):
        t,u,v=ContextualFolded(),ContextualFolded(),ContextualFolded()
        for i,(a,b,m) in enumerate([(0,4,8),(3,2,10),(7,1,8),(6,0,12)]*5):
            x=t.update(str(i),a,b,m)
            self.assertEqual(x,u.update(str(i),m-a,m-b,m))
            self.assertEqual(x,v.update(str(i),b,a,m))

    def test_gap_cancel(self):
        t=ContextualFolded()
        for i in range(40):
            r=t.update(str(i),*( (0,4,8) if i%2 else (1,0,8)))
        self.assertLess(r['expert_e'][0],1)
        self.assertGreater(r['expert_e'][1],20)
        self.assertIsNotNone(r['first_crossing'])

    def test_ties_no_change(self):
        t=ContextualFolded();x=t.update('a',1,4,8)
        y=t.update('b',1,7,8)
        self.assertEqual(x['e'],y['e'])

    def test_invalid_atomic(self):
        t=ContextualFolded();t.update('a',0,1,8)
        for args in [('a',0,1,8),('b',0,9,8),('b',0,1,True),('',0,1,8)]:
            with self.assertRaises(ValueError):t.update(*args)
        self.assertEqual(len(t.seen),1)

    def test_global_reference(self):
        t=ContextualFolded();w=l=0
        for i,(a,b,m) in enumerate([(0,1,8),(2,1,9),(1,7,8),(1,4,8)]*9):
            s,_=context_keys(a,b,m);w+=s>0;l+=s<0
            self.assertEqual(t.update(str(i),a,b,m)['expert_e'][0],beta_wealth(w,l))


if __name__=='__main__':unittest.main()
