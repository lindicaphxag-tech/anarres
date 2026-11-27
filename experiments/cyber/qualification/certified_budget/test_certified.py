import unittest
from fractions import Fraction as F
from itertools import product
from certified import Planner, wealth
from reference import exact_oracle


class Tests(unittest.TestCase):
    def test_reference(self):
        for b,costs,law in product((4,7,10,12),((1,2),(2,3)),(((F(3,4),F(1,4),0),(F(9,10),F(1,10),0)),((F(1,4),F(1,4),F(1,2)),(F(1,2),F(1,4),F(1,4))))):
            p=Planner(costs,law,b,F(1,5)); exact=exact_oracle(costs,law,b,F(1,5))['power']
            for d in range(4):
                lo,hi,_=p.solve(d);self.assertLessEqual(lo,exact);self.assertLessEqual(exact,hi)
            lo,hi,_=p.solve(b//min(costs));self.assertEqual((lo,hi),(exact,exact));p.clear()
    def test_nested(self):
        p=Planner((1,2),((.75,.25,0),(.9,.1,0)),12,.2)
        intervals=[p.solve(d)[:2] for d in range(7)]
        for (lo,hi),(nl,nh) in zip(intervals,intervals[1:]):
            self.assertLessEqual(lo,nl);self.assertGreaterEqual(hi,nh)
        p.clear()
    def test_reachability(self):
        p=Planner((1,2),((.5,.5,0),(.5,.5,0)),8)
        for w,l in product(range(4),repeat=2):
            for n in range(5):
                self.assertLessEqual(max(wealth(w+i,l+n-i) for i in range(n+1)),max(wealth(w+n,l),wealth(w,l+n)))
        self.assertFalse(p.reachable(7,((0,0),(0,0))))
        self.assertTrue(p.reachable(8,((0,0),(0,0))))
        p.clear()
    def test_no_budget(self):
        p=Planner((2,),((1,0,0),),1);self.assertEqual(p.solve(0),(0,0,None));p.clear()
    def test_null_bound(self):
        p=Planner((1,2),((.5,.5,0),(.2,.2,.6)),20)
        self.assertLessEqual(p.solve(0)[1],F(1,20));p.clear()
    def test_invalid(self):
        for c,l,b in [((0,),((1,0,0),),1),((1,),((.5,.5,.5),),3),((1,),((1,0,0),),151)]:
            with self.assertRaises(ValueError):Planner(c,l,b)
        p=Planner((1,),((1,0,0),),1)
        with self.assertRaises(ValueError):p.solve(-1)
        p.clear()

    def test_relaxation_knapsack(self):
        from relaxation import clairvoyant
        for a,b in product(product((-1,1),repeat=4),product((-1,1),repeat=2)):
            w=l=0;v=[F(1)]
            for z in a: w+=z==1;l+=z==-1;v.append(wealth(w,l))
            w=l=0;u=[F(1)]
            for z in b: w+=z==1;l+=z==-1;u.append(wealth(w,l))
            expected=any(v[i]*u[j]>=2 for i in range(5) for j in range(3) if i+2*j<=4)
            self.assertEqual(clairvoyant((1,2),(a,b),4,F(1,2)),expected)
    def test_relaxation_dominates_exact(self):
        from relaxation import clairvoyant
        # Enumerate 2^9 independent streams with exact probabilities.
        c=(1,2);b=6;alpha=F(1,5);laws=((F(3,4),F(1,4),0),(F(9,10),F(1,10),0));upper=F(0)
        for a,z in product(product((-1,1),repeat=6),product((-1,1),repeat=3)):
            mass=F(3,4)**a.count(1)*F(1,4)**a.count(-1)*F(9,10)**z.count(1)*F(1,10)**z.count(-1)
            upper+=mass*clairvoyant(c,(a,z),b,alpha)
        self.assertGreaterEqual(upper,exact_oracle(c,laws,b,alpha)['power'])

if __name__=='__main__':unittest.main()
