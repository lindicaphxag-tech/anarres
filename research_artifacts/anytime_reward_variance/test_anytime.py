import itertools, unittest
from fractions import Fraction as F
from math import comb
from anytime import UnionVarianceTest, TaskMeanTest

class Tests(unittest.TestCase):
 def test_mixture_exact(self):
  for m in range(9):
   for x in range(m+1):
    t=UnionVarianceTest()
    for a in [1]*x+[0]*(m-x): t.update(a,1-a)
    self.assertEqual(t.evalues[0],F(2**m,(m+1)*comb(m,x)))
    self.assertEqual(t.evalues[1],1)
 def test_martingale(self):
  for m in range(12):
   for x in range(m+1):
    e=lambda n,k:F(2**n,(n+1)*comb(n,k))
    self.assertEqual((e(m+1,x)+e(m+1,x+1))/2,e(m,x))
 def test_exhaustive_union(self):
  # Rational probability of EVER rejecting under both branches, including intersection.
  for p,q in [(F(1,5),F(1,5)),(F(1,5),F(4,5)),(F(1,2),F(1,2)),(F(0),F(1))]:
   total=F(0)
   for trace in itertools.product(range(4),repeat=6):
    t=UnionVarianceTest(F(1,4)); w=F(1)
    for z in trace:
     a,b=divmod(z,2); w*=(p if a else 1-p)*(q if b else 1-q); t.update(a,b)
    if all(t.crossed): total+=w
   self.assertLessEqual(total,F(1,4))
 def test_branch_only(self):
  t=UnionVarianceTest()
  for _ in range(30): t.update(1,0)
  self.assertTrue(t.crossed[0]); self.assertFalse(t.crossed[1])
  for _ in range(30): t.update(1,1)
  self.assertTrue(all(t.crossed))
 def test_task_unbiased(self):
  for n in range(2,10):
   for p in [F(0),F(1,5),F(1,2),F(1)]:
    self.assertEqual(sum(F(k*(n-k),n*(n-1))*comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)),p*(1-p))
 def test_task_martingale_asymmetry(self):
  # x=+1 with prob 1/3, x=-1/2 with prob 2/3; E[x]=0 but no sign symmetry.
  t=TaskMeanTest(); u=TaskMeanTest(); t.update('a',1,2,0,2); u.update('a',0,2,1,4)
  # U(1,4)=1/4, thus x=-1/2.
  self.assertEqual([a/3+2*b/3 for a,b in zip(t.wealth,u.wealth)],[F(1)]*len(t.bets))
 def test_duplicate_invalid(self):
  t=TaskMeanTest(); t.update('a',0,2,0,2)
  with self.assertRaises(ValueError): t.update('a',0,2,0,2)
  with self.assertRaises(ValueError): t.update('b',0,1,0,2)
 def test_task_signal(self):
  t=TaskMeanTest()
  for i in range(30): t.update(i,1,2,0,2)
  self.assertTrue(t.crossed)
if __name__=='__main__': unittest.main()
