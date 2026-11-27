import json
from relaxation import estimate
from benchmark import CASES

if __name__=='__main__':
    print(json.dumps({name:estimate(c,l,b,a,reps=1200,delta=.01)
                     for name,c,l,b,a,_ in CASES},indent=2))
