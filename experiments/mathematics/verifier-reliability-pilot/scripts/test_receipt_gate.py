import tempfile,json,unittest
from pathlib import Path
from receipt_gate import validate,digest

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.panel=[str(i) for i in range(300)]
        def write(name,obj):
            p=self.root/name;p.write_text(json.dumps(obj));return {'path':name,'sha256':digest(p)}
        self.write=write
        self.c=dict(model_revision='a'*40,dataset_revision='b'*40,trainer_commit='c'*40,training_seeds=list(range(9)),evaluation_seed=10,verifier={'control_fn':0,'treatment_fn':.38,'flip_unit':'iid_rollout'},perturbation_seeds={str(i):100+i for i in range(9)},panel=write('panel.json',self.panel),common_config=write('config.json',{'synthetic':True}),extractor=write('extractor.json',{'synthetic':True}))
        self.commit=write('contract.json',self.c)['sha256'];self.rs=[]
        for s in range(9):
            for a in ('control','treatment'):
                run=f'{s}-{a}';self.rs.append(dict(run_id=run,training_seed=s,arm=a,contract_sha256=self.commit,evaluation_seed=10,status='completed',perturbation_seed=100+s if a=='treatment' else None,checkpoint=write(run+'.ckpt',run),item_outcomes=write(run+'.json',[{'item_id':i,'correct':int(i)<(150+s if a=='treatment' else 150)} for i in self.panel])))
        write('receipts.json',self.rs)
    def tearDown(self):self.tmp.cleanup()
    def test_valid(self):
        z=validate(self.root,self.commit);self.assertEqual(len(z),9);self.assertEqual(z[-1]['correct_treatment'],'158')
    def test_tamper(self):
        (self.root/self.rs[0]['item_outcomes']['path']).write_text('[]')
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_duplicate(self):
        self.rs[1]['run_id']=self.rs[0]['run_id'];self.write('receipts.json',self.rs)
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_reused(self):
        self.rs[1]['checkpoint']=self.rs[0]['checkpoint'];self.write('receipts.json',self.rs)
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_wrong_panel(self):
        r=self.rs[0];r['item_outcomes']=self.write(r['item_outcomes']['path'],[{'item_id':i,'correct':True} for i in reversed(self.panel)]);self.write('receipts.json',self.rs)
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_wrong_seed(self):
        self.rs[1]['perturbation_seed']=999;self.write('receipts.json',self.rs)
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_missing(self):
        self.write('receipts.json',self.rs[:-1])
        with self.assertRaises(ValueError):validate(self.root,self.commit)
    def test_contract(self):
        with self.assertRaises(ValueError):validate(self.root,'0'*64)
if __name__=='__main__':unittest.main()
