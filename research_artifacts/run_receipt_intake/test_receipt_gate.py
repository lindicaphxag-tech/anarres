import copy
import json
import tempfile
import unittest
from pathlib import Path
from receipt_gate import intake, digest

class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.panel=[f'item-{i}' for i in range(300)]
        self.c={'schema':'matched-run-intake.v1','model_revision':'1'*40,'dataset_revision':'2'*40,'trainer_commit':'3'*40,
                'train_configs':{'control':self.put('clean.json',{'fn':0}), 'treatment':self.put('noise.json',{'fn':.38})},
                'eval_config':self.put('eval.json',{'mode':'greedy'}),'extractor':self.put('scorer.txt',b'do not execute'),
                'panel':self.put('panel.json',self.panel),'pairs':[dict(seed=i,eval_seed=100+i,perturbation_seed=200+i) for i in range(9)]}
        self.cp=self.put('contract.json',self.c)['sha256']; self.receipts=[]
        for s in range(9):
            for arm in ('control','treatment'):
                tag=f'{s}-{arm}'; preds=[dict(item_id=i,prediction=f'{tag}-{i}') for i in self.panel]
                outs=[dict(item_id=r['item_id'],prediction_sha256=digest(r['prediction'].encode()),correct=j<150+(s-4 if arm=='treatment' else 0)) for j,r in enumerate(preds)]
                self.receipts.append(dict(contract_sha256=self.cp,seed=s,arm=arm,eval_seed=100+s,perturbation_seed=200+s if arm=='treatment' else None,run_id=tag,status='completed',train_config_sha256=self.c['train_configs'][arm]['sha256'],checkpoint=self.put(tag+'.ckpt',tag.encode()),train_log=self.put(tag+'.log',b'synthetic only'),predictions=self.put(tag+'-pred.json',preds),outcomes=self.put(tag+'-out.json',outs)))
        self.mp=self.freeze_receipts()
    def tearDown(self): self.tmp.cleanup()
    def put(self,name,obj):
        data=obj if isinstance(obj,bytes) else json.dumps(obj,sort_keys=True).encode(); (self.root/name).write_bytes(data)
        return dict(path=name,sha256=digest(data),size_bytes=len(data))
    def freeze_receipts(self):
        refs=[self.put(f'receipt{i}.json',r) for i,r in enumerate(self.receipts)]
        return self.put('manifest.json',dict(schema='matched-run-receipts.v1',contract_sha256=self.cp,receipts=refs))['sha256']
    def verify(self): return intake(self.root,self.cp,self.mp)
    def mutate(self,key,value):
        self.receipts[0][key]=value; self.mp=self.freeze_receipts()
    def test_reconstruction(self):
        r=self.verify(); self.assertEqual(r['n_runs'],18); self.assertEqual([x['correct_treatment'] for x in r['rows']],list(range(146,155)))
        for row,audit in zip(r['rows'],r['paired_item_audit']):
            self.assertEqual(audit['both_correct']+audit['control_only'],row['correct_control']); self.assertEqual(audit['both_correct']+audit['treatment_only'],row['correct_treatment']); self.assertEqual(sum(v for k,v in audit.items() if k!='seed'),300)
    def test_bad_external_pin(self):
        with self.assertRaises(ValueError): intake(self.root,'a'*64,self.mp)
    def test_changed_outcomes_bytes(self):
        (self.root/self.receipts[0]['outcomes']['path']).write_text('[]')
        with self.assertRaises(ValueError): self.verify()
    def test_reused_run(self):
        self.mutate('run_id',self.receipts[1]['run_id'])
        with self.assertRaises(ValueError): self.verify()
    def test_unfinished(self):
        self.mutate('status','failed')
        with self.assertRaises(ValueError): self.verify()
    def test_wrong_seed(self):
        self.mutate('eval_seed',999)
        with self.assertRaises(ValueError): self.verify()
    def test_wrong_config(self):
        self.mutate('train_config_sha256',self.c['train_configs']['treatment']['sha256'])
        with self.assertRaises(ValueError): self.verify()
    def test_missing_receipt(self):
        self.receipts.pop(); self.mp=self.freeze_receipts()
        with self.assertRaises(ValueError): self.verify()
    def test_reused_slot(self):
        self.receipts[1]=copy.deepcopy(self.receipts[0]); self.receipts[1]['run_id']='other'; self.mp=self.freeze_receipts()
        with self.assertRaises(ValueError): self.verify()
    def test_duplicate_item(self):
        r=self.receipts[0]; data=json.loads((self.root/r['outcomes']['path']).read_bytes()); data[-1]=data[0]; self.mutate('outcomes',self.put('duplicate.json',data))
        with self.assertRaises(ValueError): self.verify()
    def test_unbound_score(self):
        r=self.receipts[0]; data=json.loads((self.root/r['outcomes']['path']).read_bytes()); data[0]['prediction_sha256']='0'*64; self.mutate('outcomes',self.put('wrong.json',data))
        with self.assertRaises(ValueError): self.verify()
    def test_boolean_seed(self):
        self.mutate('seed',False)
        with self.assertRaises(ValueError): self.verify()
    def test_symlink(self):
        r=self.receipts[0]['checkpoint']; (self.root/'link').symlink_to(self.root/r['path']); self.mutate('checkpoint',{**r,'path':'link'})
        with self.assertRaises(ValueError): self.verify()
    def test_aggregate_only_rejected(self):
        self.mutate('outcomes',self.put('aggregate.json',{'correct':150}))
        with self.assertRaises(ValueError): self.verify()
    def test_path_escape(self):
        self.mutate('checkpoint',{**self.receipts[0]['checkpoint'],'path':'../outside'})
        with self.assertRaises(ValueError): self.verify()
    def test_duplicate_json_key(self):
        self.mutate('outcomes',self.put('duplicate-key.json',b'[{"item_id":"a","item_id":"b"}]'))
        with self.assertRaises(ValueError): self.verify()
    def test_checkpoint_tamper(self):
        p=self.root/self.receipts[0]['checkpoint']['path']; p.write_bytes(b'x'*p.stat().st_size)
        with self.assertRaises(ValueError): self.verify()
    def test_reused_artifact_path(self):
        self.mutate('checkpoint',self.receipts[1]['checkpoint'])
        with self.assertRaises(ValueError): self.verify()
    def test_duplicate_checkpoint_bytes_flagged(self):
        r=self.receipts[1]['checkpoint']; self.mutate('checkpoint',self.put('duplicate.ckpt',(self.root/r['path']).read_bytes()))
        self.assertEqual(self.verify()['duplicate_checkpoint_digest_groups'],[['0-control','0-treatment']])
    def test_rehashed_manifest_rejected_by_original_pin(self):
        pin=self.mp; self.mutate('status','failed')
        with self.assertRaises(ValueError): intake(self.root,self.cp,pin)

if __name__=='__main__': unittest.main()
