"""Validate a prospective paired-run receipt bundle; no historical data inferred."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    return json.loads(path.read_text(encoding='utf8'), object_pairs_hook=pairs)


def hexid(x, n=64):
    return isinstance(x, str) and re.fullmatch('[0-9a-f]{' + str(n) + '}', x) is not None


def artifact(root, ref):
    if set(ref) != {'path', 'sha256'} or not hexid(ref['sha256']):
        raise ValueError('invalid artifact reference')
    p = (root / ref['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file() or digest(p) != ref['sha256']:
        raise ValueError('artifact missing, escaped root, or hash mismatch')
    return p


def validate(root, commitment):
    """Commitment must be independently published before training, not inferred here."""
    root = Path(root).resolve()
    p = root / 'contract.json'
    if not hexid(commitment) or digest(p) != commitment:
        raise ValueError('contract commitment mismatch')
    c = load(p)
    for key in ('model_revision', 'dataset_revision', 'trainer_commit'):
        if not hexid(c[key], 40):
            raise ValueError('nonimmutable revision')
    seeds = c['training_seeds']
    if len(seeds) != 9 or len(set(seeds)) != 9 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('nine distinct precommitted seeds required')
    if type(c['evaluation_seed']) is not int or c['evaluation_seed'] < 0:
        raise ValueError('invalid evaluation seed')
    if c['verifier'] != {'control_fn': 0, 'treatment_fn': 0.38, 'flip_unit': 'iid_rollout'}:
        raise ValueError('verifier contract mismatch')
    perturb = c['perturbation_seeds']
    if set(perturb) != set(map(str, seeds)) or any(type(v) is not int or v < 0 for v in perturb.values()) or len(set(perturb.values())) != 9:
        raise ValueError('precommit distinct perturbation seeds')
    panel = load(artifact(root, c['panel']))
    if not isinstance(panel, list) or len(panel) != 300 or any(not isinstance(x,str) or not x for x in panel) or len(set(panel)) != 300:
        raise ValueError('panel must contain 300 distinct ordered IDs')
    for key in ('common_config','extractor'):
        artifact(root,c[key])
    receipts = load(root / 'receipts.json')
    if not isinstance(receipts,list) or len(receipts) != 18:
        raise ValueError('18 completed run receipts required')
    ids, pairs, checkpoint_paths, outcome_paths = set(), {}, set(), set()
    for r in receipts:
        s, arm, run = r['training_seed'], r['arm'], r['run_id']
        if type(s) is not int or s not in seeds or arm not in ('control','treatment') or not isinstance(run,str) or not run or run in ids or (s,arm) in pairs:
            raise ValueError('duplicate or unplanned run')
        if r['contract_sha256'] != commitment or r['evaluation_seed'] != c['evaluation_seed'] or r['status'] != 'completed':
            raise ValueError('incompatible or incomplete run')
        if r['perturbation_seed'] != (perturb[str(s)] if arm == 'treatment' else None):
            raise ValueError('perturbation seed mismatch')
        checkpoint = artifact(root,r['checkpoint']); output = artifact(root,r['item_outcomes'])
        if checkpoint in checkpoint_paths or output in outcome_paths:
            raise ValueError('reused artifact path')
        checkpoint_paths.add(checkpoint);outcome_paths.add(output)
        rows = load(output)
        if not isinstance(rows,list) or len(rows)!=300 or any(set(x)!= {'item_id','correct'} or type(x['correct']) is not bool for x in rows) or [x['item_id'] for x in rows]!=panel:
            raise ValueError('item outcomes do not match fixed ordered panel')
        ids.add(run);pairs[s,arm] = (run, [x['correct'] for x in rows])
    result=[]
    for s in seeds:
        a,x=pairs[s,'control'];b,y=pairs[s,'treatment']
        result.append(dict(seed=str(s),control_run_id=a,treatment_run_id=b,panel_sha256=c['panel']['sha256'],n_items='300',correct_control=str(sum(x)),correct_treatment=str(sum(y))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('--commitment',required=True)
    args=p.parse_args();print(json.dumps(validate(args.bundle,args.commitment),indent=2))


if __name__=='__main__':
    main()
