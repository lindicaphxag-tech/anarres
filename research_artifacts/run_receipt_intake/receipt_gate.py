"""Read-only, externally pinned receipt intake for nine matched training pairs."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def sha(s):
    return isinstance(s, str) and re.fullmatch(r'[0-9a-f]{64}', s) is not None


def revision(s):
    return isinstance(s, str) and re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', s) is not None


def keys(obj, fields):
    require(isinstance(obj, dict) and set(obj) == set(fields.split()), 'unexpected or missing fields')


def unique_object(pairs):
    obj = {}
    for k, v in pairs:
        require(k not in obj, 'duplicate JSON key')
        obj[k] = v
    return obj


def parse(data):
    return json.loads(data, object_pairs_hook=unique_object,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def artifact(root, ref, *, read=True):
    keys(ref, 'path sha256 size_bytes')
    name, h, size = ref['path'], ref['sha256'], ref['size_bytes']
    require(isinstance(name, str) and name and '\\' not in name, 'invalid artifact path')
    rel = Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.parts, 'escaping artifact path')
    p = root
    for component in rel.parts:
        p = p / component
        require(not p.is_symlink(), 'symlink artifact')
    require(p.is_file() and sha(h) and type(size) is int and size >= 0, 'invalid artifact reference')
    require(p.stat().st_size == size, 'artifact size mismatch')
    if read:
        data = p.read_bytes()
        require(len(data) == size and digest(data) == h, 'artifact digest mismatch')
        return data
    actual = hashlib.sha256()
    length = 0
    with p.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            actual.update(chunk)
            length += len(chunk)
    require(length == size and actual.hexdigest() == h, 'artifact digest mismatch')


def pinned_json(root, name, expected):
    require(sha(expected), 'external SHA-256 pin required')
    p = root / name
    require(p.is_file() and not p.is_symlink(), 'missing pinned file')
    data = p.read_bytes()
    require(digest(data) == expected, 'external pin mismatch')
    return parse(data)


def nonnegative_int(v):
    return type(v) is int and v >= 0


def intake(root, contract_pin, manifest_pin):
    """Validate bytes and reconstruct counts, not authenticity or scorer correctness."""
    root = Path(root).resolve(strict=True)
    c = pinned_json(root, 'contract.json', contract_pin)
    keys(c, 'schema model_revision dataset_revision trainer_commit train_configs eval_config extractor panel pairs')
    require(c['schema'] == 'matched-run-intake.v1', 'invalid schema')
    require(all(revision(c[k]) for k in ('model_revision', 'dataset_revision', 'trainer_commit')), 'unpinned source revision')
    keys(c['train_configs'], 'control treatment')
    for ref in [*c['train_configs'].values(), c['eval_config'], c['extractor']]:
        artifact(root, ref, read=False)
    panel = parse(artifact(root, c['panel']))
    require(isinstance(panel, list) and len(panel) == 300 and all(isinstance(x, str) and x for x in panel), 'invalid panel')
    require(len(set(panel)) == 300, 'duplicate panel item')
    ids = set(panel)
    pairs = c['pairs']
    require(isinstance(pairs, list) and len(pairs) == 9, 'nine pairs required')
    planned, perturb = {}, set()
    for p in pairs:
        keys(p, 'seed eval_seed perturbation_seed')
        s = p['seed']
        require(all(nonnegative_int(p[k]) for k in ('seed','eval_seed','perturbation_seed')), 'invalid seed')
        require(s not in planned and p['perturbation_seed'] not in perturb, 'duplicate planned seed')
        planned[s] = p
        perturb.add(p['perturbation_seed'])
    m = pinned_json(root, 'manifest.json', manifest_pin)
    keys(m, 'schema contract_sha256 receipts')
    require(m['schema'] == 'matched-run-receipts.v1' and m['contract_sha256'] == contract_pin, 'manifest contract mismatch')
    require(isinstance(m['receipts'], list) and len(m['receipts']) == 18, 'eighteen receipts required')
    seen_runs, slots, seen_paths, checkpoints = set(), {}, set(), {}
    for ref in m['receipts']:
        r = parse(artifact(root, ref))
        keys(r, 'contract_sha256 seed arm eval_seed perturbation_seed run_id status train_config_sha256 checkpoint train_log predictions outcomes')
        s, arm, rid = r['seed'], r['arm'], r['run_id']
        require(nonnegative_int(s) and s in planned and arm in ('control','treatment'), 'unplanned slot')
        require(r['contract_sha256'] == contract_pin and r['status'] == 'completed', 'incomplete or foreign run')
        require(nonnegative_int(r['eval_seed']) and r['eval_seed'] == planned[s]['eval_seed'], 'evaluation seed mismatch')
        expected = planned[s]['perturbation_seed'] if arm == 'treatment' else None
        require(r['perturbation_seed'] == expected and (expected is None or nonnegative_int(r['perturbation_seed'])), 'perturbation seed mismatch')
        require(r['train_config_sha256'] == c['train_configs'][arm]['sha256'], 'training configuration mismatch')
        require(isinstance(rid, str) and rid.strip() == rid and rid and rid not in seen_runs and (s,arm) not in slots, 'reused run or slot')
        seen_runs.add(rid)
        for role in ('checkpoint', 'train_log', 'predictions', 'outcomes'):
            keys(r[role], 'path sha256 size_bytes')
            path = r[role]['path']
            require(isinstance(path, str), 'invalid artifact path')
            normalized = Path(path)
            require(normalized not in seen_paths, 'reused run artifact path')
            seen_paths.add(normalized)
        artifact(root, r['checkpoint'], read=False)
        artifact(root, r['train_log'], read=False)
        checkpoints.setdefault(r['checkpoint']['sha256'], []).append(rid)
        preds = parse(artifact(root, r['predictions']))
        outcomes = parse(artifact(root, r['outcomes']))
        require(isinstance(preds, list) and len(preds) == 300 and isinstance(outcomes, list) and len(outcomes) == 300, 'invalid item counts')
        pred_map = {}
        for row in preds:
            keys(row, 'item_id prediction')
            i = row['item_id']
            require(isinstance(i, str) and i in ids and i not in pred_map and isinstance(row['prediction'], str), 'invalid prediction item')
            pred_map[i] = digest(row['prediction'].encode('utf-8'))
        correct = {}
        for row in outcomes:
            keys(row, 'item_id prediction_sha256 correct')
            i = row['item_id']
            require(isinstance(i, str) and i in pred_map and i not in correct, 'invalid outcome item')
            require(row['prediction_sha256'] == pred_map[i] and type(row['correct']) is bool, 'unbound or invalid outcome')
            correct[i] = row['correct']
        require(set(correct) == ids, 'incomplete panel')
        slots[s,arm] = rid, correct
    require(len(slots) == 18, 'missing planned slot')
    rows, audits = [], []
    for s in sorted(planned):
        a, ac = slots[s,'control']; b, bc = slots[s,'treatment']
        rows.append(dict(seed=s, control_run_id=a, treatment_run_id=b, panel_sha256=c['panel']['sha256'],
                         n_items=300, correct_control=sum(ac.values()), correct_treatment=sum(bc.values())))
        audits.append(dict(seed=s, both_correct=sum(ac[i] and bc[i] for i in ids),
                           control_only=sum(ac[i] and not bc[i] for i in ids),
                           treatment_only=sum(bc[i] and not ac[i] for i in ids),
                           neither=sum(not ac[i] and not bc[i] for i in ids)))
    return {'status':'integrity_checked_not_execution_authenticated', 'n_pairs':9, 'n_runs':18,
            'rows':rows, 'paired_item_audit':audits, 'contract_sha256':contract_pin, 'manifest_sha256':manifest_pin,
            'duplicate_checkpoint_digest_groups':[v for v in checkpoints.values() if len(v)>1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('--contract-sha256',required=True); p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--csv', type=Path, help='optional new output CSV (outside input bundle)')
    a=p.parse_args(); result=intake(a.root,a.contract_sha256,a.manifest_sha256)
    if a.csv:
        require(not a.csv.resolve().is_relative_to(a.root.resolve()), 'output must be outside input bundle')
        with a.csv.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['rows'][0])); w.writeheader(); w.writerows(result['rows'])
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
