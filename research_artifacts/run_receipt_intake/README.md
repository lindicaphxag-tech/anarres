# Matched training receipt intake

A standalone supplementary research tool for Anarres PR #82, based on its
frozen feasibility design at 584efdc1414203c1ce430e5813c0e05e32cf105a.
It does not change the original protocol, manifest, historical aggregate
counts, or scientific conclusions.

## Evidence contract

The input directory contains contract.json and manifest.json. Obtain their
expected SHA-256 digests through an independent trusted record. Freeze the
contract before training; freeze the receipt manifest after collection.
Do not use a digest supplied only by the same untrusted directory as proof
of authenticity.

Every file reference has path, sha256, and size_bytes fields.
Contract schema matched-run-intake.v1 pins the model revision, dataset
revision, trainer commit, control and treatment config files, evaluation
config, extraction code, the list of 300 unique evaluation item IDs, and
nine distinct training seed pairs with evaluation and perturbation seeds.

Manifest schema matched-run-receipts.v1 binds the contract digest and
18 receipt file references. Each receipt binds the contract, seed, arm,
evaluation seed, perturbation seed (null for control), run identity,
completion status, training configuration digest, checkpoint archive,
training log, raw predictions, and scored item outcomes. Predictions
contain item_id and prediction. Outcomes contain item_id,
prediction_sha256 (UTF-8 raw prediction), and boolean correct.

The intake verifies bytes, size, references, complete panel membership and
unique run slots. It reconstructs the nine run-level rows and the four
paired item count cells, rather than accepting aggregate claims. Duplicate
artifact paths fail. Distinct checkpoint files with the same digest are
reported for review, not automatically treated as misconduct.

Use a stable read-only input snapshot under operator control. The tool
does not execute the extraction code or deserialize model checkpoints.
It cannot establish whether a training job actually executed, whether the
pinned configuration was actually used, whether a score is correct,
whether runs are statistically independent, or whether a supplied revision
exists. Independently authenticate the training receipts, inspect the
configs, and reproduce scoring in a trusted environment. The code is
not a sandbox for a concurrently mutating filesystem.

## Run

    python -m unittest -v
    python receipt_gate.py /path/to/frozen-run-bundle \
        --contract-sha256 EXPECTED_PRE_RUN_DIGEST \
        --manifest-sha256 EXPECTED_POST_RUN_DIGEST \
        --csv /path/outside/bundle/matched.csv

The CSV fields match the existing PR #82 resolution_plan.py --runs
input. Its nine pairs are the experimental units. The 300 evaluation
items per run are not independent training replicates.

Local Python 3.13.5: 20/20 tests passed, including one 18-run synthetic
fixture with 5,400 item outcomes and deliberately corrupted variants.
No real GPU training, historical data recovery, or full upstream
make check / make study-check is claimed.
