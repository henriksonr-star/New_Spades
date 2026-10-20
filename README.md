# Spades training and controlled evaluation

This repository preserves earlier Kaggle/Colab work and adds the verified normal-learning-rate training source snapshot and CONTROL1329-versus-checkpoint8351 evaluation results.

## Verified checkpoint8351 result

- Active training: 10 h 51 m 10.693 s, 8,351 updates.
- Training mixture: 50% self-play, 25% saved-policy snapshots, 25% scripted opponents; 237 scripted parameter variants, not 237 independent strategies.
- Held-out evaluation: 64 variants, 4,096 completed games (2,048 per checkpoint). Win rate improved from 78.7598% to 81.8848%, a +3.125 percentage-point difference. Nominal 95% paired bootstrap interval: +1.4148 to +4.7852 points.
- Development evaluation: 16 variants, 1,024 games (512 per checkpoint), 66.2109% to 66.9922%. Difference interval: -2.5391 to +4.2969 points, not a clear improvement.
- All 5,120 games finished with finite outcomes and zero illegal actions.

See [result details](results/control1329_vs_8351/README.md), [machine-readable summary](results/control1329_vs_8351/summary.json), and [game-level outcomes](results/control1329_vs_8351/outcomes.csv).

## Source layout and important limits

- `training/normal_lr_v3/source/`: unchanged, reviewed source modules and opponent manifests from the checkpoint8351 training lineage.
- `evaluation/controlled_v1/`: unchanged controlled-evaluation runtime, frozen evaluation kernel, original protocol and source hashes.
- Existing `notebooks/`, `runtime/`, `tests/` and `tools/` remain historical material; they are not updated launch instructions for checkpoint8351.

**This new source snapshot is not a turnkey runnable distribution.** Native checkpoints, account-specific launch/approval receipts, private provenance metadata, and source files containing private artifact references are intentionally not published. Some modules therefore reference omitted inputs. Pinned hashes record original reviewed inputs, not substituted or sanitized files. Do not disable integrity checks or invent receipts to run this snapshot. A sanitized, re-pinned and tested distribution is separate work.

The original evaluation plan retains its pre-execution status because it is the frozen protocol, not the result report. Results are in the directory above. New opponent expansion and further training are not included or claimed complete here.

## Validation scope

The actual result evidence passed an independent content audit. Runtime records report a Kaggle Tesla T4; this is not independent hardware attestation. The upstream evaluation package underwent host tests and a synthetic CPU kernel smoke test before the actual run. This publication is artifact curation: those tests were not rerun against this intentionally incomplete public source snapshot, and no new GPU run was performed for publication.

Credentials, private account details, checkpoint weights, campaign journals, issued execution receipts and large output archives are excluded. Existing repository content is preserved.
