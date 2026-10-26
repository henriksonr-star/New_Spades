# Spades training and controlled evaluation

This repository preserves earlier Kaggle/Colab work, the verified checkpoint 17,574 training source/configuration snapshot, and earlier controlled evaluation results.

## Latest verified training: checkpoint 17,574

- Continued from **15,247** for **2,327 updates**, reaching **8,062,107,648 environment steps**.
- Measured **2 h 56 m 3.749 s active training**, **236.251 seconds short** of the three-hour target because the unchanged reservation guard stopped before another collect/optimize pair.
- Two coordinated T4 GPUs, **3,584 global environments / 1,792 per GPU**, teacher-free objective, learning rate **0.0002**, and **845 scripted opponent styles** plus self-play/snapshots.
- Healthy terminal content audit, all **214 finite native leaves**, preserved source history, and reported successful accepted-state readback.
- **No new playing-strength claim or audited checkpoint-17,574 evaluation result is included.**

See [verified training facts and checkpoint availability](results/update17574/README.md), [machine-readable summary](results/update17574/summary.json), and [the current source snapshot](training/teacher_free_dualt4_v1/README.md).

The model binary remains in the owner's verified backup; no public checkpoint download is published here. The public snapshot excludes private operational authorizations, journals and account data and is not a ready-to-run training launcher.

## Verified checkpoint8351 result

- Active training: 10 h 51 m 10.693 s, 8,351 updates.
- Training mixture: 50% self-play, 25% saved-policy snapshots, 25% scripted opponents; 237 scripted parameter variants, not 237 independent strategies.
- Held-out evaluation: 64 variants, 4,096 completed games (2,048 per checkpoint). Win rate improved from 78.7598% to 81.8848%, a +3.125 percentage-point difference. Nominal 95% paired bootstrap interval: +1.4148 to +4.7852 points.
- Development evaluation: 16 variants, 1,024 games (512 per checkpoint), 66.2109% to 66.9922%. Difference interval: -2.5391 to +4.2969 points, not a clear improvement.
- All 5,120 games finished with finite outcomes and zero illegal actions.

See [result details](results/control1329_vs_8351/README.md), [machine-readable summary](results/control1329_vs_8351/summary.json), and [game-level outcomes](results/control1329_vs_8351/outcomes.csv).

## Source layout and important limits

- `training/teacher_free_dualt4_v1/`: latest teacher-free dual-T4 source/configuration snapshot, documented excerpts and source checks.
- `training/normal_lr_v3/source/`: unchanged, reviewed source modules and opponent manifests from the checkpoint8351 training lineage.
- `evaluation/controlled_v1/`: controlled-evaluation runtime, frozen evaluation kernel and source hashes; two descriptive protocol metadata fields have been sanitized to remove historical machine paths.
- Existing `notebooks/`, `runtime/`, `tests/` and `tools/` remain historical material; they are not updated launch instructions for checkpoint8351.

**This new source snapshot is not a turnkey runnable distribution.** Native checkpoints, account-specific launch/approval receipts, private provenance metadata, and source files containing private artifact references are intentionally not published. Some modules therefore reference omitted inputs. Pinned hashes record original reviewed inputs, not substituted or sanitized files. Do not disable integrity checks or invent receipts to run this snapshot. A sanitized, re-pinned and tested distribution is separate work.

The original evaluation plan retains its pre-execution status because it is the frozen protocol, not the result report. Results are in the directory above. The latest training snapshot is separately documented above. Earlier evaluation results are preserved and are not results for checkpoint 17,574.

## Validation scope

The historical checkpoint8351 evaluation evidence passed an independent content audit. Its runtime records report a Kaggle Tesla T4; this is not independent hardware attestation. Its upstream package underwent host tests and a synthetic CPU kernel smoke test before that evaluation. Those historical tests are separate from the latest training publication.

The checkpoint17574 publication passed **183 host-only checks** (168 repository distribution/runtime checks and 15 source-snapshot checks); see [publication checks](results/update17574/publication_checks.json). No training, checkpoint gameplay or new GPU run was performed for publication. The latest terminal content audit and the public source checks have distinct scopes.

Credentials, private account details, checkpoint weights, campaign journals, issued execution receipts and large output archives are excluded. Existing repository content is preserved.

Two historical protocol description fields were privacy-filtered without changing checkpoint identities, rules, schedules, seeds, pairing, statistics or outcomes; [sanitization provenance](docs/publication_metadata_sanitization.json) records both file hashes. Historical runtime pins remain original and must not be disabled to run a partial snapshot.
