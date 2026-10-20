# CONTROL1329 vs checkpoint8351

This is a common-opponent comparison, not head-to-head play. Each policy controls both teammates. Both policies share deal roots, opponent styles and team roles. Win credit is 1 for a win, 0.5 for a draw and 0 for a loss; there were no draws.

## Held-out primary assessment

64 training-excluded variants, 1,024 paired deal roots, 4,096 total games:

- CONTROL1329: 1,613 wins, 0 draws, 435 losses; 78.759765625%.
- Checkpoint8351: 1,677 wins, 0 draws, 371 losses; 81.884765625%.
- Candidate minus baseline: +3.125 percentage points.
- Nominal 95% stratified paired-root bootstrap interval: +1.414794921875 to +4.78515625 percentage points, using 10,000 resamples.

## Development assessment

16 variants, 256 paired roots, 1,024 total games:

- CONTROL1329: 339 wins, 0 draws, 173 losses; 66.2109375%.
- Checkpoint8351: 343 wins, 0 draws, 169 losses; 66.9921875%.
- Difference: +0.78125 percentage points; interval -2.5390625 to +4.296875 points.

Do not pool development and held-out results. The intervals condition on these checkpoints and fixed opponent styles; they do not include training-seed uncertainty or establish universal playing strength. Individual held-out families need not show clear improvement. After these results inform future selection or training, this opponent population is exposed; new deal seeds alone do not restore unseen-family status.

## Evidence and identity

`outcomes.csv` is the unchanged 5,120-row game-level output. `summary.json` is a curated copy of the verified summary with machine-specific GPU identifiers removed and training/scope notes added. All games completed, all outputs were finite, and no illegal actions occurred. There were 56,979 hands (5–47 per game).

Original CONTROL1329 native SHA-256: `e125a9a946bbd50185829cde5c4e577d4198d39b3ecede06da9aa44a77e496f6`.
Checkpoint8351 native SHA-256: `15c6ca425eee1298ddd50f01215b6be7011cab1031f0a62ff6cdbc5bd4ae0cde`.

Native inputs, private provider/job receipts, full device records and model weights are not included in Git. The summary's content-audit result is not an independent attestation of GPU hardware. Raw outcome statistics can be inspected without those files; rerunning the exact policy comparison requires the omitted native inputs and a complete reviewed runtime distribution.
