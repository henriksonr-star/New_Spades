# Teacher-free checkpoint17574 source snapshot

This is the public algorithm source snapshot used by the continuation from update15247 to update17574. It includes the teacher-free PPO objective, the 845-variant scripted training population, two-device GSPMD coordination, sequential blind-nil visibility, passive diagnostics, and selected checkpoint readback guards.

**This is an intentionally incomplete source snapshot, not a turnkey training distribution.** Native checkpoints, bound operators, issued execution inputs, private provenance maps, campaign journals, account data, full notebooks and packaged launch frontends are excluded. Do not weaken the retained source pins, substitute the excerpts for pinned notebooks, or invent operational inputs to run it. Checkpoint availability and audited run facts belong in the repository's result report, not in the source examples here. No playing-strength result is claimed by this snapshot.

## What is included

| Location | Meaning |
| --- | --- |
| `source/frontend/prior9971/teacher_free_objective.py` | Exact objective transformation and installation: clipped PPO policy loss plus weighted value loss minus legal-action entropy. Teacher targets, teacher losses and supervised bootstrap are removed or disabled. |
| `source/frontend/prior9971/dual_gpu_runtime.py` | Exact coordination for one global learner on two T4s: partitioned environment/sample axes, replicated model/Adam/RNG/counters, collective and shard checks, and per-device memory admission. |
| `source/frontend/prior9971/*opponents*.py`, `source/opponents/` | Exact inherited v2/v3/v4 policies and novel v5 expansion. There are 845 training parameter variants, not 845 independent strategies; 256 excluded evaluation variants never enter the training sampler. |
| `source/runtime/novel845_profile.py`, `recovery12120_profile.py` | Exact profile adapters. Historical names remain intact; the continuation preserves the saved novel845 configuration. |
| `source/runtime/sequential_bn_rules.py`, `sequential_bn_visibility.py` | Exact combined rules and installation patch. Responses become public clockwise after each seat is polled. Own hands remain hidden during blind decisions. Policy/critic/action dimensions stay 753/1173/68. |
| `source/diagnostics/blind_diagnostics.py` | Exact passive diagnostics and source instrumentation. These counters introduce no reward, loss, RNG, or TrainingState changes. |
| `derived/base_opponents_cell8.py`, `base_learner_cell10.py` | Exact code cells extracted from the pinned production notebook. These are base definitions, not the final installed trainer. Cell10 retains historical teacher/bootstrap functions that the teacher-free installation disables or replaces. |
| `derived/teacher_free_make_train_step.py` | Exact output of the reviewed teacher-free transformation, before passive instrumentation. |
| `derived/instrumented_make_train_step.py` | Exact final generated collect/optimizer definition after passive diagnostics; requires its original installed namespace. |
| `derived/checkpoint_guard_excerpt.py` | Exact selected save/readback definitions, with only module documentation and standard-library imports added. They are inspection excerpts with omitted caller dependencies, not a new save runtime. |

The duplicate definitions in `sequential_bn_rules.py` are intentional: it contains the exact base rules followed by the reviewed visibility patch that replaces selected callables. The first rules-ready print precedes these replacements; it is not an execution or test receipt.

## Configuration and provenance

`training_configuration.json` reproduces the complete saved training configuration. Its canonical SHA256 (sorted keys, compact JSON separators) is `ca6e57b013d44bca93f87e03927e151b13535d61709cbc3ff836635785240824`. It was derived by applying the exact novel845 profile migration to the original source configuration and matching the recorded latest configuration hash. No fields were dropped to hide legacy settings; this schema contains no teacher/anchor/bootstrap configuration keys.

The learning rate is 0.0002, with 3,584 global environments, 128 rollout steps, three PPO epochs and a 57,344-sample global minibatch. Across two devices these are 1,792 environments and 28,672 minibatch samples per device. The outer mixture is 50% self-play, 25% saved-policy snapshots and 25% scripted opponents. Within scripted assignments, the retained333 and novel512 populations each receive half the probability. `hand_score_reward_coef` remains zero. These are configuration values, not claims that every planned minibatch was accepted.

`source_manifest.json` is the publication allowlist. Each entry records the logical original path, original byte count/SHA256, public byte count/SHA256, and the exact extraction or generation method. It also lists omitted runtime dependencies and the reason for each exclusion. All 290 files in the original frozen source map were checked during curation; the canonical map digest is `543090a679fe74e48755593eb73a003337979edf59363f5520bc5d293f87b471`. The private map itself is not published.

Seventeen complete source/manifest files are unchanged. Exact cell extractions and generated definitions retain all algorithm bytes. The checkpoint excerpt retains the exact original function ASTs, including decorators. The bound current operator's separate validation of operational checkpoint metadata is intentionally omitted with its private inputs; no claim is made that the excerpt reproduces that full operator.

## Host checks

From this directory, run:

```sh
python -m unittest discover -s tests -v
```

These checks verify public hashes and Python syntax, the exact configuration schema/hash, the final diagnostic source transformation, opponent roster consistency and exclusions, synthetic host diagnostic accounting, pure compiler-evidence parsing, and scoped checkpoint readback guard behavior. They use the Python standard library and NumPy. The NumPy opponent-ID checks are host logic checks, not JAX gameplay tests.

`verify_originals.py --source-root /path/to/original/frozen/package` additionally verifies all original closure members and every selected extraction/generation against the private package, if a reviewer already has it. This never requests or reconstructs the omitted inputs.

No training, checkpoint restore, gameplay, GPU compilation, two-T4 execution, or external publication is performed by these checks. `host_check_report.json` records the checks actually run for this snapshot; it is publication validation, not evidence of learning quality or hardware execution.
