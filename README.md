# Spades AI

JAX self-play training for the modified partnership Spades rules in the notebooks. The project retains the exact game engine, information boundaries, 13 development opponents, saved full training configuration, and native checkpoint schema. Training segments last three hours with 90-minute checkpoints. The continuation operators add supplemental 15-minute exports and stop at every export until an externally verified receipt is acknowledged.

This repository is a sanitized source distribution. Notebook cell source is preserved, but outputs and private metadata were removed. Source hash pins were rebound to those cleaned notebooks, and the private initial receipt identity was replaced with `LOCAL_VERIFIED_UPDATE674`. These files are **not byte-identical to the operational release artifacts**. See [provenance](docs/source_provenance.json) and [distribution notes](docs/DISTRIBUTION.md).

## Current validation status

| Route | Evidence and limits |
|---|---|
| Earlier Colab AOT V1 | Actual T4 training reached observed update 1075 / 493,158,400 decisions. The latest independently retained checkpoint is update 674 / 309,198,848 decisions; later work was lost on disconnection. |
| Colab Candidate3 | Host-reviewed continuation and export handling. Physical QA was blocked when Colab denied GPU service due to usage limits. No Candidate3 GPU validation is claimed. |
| Kaggle adapter V1 | Experimental platform migration. Host tests check inventory, one selected T4, UUID-bound telemetry, source preservation and export validation. Actual provider runtime, checkpoint continuation, numerical behavior and durability still require physical QA. |

The original combined-JIT continuation comparison remains unresolved: an unmodified T4 replay differed in 125 floating-point state leaves, by at most `1.1920928955078125e-7`. Integer state and the separately captured metrics matched. Strict acceptance has not been relaxed. The explicit sequential AOT route does not establish that the original JIT issue is fixed.

Longer training has not received a playing-strength evaluation. A previous candidate37 paired evaluation covered four development styles and one PPO seed; it does not establish expert strength, complete held-out coverage, or the strength of checkpoint674. No checkpoint is automatically promoted.

## Source layout

- `notebooks/source_recovered.ipynb`: recovered original cell source, with reconstructed notebook metadata removed.
- `notebooks/runtime_safety_v2.ipynb`: checkpoint and memory-safety derivative. Its rules/opponents/learning cells match the recovered source.
- `notebooks/runtime_kaggle_v1.ipynb`: narrow platform derivative; only the memory telemetry cell changes from runtime safety V2.
- `runtime/colab_c3/`: continuation operator, notebook builder, content auditor and host tests.
- `runtime/kaggle_v1/`: experimental selected-board adapter, corresponding builder/auditor and host tests.
- `tests/support/`: pure production journal helpers and synthetic host state used by tests.
- `tools/`: local host verification and payload construction. These tools do not allocate a GPU.

## Host checks

Use Python 3.12 or newer and install the host test dependency in a local environment:

```bash
python -m pip install -r requirements-host.txt
python tools/run_host_checks.py
```

The checks use synthetic NumPy state and mocked hardware. They do not load the private checkpoint, start JAX training, prove GPU memory capacity, or verify an external upload.

The continuation runtime requires JAX/JAXlib `0.11.1`, Flax `0.12.10` and Optax `0.2.8`. The Kaggle route additionally requires matching CUDA12 plugin/PJRT `0.11.1`, a supported NVIDIA driver, and exactly one selected JAX-visible T4. Its inventory gate rejects a changed stack or ambiguous board mapping; no installer, CPU fallback, configuration reduction, or additional-GPU memory pooling is provided.

## Preparing a reviewed runtime

Read [runtime gates](docs/RUNTIME_GATES.md) first. The checkpoint is intentionally excluded. The exact external update674 package SHA-256 is:

```text
2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13
```

Construct a **source-distribution** payload and separate unexecuted notebook in a new output directory:

```bash
python tools/build_payload.py --provider kaggle --output-dir dist/kaggle-review
```

Use `--provider colab` for the host-reviewed C3 distribution. Building a payload does not authorize or start a run. All coordination review gates remain false, no private checkpoint is embedded, and time-limited launch gates still apply. Do not run all cells in the source/reference notebooks to bypass the operator.

There is no automatic fresh-VM recovery or platform-limit bypass. If the provider ends service, stop and retain/report the latest externally verified checkpoint. Free service only; no paid resources or account changes are required by the code.

No license has been selected for this project.
