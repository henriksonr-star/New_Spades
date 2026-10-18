# Sanitized source distribution

Operational release files are preserved outside this source tree. This mirror removes notebook-level and cell metadata, outputs, execution counts, private external file identities, and machine-specific build/test paths. All 38 cell source fields in each reference notebook are preserved exactly. The Kaggle notebook intentionally differs from the Colab safety notebook only in cell12, the selected-board telemetry adapter.

Whole-file SHA-256 values and byte lengths change after metadata removal. Each distribution operator, auditor, builder and coordination template is rebound to the matching cleaned notebook identities. The provenance manifest records both original and distribution hashes. Do not mix operational and distribution payload members or receipts.

The original private receipt label is replaced with the literal `LOCAL_VERIFIED_UPDATE674`. This is a local alias for an external verification record; it is not proof of an upload. Before setting `candidate_native_export_verified:true`, the coordinator must independently verify the exact674 package against the retained external record and its fixed SHA-256. Preserve that private record separately. The operator still rejects any other initial receipt label, changed package hash, native schema, saved configuration, runtime stack or counters.

Later receipt checks retain nonce, package/native byte lengths, hashes, exact counters, expiry and nonempty external receipt requirements. The host content auditor always reports that it has not verified a Library upload or authorized an ACK. An actual external upload and independent downloaded-byte audit are separate required steps.

The repository contains no checkpoint binaries, account credentials, browser files, runtime logs, private coordination data, or private external file IDs. Generated payloads and runtime artifacts are ignored. Synthetic test receipts are explicitly labeled `TEST_EXTERNAL_RECEIPT`.
