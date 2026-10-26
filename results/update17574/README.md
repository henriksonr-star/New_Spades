# Verified training update 17,574

Checkpoint **17,574** continues checkpoint **15,247**, adding **2,327 updates** and ending at **8,062,107,648 environment steps**. The completed run measured **10,563.749022841 active seconds (2 h 56 m 3.749 s)**. It finished **236.250977159 seconds short** of the requested three active hours because the unchanged dispatch guard reserves 240 seconds for another collect/optimize pair. Provider elapsed time was **20,396.9 seconds**; active time and provider wall time measure different work.

Training used two coordinated Tesla T4s, **3,584 global environments / 1,792 per device**, learning rate **0.0002**, the existing **845 scripted opponent styles** and self-play/snapshot population. These are parameterized opponent styles, not 845 independent strategies. Teacher action supervision remains disabled. Training resumed from the source policy/optimizer/RNG state without reset. Rules, scoring, sequential blind-nil visibility and the learning objective were unchanged.

The terminal content audit passed with a healthy-settlement result: all **214 native leaves were finite**, source history was preserved, the accepted-state save/readback was reported successful, and final error/preservation-error values were null. The audit records **55,801 accepted optimizer minibatches**; it does not claim every planned minibatch ran. Offline content validation does not independently prove live RAM equality or physical hardware observation.

**No stronger-play claim is made.** No audited evaluation result for checkpoint 17,574 is included. Training health and source tests do not establish win-rate improvement or better blind-nil play.

## Checkpoint availability

The original native checkpoint is **114,647,443 bytes**, SHA256:

```text
aa64a6277ec5b31e6563749215682be82aaa42f7b0e2cf7ca60269575fdc8569
```

Its verified owner-held recovery bundle is **123,225,562 bytes**, SHA256:

```text
837fb03f5e4bb13dbd4d745def5b2811d55223fef0371156cb1d67beb4f9340e
```

The binary is not committed to ordinary Git, and no public model download or new LFS/storage arrangement is provided by this publication. The owner retains the verified backup. Private journals, launch authorizations, account records, raw operational evidence and checkpoint archives are excluded.

See [machine-readable training facts](summary.json) and the [latest source snapshot](../../training/teacher_free_dualt4_v1/README.md). The public source is a documented snapshot, not an authorized training launcher.
