"""Host installation/reporting only; no checkpoint/state or config mutation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'expanded_opponents_v5.py'
MANIFEST = ROOT / 'expanded_opponent_manifest_v5.json'


def install(ns, cfg=None):
    """Call once after exact old namespace/restore, before building new trainer.

    cfg, if supplied, is checked but never changed. Existing match styles,
    RNG arrays, optimizer, model, snapshots and all other state remain untouched.
    """
    if 'NOVEL_OPPONENT_REVISION' in ns:
        raise ValueError('Novel opponent namespace already installed; refusing duplicate wrapping.')
    if len(ns.get('TACTICAL_TRAIN_STYLE_IDS', ())) != 333 or len(ns.get('TACTICAL_HOLDOUT_STYLE_IDS', ())) != 128:
        raise ValueError('Exact tactical-v4 predecessor population required.')
    if cfg is not None:
        expected = 'teacher_free_novel845_dual_t4_v1'
        if cfg.get('opponent_training_profile') != expected:
            raise ValueError('Expanded training profile required; caller owns configuration transition.')
        if (cfg.get('selfplay_prob'),cfg.get('snapshot_prob')) != (.5,.25):
            raise ValueError('Actual outer config selfplay_prob/snapshot_prob must remain .5/.25; scripted is the .25 complement.')
    expected_manifest = json.loads(MANIFEST.read_text())
    if (list(ns['TACTICAL_TRAIN_STYLE_IDS']) != expected_manifest['inherited_train_style_ids']
            or list(ns['TACTICAL_HOLDOUT_STYLE_IDS']) != expected_manifest['inherited_holdout_style_ids']
            or ns.get('TACTICAL_OPPONENT_REVISION') != expected_manifest['base_revision']):
        raise ValueError('Exact inherited IDs and tactical revision required.')
    source = SOURCE.read_bytes()
    # A temporary namespace validates the module/manifest before touching the
    # live function bindings. No game state is available to this installer.
    trial = dict(ns)
    exec(compile(source, str(SOURCE), 'exec'), trial)
    if trial['NOVEL_OPPONENT_MANIFEST'] != expected_manifest:
        raise ValueError('Novel opponent source/manifest mismatch.')
    exec(compile(source, str(SOURCE), 'exec'), ns)
    return dict(revision=ns['NOVEL_OPPONENT_REVISION'],
        source_sha256=hashlib.sha256(source).hexdigest(),
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        train_style_count=845, heldout_style_count=256,
        new_train_style_count=512, new_heldout_style_count=128,
        state_mutated=False, cfg_mutated=False, sampler_scope='future scripted match assignments',
        outer_sampling_unchanged=True, fresh_evaluation_seeds=expected_manifest['fresh_evaluation_seeds'])


def sampled_id_diagnostics(ns, style_ids, kinds=None):
    """Report ACTUAL IDs supplied by caller; kinds, if supplied, uses scripted=2.

    Use runtime assignment arrays or a separately labeled sampler dry-run.
    This function never claims actual rollout exposure from a synthetic draw.
    """
    import numpy as np
    styles = np.asarray(style_ids)
    if styles.dtype.kind not in ('i','u'):
        raise ValueError('Actual style IDs must have an integer dtype.')
    if kinds is not None:
        kind = np.asarray(kinds)
        if kind.shape != styles.shape:
            raise ValueError('styles/kinds shape mismatch')
        styles = styles[kind == 2]
    styles = styles.reshape(-1)
    train = set(ns['NOVEL_TRAIN_STYLE_IDS'])
    hold = set(ns['NOVEL_HOLDOUT_STYLE_IDS'])
    actual = set(int(x) for x in styles)
    illegal = sorted(actual - train)
    if illegal:
        raise ValueError('Nontraining or heldout IDs in scripted assignment: '+str(illegal))
    ids, counts = np.unique(styles, return_counts=True)
    families = {name:int(np.sum((styles>=20000+64*i)&(styles<20000+64*(i+1))))
        for i,name in enumerate(ns['NOVEL_TRAIN_FAMILIES'])}
    return dict(scripted_assignment_count=int(styles.size),
        inherited_assignment_count=int(np.sum(styles<20000)),
        new_assignment_count=int(np.sum((styles>=20000)&(styles<20512))),
        new_family_counts=families, unique_style_count=len(actual),
        observed_style_counts={str(int(k)):int(v) for k,v in zip(ids,counts)},
        heldout_assignment_count=len(actual&hold), all_train_ids_valid=not illegal)
