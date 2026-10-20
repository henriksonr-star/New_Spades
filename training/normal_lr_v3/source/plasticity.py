"""Bounded advantage-gated teacher anchoring; no optimizer or state migration.

Positive normalized on-policy advantage is relative evidence, not proof that a
teacher action is inferior. This changes only the teacher-loss numerator.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

PRODUCTION_SHA256 = 'ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8'
MODE = 'advantage_gated_v1'
SETTINGS = dict(teacher_anchor_mode=MODE, anchor_conflict_floor=0.25,
                anchor_advantage_scale=1.0)
CANDIDATE_TRAINING_REVISION = 'strong-teacher-hand-gae-v3+advantage-gated-anchor-v1'
CANDIDATE_NATIVE_REVISION = 'strong-teacher-hand-gae-v3+opponent-diversity-v1+advantage-gated-anchor-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def candidate_config(source_cfg):
    if not isinstance(source_cfg, dict) or any(k in source_cfg for k in SETTINGS):
        raise ValueError('An original unmigrated source configuration is required.')
    result = copy.deepcopy(source_cfg)
    result.update(SETTINGS)
    return result


def validate_candidate_config(source_cfg, candidate_cfg):
    if canonical(candidate_cfg) != canonical(candidate_config(source_cfg)):
        raise ValueError('Only the exact three-key teacher-anchor migration is allowed.')
    return True


def migration_record(source_cfg, candidate_cfg):
    validate_candidate_config(source_cfg, candidate_cfg)
    return dict(kind='advantage_gated_teacher_anchor_migration', format='spades-policy-plasticity-v1',
                source_cfg_sha256=hashlib.sha256(canonical(source_cfg).encode()).hexdigest(),
                candidate_cfg_sha256=hashlib.sha256(canonical(candidate_cfg).encode()).hexdigest(),
                changes={k: dict(from_present=False, to=v) for k, v in SETTINGS.items()},
                optimizer_state_retained=True, architecture_unchanged=True,
                rng_state_retained=True, parameter_reset=False,
                reason='Reduce teacher opposition on positively advantaged sampled disagreements; '
                       'preserve full imitation on aligned or nonpositive samples.',
                evidence_scope='Relative normalized GAE signal; no verified playing-strength claim.')


def anchor_weights(xp, actions, teacher_actions, advantages, *, stop_gradient=None):
    """Return weights in[0.25,1]; stop all gradients through evidence when JAX is used."""
    evidence = xp.clip(advantages / SETTINGS['anchor_advantage_scale'], 0., 1.)
    conflict = actions != teacher_actions
    weights = 1. - (1. - SETTINGS['anchor_conflict_floor']) * evidence * conflict
    return stop_gradient(weights) if stop_gradient is not None else weights


def patched_training_definition(production_notebook_path):
    """Return only make_train_step from the exact source, with a narrow loss patch."""
    raw = Path(production_notebook_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRODUCTION_SHA256:
        raise ValueError('Exact reviewed production notebook required.')
    notebook = json.loads(raw)
    cell = ''.join(notebook['cells'][10]['source'])
    matches = [n for n in ast.parse(cell).body
               if isinstance(n, ast.FunctionDef) and n.name == 'make_train_step']
    if len(matches) != 1:
        raise ValueError('Unique original training definition required.')
    source = ast.get_source_segment(cell, matches[0])
    replacements = [
        ('"teacher_accuracy_bid", "teacher_accuracy_play", "grad_norm")',
         '"teacher_accuracy_bid", "teacher_accuracy_play", "teacher_anchor_gate_bid", '
         '"teacher_anchor_gate_play", "teacher_anchor_relaxed_bid", '
         '"teacher_anchor_relaxed_play", "grad_norm")'),
        ('        loss = (policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy\n'
         "                + cfg['bidding_anchor_coef'] * bidding_anchor_loss\n"
         "                + cfg['play_anchor_coef'] * play_anchor_loss)",
         '        anchor_gate = _plasticity_anchor_gate(batch.actions, batch.teacher_actions, batch.advantages, cfg)\n'
         '        # Keep the original phase-mask denominator: weighting the mask would cancel relaxation.\n'
         '        effective_bid = -_masked_mean(teacher_logp * anchor_gate, bidding_mask)\n'
         "        effective_play = (-_masked_mean(teacher_logp * anchor_gate, card_mask)\n"
         "                          if cfg['play_anchor_coef'] > 0 else jnp.float32(0.))\n"
         '        gate_bid = _masked_mean(anchor_gate, bidding_mask)\n'
         '        gate_play = _masked_mean(anchor_gate, card_mask)\n'
         '        relaxed_bid = _masked_mean((anchor_gate < 1.).astype(jnp.float32), bidding_mask)\n'
         '        relaxed_play = _masked_mean((anchor_gate < 1.).astype(jnp.float32), card_mask)\n'
         '        loss = (policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy\n'
         "                + cfg['bidding_anchor_coef'] * effective_bid\n"
         "                + cfg['play_anchor_coef'] * effective_play)"),
        ('                               bid_accuracy, play_accuracy])',
         '                               bid_accuracy, play_accuracy, gate_bid, gate_play,\n'
         '                               relaxed_bid, relaxed_play])'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Original teacher-loss source differs; refuse an ambiguous patch.')
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source + '\n'


def install_training(ns, production_notebook_path):
    """Install before compiling; leave old revision constants intact for exact906 load."""
    source = patched_training_definition(production_notebook_path)
    jax, jnp = ns['jax'], ns['jnp']
    if '_plasticity_base_learning_config' in ns:
        raise ValueError('Install once into a fresh training namespace.')
    original_config = ns['learning_config']
    def learning_config(overrides=None):
        if overrides is None:
            return original_config()
        if not isinstance(overrides, dict):
            return original_config(overrides)
        extras = {k: overrides[k] for k in SETTINGS if k in overrides}
        if not extras:
            return original_config(overrides)
        if canonical(extras) != canonical(SETTINGS):
            raise ValueError('All three exact approved teacher-anchor settings are required.')
        original = {k: v for k, v in overrides.items() if k not in SETTINGS}
        result = original_config(original)
        result.update(SETTINGS)
        return result
    ns['_plasticity_base_learning_config'] = original_config
    ns['learning_config'] = learning_config
    def gate(actions, teacher_actions, advantages, cfg):
        if any(cfg.get(k) != v for k, v in SETTINGS.items()):
            raise ValueError('Explicit approved teacher-anchor settings required.')
        return anchor_weights(jnp, actions, teacher_actions, advantages,
                              stop_gradient=jax.lax.stop_gradient)
    ns['_plasticity_anchor_gate'] = gate
    exec(compile(source, 'reviewed_advantage_gated_make_train_step.py', 'exec'), ns)
    return dict(format='spades-policy-plasticity-install-v1', mode=MODE,
                definition_sha256=hashlib.sha256(source.encode()).hexdigest(),
                settings=dict(SETTINGS), optimizer_state_reset=False,
                source_notebook_sha256=PRODUCTION_SHA256)


def activate_candidate_revision(ns):
    """Call only after source906 has loaded and its214 arrays matched byte-for-byte."""
    ns['TRAINING_REVISION'] = CANDIDATE_TRAINING_REVISION
    ns['CHECKPOINT_TRAINING_REVISION'] = CANDIDATE_NATIVE_REVISION
    return CANDIDATE_NATIVE_REVISION
