"""Teacher-free PPO continuation of the exact frozen production trainer.

This module never loads, initializes, migrates, or mutates a TrainingState.
Install only after the source native's 214 arrays have been restored and checked.
The notebook is a pinned source input, not an alternative execution entry point.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

PRODUCTION_SHA256 = 'ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8'
OBJECTIVE = 'teacher_free_ppo_v1'
TRAINING_PROFILE = 'teacher_free_tactical333_dual_t4_v1'
TRAINING_REVISION = 'teacher-free-ppo-v1'
NATIVE_REVISION = 'teacher-free-ppo-v1+tactical333+dual-t4-v1'
REMOVED_SETTINGS = (
    'bootstrap_epochs', 'bootstrap_minibatch_size', 'teacher_warmup_decisions',
    'bidding_anchor_coef', 'play_anchor_coef', 'teacher_style',
    'teacher_decline_blind', 'teacher_anchor_mode', 'anchor_conflict_floor',
    'anchor_advantage_scale',
)
LOSS_NAMES = ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl',
              'clip_fraction', 'grad_norm')
BLIND_METRIC_NAMES = (
    'eligible_blind_count', 'eligible_blind_dn_probability_sum',
    'eligible_blind_dn_sample_count', 'eligible_blind_entropy_sum',
)
PRESERVED_SETTINGS = (
    'seed', 'n_env', 'rollout_len', 'ppo_epochs', 'minibatch_size', 'width', 'lr',
    'gae_lambda', 'gae_per_hand', 'clip_eps', 'value_clip', 'max_grad_norm',
    'entropy_coef', 'value_coef', 'target_kl', 'snapshot_size',
    'snapshot_interval_updates', 'selfplay_prob', 'snapshot_prob',
    'hand_score_reward_coef', 'max_training_match_hands',
)


class TeacherFreeTransition(NamedTuple):
    obs: object
    critic_obs: object
    actions: object
    logp: object
    values: object
    rewards: object
    discounts: object
    policy_mask: object
    actor_sign: object
    legal: object


class TeacherFreePPOBatch(NamedTuple):
    obs: object
    critic_obs: object
    actions: object
    logp: object
    old_values: object
    returns: object
    advantages: object
    policy_mask: object
    legal: object


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('Teacher-free configuration must be a dictionary.')
    if any(k in cfg for k in REMOVED_SETTINGS):
        raise ValueError('Teacher/bootstrap/anchor settings are excluded from this objective.')
    if set(cfg) != set(PRESERVED_SETTINGS) | {'training_objective', 'opponent_training_profile'}:
        raise ValueError('Exact teacher-free configuration schema required.')
    if (cfg.get('training_objective') != OBJECTIVE
            or cfg.get('opponent_training_profile') != TRAINING_PROFILE):
        raise ValueError('Exact teacher-free objective and tactical333 profile required.')
    if type(cfg.get('lr')) is not float or cfg['lr'] != .0002:
        raise ValueError('Unchanged learning rate 0.0002 required.')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('Integer source seed in [0, 2**32) required.')
    for key in ('n_env', 'rollout_len', 'ppo_epochs', 'minibatch_size', 'width',
                'snapshot_size', 'snapshot_interval_updates', 'max_training_match_hands'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Positive integer setting required: ' + key)
    if cfg['n_env'] * cfg['rollout_len'] % cfg['minibatch_size']:
        raise ValueError('The global minibatch must divide the global rollout.')
    if type(cfg['gae_per_hand']) is not bool:
        raise ValueError('Boolean gae_per_hand required.')
    for key in ('gae_lambda', 'clip_eps', 'value_clip', 'max_grad_norm', 'target_kl',
                'entropy_coef', 'value_coef', 'hand_score_reward_coef',
                'selfplay_prob', 'snapshot_prob'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError('Finite numeric setting required: ' + key)
    if not 0 < cfg['gae_lambda'] <= 1:
        raise ValueError('gae_lambda must lie in (0, 1].')
    if any(cfg[k] <= 0 for k in ('clip_eps', 'value_clip', 'max_grad_norm', 'target_kl')):
        raise ValueError('Positive PPO settings required.')
    if any(cfg[k] < 0 for k in ('entropy_coef', 'value_coef', 'hand_score_reward_coef')):
        raise ValueError('Nonnegative loss/reward coefficients required.')
    if (not 0 <= cfg['selfplay_prob'] <= 1 or not 0 <= cfg['snapshot_prob'] <= 1
            or cfg['selfplay_prob'] + cfg['snapshot_prob'] > 1):
        raise ValueError('Nonnegative matchmaking probabilities summing to at most one required.')
    canonical(cfg)
    return True


def build_training_config(source_cfg):
    if not isinstance(source_cfg, dict) or 'training_objective' in source_cfg:
        raise ValueError('Unmigrated frozen source configuration required.')
    cfg = {k: copy.deepcopy(v) for k, v in source_cfg.items() if k not in REMOVED_SETTINGS}
    cfg.update(training_objective=OBJECTIVE, opponent_training_profile=TRAINING_PROFILE)
    validate_config(cfg)
    return cfg


def objective_metadata(cfg):
    validate_config(cfg)
    return dict(format='spades-teacher-free-objective-v1', objective=OBJECTIVE,
        formula='policy_loss + value_coef * value_loss - entropy_coef * entropy',
        policy='masked clipped on-policy PPO surrogate',
        value='original clipped value mean over all rows',
        entropy='original legal-action entropy mean over actionable learner rows',
        value_coef=cfg['value_coef'], entropy_coef=cfg['entropy_coef'],
        hand_score_reward_coef=cfg['hand_score_reward_coef'],
        teacher_actions_generated=False, teacher_targets_in_batch=False,
        teacher_losses_present=False, supervised_bootstrap_enabled=False,
        engine_rewards_unchanged=True, hand_score_reward_unchanged=True,
        gae_unchanged=True, optimizer_unchanged=True, rng_split_order_unchanged=True,
        logit_forcing=False, exploration_forcing=False,
        removed_settings=list(REMOVED_SETTINGS), loss_metrics=list(LOSS_NAMES),
        eligibility_diagnostics=list(BLIND_METRIC_NAMES),
        diagnostic_scope='Learner-controlled BLIND decisions where Double Nil is legal; '
                         'probabilities are from the sampled policy before acting, with no floor or forcing.',
        production_notebook_sha256=PRODUCTION_SHA256)


def training_profile_metadata(cfg):
    validate_config(cfg)
    return dict(format='spades-teacher-free-tactical333-profile-v1',
        profile=TRAINING_PROFILE, training_objective=objective_metadata(cfg),
        opponent_behavior_unchanged=True, training_style_count=333,
        heldout_style_count=128, holdout_used_for_training=False,
        saved_outer_configuration={k: cfg[k] for k in ('selfplay_prob', 'snapshot_prob')},
        assignment_scope='match; not decision or time fractions',
        frozen_benchmark_unchanged=True, playing_strength_claim=False)


def migration_record(source_cfg, target_cfg):
    if canonical(target_cfg) != canonical(build_training_config(source_cfg)):
        raise ValueError('Only removal of teacher settings and explicit objective/profile migration allowed.')
    changes = {k: dict(before=source_cfg[k], after_present=False)
               for k in REMOVED_SETTINGS if k in source_cfg}
    changes.update(training_objective=dict(before_present=False, after=OBJECTIVE),
                   opponent_training_profile=dict(before=source_cfg.get('opponent_training_profile'),
                                                  after=TRAINING_PROFILE))
    return dict(kind='teacher_free_dual_gpu_configuration',
        format='spades-teacher-free-dual-gpu-migration-v1',
        source_cfg_sha256=_digest(source_cfg), cfg_sha256=_digest(target_cfg), changes=changes,
        training_revision=NATIVE_REVISION, normal_lr=.0002, lr_changed=False,
        all_other_configuration_unchanged=True, optimizer_state_retained=True,
        rng_state_retained=True, parameter_reset=False, state_arrays_mutated=False,
        training_state_schema_unchanged=True, training_state_leaf_count=214,
        reset_bootstrap_progress=False)


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Exact pinned source region required: ' + before[:100])
    return source.replace(before, after, 1)


def patched_training_definition(production_notebook_path):
    raw = Path(production_notebook_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRODUCTION_SHA256:
        raise ValueError('Exact reviewed production notebook required.')
    notebook = json.loads(raw)
    cell = ''.join(notebook['cells'][10]['source'])
    matches = [n for n in ast.parse(cell).body
               if isinstance(n, ast.FunctionDef) and n.name == 'make_train_step']
    if len(matches) != 1:
        raise ValueError('Unique frozen training definition required.')
    source = ast.get_source_segment(cell, matches[0])
    before = '''            # Continue public-information guidance on the learner's own visited
            # positions, including card play that the old bidding-only anchor omitted.
            teacher_phase = (s.envs.turn < 4) | ((s.envs.turn >= 8) & (s.envs.turn < 12))
            teacher_actions = jax.lax.cond(jnp.any(teacher_phase) | (cfg['play_anchor_coef'] > 0),
                lambda _: jax.vmap(lambda env, key: teacher_action(env, key, cfg))(
                    s.envs, jax.random.split(k_heuristic, n)),
                lambda _: jnp.full(n, NOOP, jnp.int32), operand=None)
            teacher_mask = teacher_phase & learner_turn & (jnp.sum(legal, axis=-1) > 1)
'''
    source = _replace_once(source, before, '')
    source = _replace_once(source, 'transition = Transition(', 'transition = TeacherFreeTransition(')
    source = _replace_once(source,
        'jnp.where(who % 2 == 0, 1., -1.), legal, teacher_actions, teacher_mask)',
        'jnp.where(who % 2 == 0, 1., -1.), legal)')
    source = _replace_once(source, 'batch = PPOBatch(', 'batch = TeacherFreePPOBatch(')
    source = _replace_once(source,
        'trajectory.values, returns, normalized_adv, mask, trajectory.legal,\n'
        '                         trajectory.teacher_actions, trajectory.teacher_mask)',
        'trajectory.values, returns, normalized_adv, mask, trajectory.legal)')
    source = _replace_once(source,
        '''    _loss_names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                   "clip_fraction", "bidding_anchor_loss", "play_anchor_loss",
                   "teacher_accuracy_bid", "teacher_accuracy_play", "grad_norm")''',
        '    _loss_names = _TEACHER_FREE_LOSS_NAMES')
    begin = source.index('        teacher_logp = ')
    end = source.index('\n    @jax.jit\n    def optimize', begin)
    source = source[:begin] + '''        loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
        return loss, jnp.stack([loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction])
''' + source[end:]
    source = _replace_once(source, '    model, tx = model_for_config(cfg), _optimizer(cfg)',
        '    model, tx = model_for_config(cfg), _optimizer(cfg)\n'
        '    _batch_constraint = _teacher_free_batch_constraint')
    source = _replace_once(source, '    def loss_fn(params, batch):\n',
        '    def loss_fn(params, batch):\n        batch = _batch_constraint(batch)\n')
    source = _replace_once(source,
        '            entropy_counts = jnp.sum(entropy_masks, axis=1)\n',
        '''            entropy_counts = jnp.sum(entropy_masks, axis=1)
            eligible_blind = learner_turn & (phases == BLIND) & legal[:, DOUBLE_NIL]
            eligible_blind_metrics = (
                jnp.sum(eligible_blind),
                jnp.sum(jnp.where(eligible_blind, jnp.exp(logs[:, DOUBLE_NIL]), 0.)),
                jnp.sum(eligible_blind & (actions == DOUBLE_NIL)),
                jnp.sum(jnp.where(eligible_blind, entropies, 0.)),
            )
''')
    source = _replace_once(source, '                jnp.sum(truncated),\n',
        '                jnp.sum(truncated), *eligible_blind_metrics,\n')
    source = _replace_once(source,
        'metrics = dict(zip(_ROLLOUT_METRIC_NAMES, jnp.sum(metric_rows, axis=0)))',
        'metrics = dict(zip(_ROLLOUT_METRIC_NAMES + _TEACHER_FREE_BLIND_METRIC_NAMES,\n'
        '                           jnp.sum(metric_rows, axis=0)))')
    source = _replace_once(source, '        metrics["env_steps"] = state.env_steps\n',
        '''        metrics["env_steps"] = state.env_steps
        blind_count = metrics['eligible_blind_count']
        # Zero count remains explicit; the derived zero is not evidence of collapse.
        metrics['eligible_blind_dn_probability_mean'] = metrics['eligible_blind_dn_probability_sum'] / jnp.maximum(blind_count, 1.)
        metrics['eligible_blind_dn_sample_rate'] = metrics['eligible_blind_dn_sample_count'] / jnp.maximum(blind_count, 1.)
        metrics['eligible_blind_entropy_mean'] = metrics['eligible_blind_entropy_sum'] / jnp.maximum(blind_count, 1.)
''')
    source = _replace_once(source, '    train_step.loss_fn = loss_fn\n',
        '    train_step.loss_fn = loss_fn\n'
        '    train_step.loss_names = _loss_names\n'
        '    train_step.objective = _TEACHER_FREE_OBJECTIVE\n')
    tree = ast.parse(source)
    forbidden = {'teacher_action', 'teacher_actions', 'teacher_mask', '_plasticity_anchor_gate',
                 'bidding_anchor_loss', 'play_anchor_loss', 'make_bootstrap_step'}
    if any(isinstance(n, ast.Name) and n.id in forbidden for n in ast.walk(tree)):
        raise ValueError('Teacher target or loss reference survived source surgery.')
    return source + '\n'


def _identity_batch(batch):
    return batch


def _disabled_guidance(*args, **kwargs):
    raise RuntimeError('Teacher actions, supervised bootstrap, and state-reset migrations are disabled '
                       'in the teacher-free continuation.')


def install_training(ns, production_notebook_path, source_cfg):
    """Install source surgery after exact state restore; no TrainingState argument.

    The runtime may set ns['_teacher_free_batch_constraint'] to its two-device
    NamedSharding constraint before constructing the trainer. Each trainer captures
    the callable once. Global array semantics keep all masked denominators global.
    """
    if '_teacher_free_source_learning_config' in ns:
        raise ValueError('Install teacher-free training only once per namespace.')
    source = patched_training_definition(production_notebook_path)
    original_config = ns['learning_config']
    if canonical(original_config(source_cfg)) != canonical(source_cfg):
        raise ValueError('Source learning configuration did not round-trip exactly.')
    source_cfg = copy.deepcopy(source_cfg)
    target_cfg = build_training_config(source_cfg)

    def learning_config(overrides=None):
        cfg = copy.deepcopy(target_cfg)
        if overrides is not None:
            if not isinstance(overrides, dict) or set(overrides) - set(cfg):
                raise ValueError('Only teacher-free configuration keys are accepted.')
            cfg.update(overrides)
        validate_config(cfg)
        # The frozen pure validator still knows legacy schema keys. Reconstruct
        # only its input, then discard them; no training callable sees those keys.
        prior = {k: copy.deepcopy(v) for k, v in source_cfg.items() if k in REMOVED_SETTINGS}
        prior.update({k: v for k, v in cfg.items() if k != 'training_objective'})
        if 'opponent_training_profile' in source_cfg:
            prior['opponent_training_profile'] = source_cfg['opponent_training_profile']
        else:
            prior.pop('opponent_training_profile', None)
        checked = original_config(prior)
        result = build_training_config(checked)
        if canonical(result) != canonical(cfg):
            raise ValueError('Teacher-free configuration failed preserved source validation.')
        return result

    ns['_teacher_free_source_learning_config'] = original_config
    ns['learning_config'] = learning_config
    ns['TeacherFreeTransition'] = TeacherFreeTransition
    ns['TeacherFreePPOBatch'] = TeacherFreePPOBatch
    ns['Transition'] = TeacherFreeTransition
    ns['PPOBatch'] = TeacherFreePPOBatch
    ns['_TEACHER_FREE_LOSS_NAMES'] = LOSS_NAMES
    ns['_TEACHER_FREE_BLIND_METRIC_NAMES'] = BLIND_METRIC_NAMES
    ns['_TEACHER_FREE_OBJECTIVE'] = OBJECTIVE
    ns.setdefault('_teacher_free_batch_constraint', _identity_batch)
    exec(compile(source, 'reviewed_teacher_free_make_train_step.py', 'exec'), ns)
    # Disabling these entry points also prevents accidental optimizer/RNG/game
    # resets through finish_bootstrap or legacy recovery during continuation.
    for name in ('teacher_action', 'make_bootstrap_step', 'gpu_bootstrap_step',
                 'finish_bootstrap', 'recover_legacy_training', 'upgrade_training_workload'):
        if name in ns:
            ns[name] = _disabled_guidance
    ns.pop('_plasticity_anchor_gate', None)
    return dict(format='spades-teacher-free-install-v1',
        definition_sha256=hashlib.sha256(source.encode()).hexdigest(),
        production_notebook_sha256=PRODUCTION_SHA256,
        training_objective=objective_metadata(target_cfg),
        state_arrays_mutated=False, optimizer_state_reset=False,
        ephemeral_transition_fields=list(TeacherFreeTransition._fields),
        ephemeral_batch_fields=list(TeacherFreePPOBatch._fields))


def activate_candidate_revision(ns):
    if '_teacher_free_source_learning_config' not in ns:
        raise ValueError('Teacher-free definitions must be installed first.')
    ns['TRAINING_REVISION'] = TRAINING_REVISION
    ns['CHECKPOINT_TRAINING_REVISION'] = NATIVE_REVISION
    return NATIVE_REVISION


def audit_work_history(history, cfg, start_update, end_update):
    """Supplement, never replace, the frozen cadence/numeric work-history audit."""
    validate_config(cfg)
    rows = [r for r in history if r.get('kind') == 'aot_raw_metrics'
            and type(r.get('update')) is int and start_update < r['update'] <= end_update]
    if [r['update'] for r in rows] != list(range(start_update + 1, end_update + 1)):
        raise ValueError('Complete ordered teacher-free raw metric coverage required.')
    for row in rows:
        c, o = row.get('collection', {}), row.get('optimizer', {})
        if not set(LOSS_NAMES) <= set(o) or not set(BLIND_METRIC_NAMES) <= set(c):
            raise ValueError('Teacher-free loss and eligible BLIND metrics required.')
        if any(any(part in key for part in ('teacher', 'anchor', 'bootstrap')) for key in o):
            raise ValueError('Teacher/anchor/bootstrap metrics forbidden after objective migration.')
        values = [o[k] for k in LOSS_NAMES] + [c[k] for k in BLIND_METRIC_NAMES]
        if not all(type(v) in (float, int) and math.isfinite(v) for v in values):
            raise ValueError('Finite numeric teacher-free objective diagnostics required.')
        count = c['eligible_blind_count']
        if (count < 0 or int(count) != count or not 0 <= c['eligible_blind_dn_sample_count'] <= count
                or int(c['eligible_blind_dn_sample_count']) != c['eligible_blind_dn_sample_count']
                or not 0 <= c['eligible_blind_dn_probability_sum'] <= count + 1e-4
                or not 0 <= c['eligible_blind_entropy_sum'] <= count * math.log(2.) + 1e-3):
            raise ValueError('Eligible BLIND probability/count/entropy bounds failed.')
        expected = o['policy_loss'] + cfg['value_coef'] * o['value_loss'] - cfg['entropy_coef'] * o['entropy']
        if not math.isclose(o['loss'], expected, rel_tol=2e-5, abs_tol=2e-5):
            raise ValueError('Raw loss is not solely PPO plus value minus entropy.')
    return dict(format='spades-teacher-free-work-audit-v1', objective=OBJECTIVE,
                recorded_rollouts=len(rows), teacher_action_supervision=False,
                requires_frozen_optimizer_work_audit=True,
                complete_raw_metrics=True)
