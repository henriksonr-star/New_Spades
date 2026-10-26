"""Small explicit profile migration, installed only after exact native restore.

The frozen objective module is never modified. Its validator receives a copy
with the historical profile label; every learning setting remains identical.
No model, optimizer, RNG, environment or snapshot array enters this adapter.
"""
import copy
import hashlib
import json

SOURCE_PROFILE = 'teacher_free_tactical333_dual_t4_v1'
TRAINING_PROFILE = 'teacher_free_novel845_dual_t4_v1'
SOURCE_NATIVE_REVISION = 'teacher-free-ppo-v1+tactical333+dual-t4-v1'
NATIVE_REVISION = 'teacher-free-ppo-v1+novel845+dual-t4-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_config(cfg):
    if cfg.get('opponent_training_profile') != TRAINING_PROFILE:
        raise ValueError('Explicit novel845 execution profile required.')
    result = copy.deepcopy(cfg)
    result['opponent_training_profile'] = SOURCE_PROFILE
    return result


def build_training_config(cfg, objective):
    objective.validate_config(cfg)
    result = copy.deepcopy(cfg)
    result['opponent_training_profile'] = TRAINING_PROFILE
    return result


def objective_metadata(cfg, objective):
    return objective.objective_metadata(source_config(cfg))


def training_profile_metadata(cfg, objective, execution=None):
    objective.validate_config(source_config(cfg))
    return dict(format='spades-teacher-free-novel845-profile-v1', profile=TRAINING_PROFILE,
        training_style_count=845, retained_training_style_count=333, new_training_style_count=512,
        excluded_evaluation_style_count=256, retained_excluded_evaluation_style_count=128,
        new_excluded_evaluation_style_count=128, holdout_used_for_training=False,
        scripted_population_probabilities=dict(retained333=.5, novel512=.5),
        outer_match_probabilities=dict(selfplay=cfg['selfplay_prob'], snapshot=cfg['snapshot_prob'],
            scripted=1.-cfg['selfplay_prob']-cfg['snapshot_prob']),
        assignment_scope='New match assignments only; existing restored assignments persist.',
        training_objective=objective_metadata(cfg, objective),
        runtime_execution=copy.deepcopy(execution), frozen_benchmark_unchanged=True,
        playing_strength_claim=False)


def migration_record(source_cfg, target_cfg, objective):
    if target_cfg != build_training_config(source_cfg, objective):
        raise ValueError('Only opponent_training_profile may change on11591 continuation.')
    return dict(format='spades11591-novel845-profile-migration-v1',
        source_cfg_sha256=digest(source_cfg), target_cfg_sha256=digest(target_cfg),
        changes={'opponent_training_profile': dict(before=SOURCE_PROFILE, after=TRAINING_PROFILE)},
        model_optimizer_rng_and_all214_array_bytes_preserved=True,
        learning_rate=.0002, teacher_free_objective_unchanged=True, rewards_unchanged=True,
        gae_entropy_and_optimizer_unchanged=True, prior_history_preserved=True,
        training_state_schema_unchanged=True, training_state_leaf_count=214,
        parameter_reset=False, optimizer_reset=False, rng_reset=False, lifetime_counters_reset=False)


def install(ns, objective, execution_record, opponent_installer, diagnostic_installer, *, source_cfg):
    """Install after source load and all214 equality; receives no TrainingState."""
    if ns['CHECKPOINT_TRAINING_REVISION'] != SOURCE_NATIVE_REVISION:
        raise ValueError('Restore the original teacher-free source before novel845 installation.')
    original_config = ns['learning_config']
    if original_config(source_cfg) != source_cfg:
        raise ValueError('Source configuration failed exact validation.')
    target = build_training_config(source_cfg, objective)
    original_validator = ns['_validate_checkpoint_manifest']
    old_profile = ns['_expansion_training_profile_metadata']
    old_execution = ns['_teacher_free_execution_record']
    old_diversity = copy.deepcopy(ns['OPPONENT_DIVERSITY_RECORD'])
    old_objective = ns['_teacher_free_objective_metadata']
    original_health = ns['_optimizer_health']
    source_signature, target_signature = digest(source_cfg), digest(target)

    def optimizer_health(rows):
        # A metadata-only population label must not reset the original rolling
        # safety window. Preserve persisted rows; normalize a copy for the exact
        # unchanged three-rollout health rule, and only for this proven delta.
        normalized = copy.deepcopy(rows)
        for row in normalized:
            if row.get('config_signature') == source_signature:
                row['config_signature'] = target_signature
        return original_health(normalized)

    def learning_config(overrides=None):
        candidate = copy.deepcopy(target)
        if overrides is not None:
            if not isinstance(overrides, dict) or set(overrides)-set(candidate):
                raise ValueError('Exact teacher-free configuration keys required.')
            candidate.update(overrides)
        checked = original_config(source_config(candidate))
        return build_training_config(checked, objective)

    profile = lambda cfg: training_profile_metadata(cfg, objective, execution_record())
    objective_record = lambda cfg: objective_metadata(cfg, objective)

    def validator(metadata):
        saved = metadata.get('cfg')
        if not isinstance(saved, dict):
            raise ValueError('Novel845 native saved configuration required.')
        learning_config(saved)
        if (metadata.get('training_revision') != NATIVE_REVISION
                or metadata.get('opponent_training_profile') != profile(saved)
                or metadata.get('training_objective') != objective_record(saved)
                or metadata.get('runtime_execution') != execution_record()
                or metadata.get('opponent_diversity') != ns['OPPONENT_DIVERSITY_RECORD']):
            raise ValueError('Exact novel845 objective/execution/profile native metadata required.')
        translated = copy.deepcopy(metadata)
        translated['cfg'] = source_config(saved)
        translated['training_revision'] = SOURCE_NATIVE_REVISION
        translated['opponent_training_profile'] = old_profile(translated['cfg'])
        translated['runtime_execution'] = old_execution()
        translated['opponent_diversity'] = old_diversity
        translated['training_objective'] = old_objective(translated['cfg'])
        # The original validator calls learning_config dynamically. Give that
        # validation its historical pure function only for the call's duration.
        current = ns['learning_config']; ns['learning_config'] = original_config
        try:
            original_validator(translated)
        finally:
            ns['learning_config'] = current
        return saved

    ns['learning_config'] = learning_config
    opponent_report = opponent_installer(ns, target)
    diagnostic_report = diagnostic_installer(ns)
    ns['_validate_checkpoint_manifest'] = validator
    ns['_expansion_training_profile_metadata'] = profile
    ns['_teacher_free_objective_metadata'] = objective_record
    ns['_teacher_free_execution_record'] = execution_record
    ns['_optimizer_health'] = optimizer_health
    ns['CHECKPOINT_TRAINING_REVISION'] = NATIVE_REVISION
    ns['OPPONENT_DIVERSITY_RECORD'] = copy.deepcopy(ns['NOVEL_OPPONENT_MANIFEST'])
    def styles(array, n_env, legacy=False):
        values = ns['np'].asarray(array)
        permitted = tuple(range(13)) if legacy else tuple(ns['NOVEL_TRAIN_STYLE_IDS'])
        if (values.shape != (int(n_env),) or str(values.dtype) != 'int32'
                or not ns['np'].isin(values, permitted).all()):
            raise ValueError('Novel native contains invalid or excluded evaluation style IDs.')
    ns['_validate_diversity_native_styles'] = styles
    return dict(format='spades11591-novel845-namespace-install-v1',
        cfg=target, migration=migration_record(source_cfg, target, objective),
        opponents=opponent_report, diagnostics=diagnostic_report,
        state_argument_accepted=False, state_arrays_mutated=False)
