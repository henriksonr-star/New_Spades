"""Same novel845 profile; install definitions before restoring exact12120 arrays."""
from pathlib import Path
import copy
import hashlib
import types

_PATH=Path(__file__).with_name('novel845_profile.py')
_RAW=_PATH.read_bytes()
if hashlib.sha256(_RAW).hexdigest()!='5e752517e877bd86ade1c932a771bb3fb2f20b2ef6c53dc14dfdca067c5938d0':
    raise ValueError('Frozen novel845 profile bytes changed.')
_prior=types.ModuleType('recovery12120_frozen_profile');_prior.__file__=str(_PATH)
exec(compile(_RAW,str(_PATH),'exec'),_prior.__dict__)

SOURCE_PROFILE=TRAINING_PROFILE=_prior.TRAINING_PROFILE
SOURCE_NATIVE_REVISION=NATIVE_REVISION=_prior.NATIVE_REVISION
digest=_prior.digest
source_config=_prior.source_config
objective_metadata=_prior.objective_metadata
training_profile_metadata=_prior.training_profile_metadata
install=_prior.install


def build_training_config(cfg,objective):
    if cfg.get('opponent_training_profile')==TRAINING_PROFILE:
        objective.validate_config(source_config(cfg))
        return copy.deepcopy(cfg)
    return _prior.build_training_config(cfg,objective)


def migration_record(source_cfg,target_cfg,objective):
    if source_cfg.get('opponent_training_profile')!=TRAINING_PROFILE:
        return _prior.migration_record(source_cfg,target_cfg,objective)
    if source_cfg!=target_cfg or build_training_config(source_cfg,objective)!=source_cfg:
        raise ValueError('Recovery must preserve the complete12120 saved configuration.')
    return dict(format='spades12120-same-profile-recovery-v1',source_cfg_sha256=digest(source_cfg),
        target_cfg_sha256=digest(target_cfg),changed_keys=[],
        model_optimizer_rng_and_all214_array_bytes_preserved=True,learning_rate=.0002,
        teacher_free_objective_unchanged=True,rewards_unchanged=True,gae_entropy_and_optimizer_unchanged=True,
        prior_history_preserved=True,training_state_schema_unchanged=True,training_state_leaf_count=214,
        parameter_reset=False,optimizer_reset=False,rng_reset=False,lifetime_counters_reset=False,
        original_run_setup_guard_failure_preserved=True,original_live_ram_readback_proven=False)
