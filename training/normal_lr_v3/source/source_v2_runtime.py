"""Definitions-only expanded-training Kaggle LR1.5 A/B from durable1201.

Both arms restore all214 leaves before adopting the same expanded training mix;
candidate lr alone differs between arms. No provider/network/installer or retry.
"""
from pathlib import Path
import ast,copy,hashlib,json,math,os,sys,time,types,uuid,zipfile,gc
from datetime import datetime,timezone

FORMAT='spades-lr-ab-expansion-config-v2'
PAYLOAD_FORMAT='spades-lr-ab-expansion-payload-manifest-v2'
PROFILE_KEY='opponent_training_profile'
PROFILE_VALUE='training_expansion_v2'
EXPANSION_SOURCE='training_opponents_v2.py'
EXPANSION_MANIFEST='training_opponent_manifest_v2.json'
MAX_ROLLOUTS=128
ARM_TRAIN_SECONDS=1800
TOTAL_SECONDS=6600
LR_MULTIPLIER=1.5
_RUN_INVOKED=False
_ACTIVE_RECOVERY={}
CORE_PINS={
 'pinned_fresh_helpers.py':'3866229a0e96ce8d738858ad5b34f9b3652d7ac35982fea6b14ecc5f79949e76',
 'spades_legacy_aot.py':'66f00e9a245d85d236eea4cda545bce1fef4578d44f7d591d36c7a803e782db1',
 'original_source.ipynb':'3236d2c95d4d1a95bf31a5cb7299853be6871783691829ee775fbb38366d1389',
 'production_source.ipynb':'ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8',
 'diversity_evaluation.py':'1751557e800d48a596b7a783595aa1bd42ecf6c20b130c97a9a498e583e16ecc',
 'sampling_policy.py':'824c41ee9318509901af82a1ca99d4814c88e2ac8112efcbc03cf9e13cc709a0',
 'audit_export_original.py':'a354a362dad26b68dc8ad3030654adc9d7646a362b2a0ca024b7977ce04b9fb7',
 'opponent_manifest.json':'20e4727fdf4d37d035c9ec1bc5d770eb3510f93c9bd20c680c512ceee5f78f1d',
 'heldout_protocol.json':'0b5970eb566bfcf3db55ffc5559a80e27c072babffe5e1c64f92d3aa5ed2942d',
}
PIN_NAMES=dict(operator_sha256='spades_lr_ab_run.py',contract_sha256='lr_ab_contract.py',
 evaluator_sha256='lr_ab_evaluation.py',plasticity_sha256='plasticity.py',
 helper_sha256='pinned_fresh_helpers.py',legacy_operator_sha256='spades_legacy_aot.py',
 original_sha256='original_source.ipynb',production_sha256='production_source.ipynb',
 frozen_evaluator_sha256='diversity_evaluation.py',sampling_sha256='sampling_policy.py',
 training_opponents_sha256=EXPANSION_SOURCE,training_opponent_manifest_sha256=EXPANSION_MANIFEST)

class BoundaryStop(RuntimeError):pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    _canonical(result)
    return result

def _identity(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Absolute regular nonsymlink file required: ' + str(path))
    info = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    if any(getattr(info, key) != getattr(after, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('File changed during identity read: ' + str(path))
    return dict(path=str(path), bytes=info.st_size, sha256=digest.hexdigest())

def _load(path, name, *, expected_sha256=None):
    # Execute the exact verified source snapshot, never a stale .pyc and never
    # by adding the payload directory to sys.path (operator.py collision).
    import types
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError('Actual module source changed before load: ' + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module

def _write_new_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return _identity(Path(path).absolute())

def _atomic_status(root, value):
    temporary = root / ('status.' + uuid.uuid4().hex + '.tmp')
    _write_new_json(temporary, value)
    temporary.replace(root / 'status.json')
    return value

def _context(platform):
    return dict(utc_now=datetime.now(timezone.utc).isoformat(), monotonic_now=time.monotonic(),
                host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                selected_gpu_uuid=platform['selected_gpu_uuid'],
                selected_gpu_pci_bus_id=platform['selected_gpu_pci_bus_id'])

def _choose_device(runtime, device_uuid):
    inventory = runtime.inventory_kaggle_nojax()
    boards = inventory['physical_gpu_inventory']
    if device_uuid is None:
        boards = sorted((row for row in boards if 'T4' in row['name']), key=lambda row: row['pci_bus_id'])
        if not boards:
            raise RuntimeError('No physical free-provider T4 visible; no CPU or paid fallback.')
        device_uuid = boards[0]['uuid']
    return runtime.preflight_kaggle_nojax(device_uuid)

def _memory_options_guard(options):
    if any(options.get(key) != value for key, value in dict(
            target_gib=14., reserve_mib=1024, safety_factor=1.45, extra_guard_mib=256).items()):
        raise RuntimeError('Reviewed exact memory options changed; no launch.')

def _append_event(root, kind, values):
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(kind=kind, utc=datetime.now(timezone.utc).isoformat(), **values), allow_nan=False) + '\n')


def _no_jax():
    if any(name in ('jax','jaxlib','jax_plugins') or name.startswith(('jax.','jaxlib.','jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Fresh kernel before JAX required; no implicit reset or second run.')


def _host_inputs(config_path,payload_dir,*,started_root=None):
    payload=Path(payload_dir).absolute();config=_read_json(config_path)
    required={'format','source1201_bundle_path','output_dir','authorization'}
    if not isinstance(config,dict) or set(config)-required-{'device_uuid'} or not required<=set(config) or config['format']!=FORMAT:
        raise ValueError('Exact LR A/B configuration schema required.')
    root=Path(config['output_dir'])
    if not root.is_absolute() or root.is_symlink() or '..' in root.parts or (root.exists() and root!=started_root):
        raise ValueError('Unique absolute new output directory required.')
    identities={}
    for name,sha in CORE_PINS.items():
        info=_identity(payload/name)
        if info['sha256']!=sha:raise ValueError('Pinned core source differs:'+name)
        identities[name]=info
    manifest=_read_json(payload/'lr_ab_payload_manifest.json')
    if manifest.get('format')!=PAYLOAD_FORMAT:raise ValueError('Reviewed expansion LR A/B payload manifest required.')
    for name in ('spades_lr_ab_run.py','lr_ab_contract.py','lr_ab_evaluation.py','plasticity.py',EXPANSION_SOURCE,EXPANSION_MANIFEST):
        info=_identity(payload/name)
        if manifest.get('members',{}).get(name)!=dict(bytes=info['bytes'],sha256=info['sha256']):
            raise ValueError('Fresh LR A/B source identity differs:'+name)
        identities[name]=info
    contract=_load(payload/'lr_ab_contract.py','spades_lr_ab_contract_'+uuid.uuid4().hex,expected_sha256=identities['lr_ab_contract.py']['sha256'])
    if (contract.DURATION_SECONDS!=TOTAL_SECONDS or contract.ARM_TRAINING_SECONDS!=ARM_TRAIN_SECONDS
            or contract.PLANNED_ROLLOUTS!=MAX_ROLLOUTS):
        raise ValueError('Actual driver and authorization duration/rollout ceilings disagree.')
    pins={key:identities[name]['sha256'] for key,name in PIN_NAMES.items()}
    contract.validate_lr_ab_authorization(config['authorization'],expected_source_pins=pins)
    source=contract.validate_source_artifacts(config['source1201_bundle_path'])
    arms=contract.build_arm_configs(source['source_cfg'])
    training_profile=contract.training_profile_metadata(source['source_cfg'])
    if (config['authorization']['opponent_training_profile']!=PROFILE_VALUE
            or config['authorization']['training_profile_sha256']!=hashlib.sha256(_canonical(training_profile).encode()).hexdigest()):
        raise ValueError('Authorization must bind actual source-derived expansion assignment proportions.')
    signatures={arm:hashlib.sha256(_canonical(cfg).encode()).hexdigest() for arm,cfg in arms.items()}
    if config['authorization']['arm_configuration_sha256']!=signatures:
        raise ValueError('Authorization must bind both actual complete arm configurations.')
    if source['source_health'].get('should_stop') is not False:
        raise RuntimeError('Original source configuration health forbids migration or continuation.')
    _no_jax()
    return config,payload,identities,contract,source,arms


def preflight(config_path,payload_dir):
    """CPU content-only admission; no framework import or GPU inventory."""
    _no_jax();config,payload,identities,contract,source,arms=_host_inputs(config_path,payload_dir)
    return dict(format='spades-lr-ab-expansion-preflight-v2',status='PASS',source_identities=source['source_identities'],
        source_health=source['source_health'],source_cfg=copy.deepcopy(source['source_cfg']),arm_configs=arms,source_pins=identities,
        training_profile=contract.training_profile_metadata(source['source_cfg']),
        planned_rollouts_per_arm=MAX_ROLLOUTS,arm_training_seconds=ARM_TRAIN_SECONDS,
        duration_seconds=TOTAL_SECONDS,jax_imported=False,gpu_work_started=False,
        playing_strength_claim=False,training_authorized_by_preflight=False)


def _shared_guard(shared,label):
    now=_context(shared['platform'])
    try:
        shared['contract'].validate_lr_ab_context(shared['authorization'],shared['clock'],now,
            previous_context=shared['highwater'])
    except ValueError as exc:raise BoundaryStop(str(exc)+':'+label) from exc
    shared['highwater']=now
    if time.monotonic()-shared['entry']['monotonic']>=TOTAL_SECONDS:raise BoundaryStop('total6600_complete:'+label)
    for name,info in shared['identities'].items():
        path=Path(info['path']);st=path.stat()
        if path.is_symlink() or tuple(getattr(st,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))!=shared['stat_pins'][name]:
            raise RuntimeError('Pinned immutable source changed:'+name+':'+label)


def _pin_sources(identities):
    pins={}
    for name,info in identities.items():
        before=Path(info['path']).stat()
        if _identity(info['path'])!=info:raise RuntimeError('Source changed after host validation:'+name)
        after=Path(info['path']).stat();keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(before,key)!=getattr(after,key) for key in keys):raise RuntimeError('Source changed while pinning:'+name)
        pins[name]=tuple(getattr(after,key) for key in keys)
    return pins


def _elapsed(shared):return max(0.,time.monotonic()-shared['entry']['monotonic'])


def _arm_elapsed(session):
    start=session.get('training_started_monotonic')
    if start is None:return 0.
    end=session.get('training_stopped_monotonic')
    return max(0.,(time.monotonic() if end is None else end)-start)


def _arm_guard(session,label,*,training=True):
    shared=session['shared'];_shared_guard(shared,label)
    runtime=session['runtime'];ns=session['ns']
    runtime._aot_kaggle_identity_guard(session);_memory_options_guard(session['options'])
    if (_canonical(session['cfg'])!=_canonical(session['saved_cfg']) or session['handles']!=session['saved_handles']
            or runtime._aot_eval_namespace_identity(ns)!=session['runtime_callable_identity']
            or runtime._aot_eval_namespace_identity(vars(runtime))!=session['operator_callable_identity']
            or ns['_gpu_runtime_identity']()!=session['runtime_identity']
            or _expansion_namespace_binding(ns)!=session['expansion_namespace_binding']):
        raise RuntimeError('Arm source/config/callable/AOT identity changed:'+label)
    if training:
        runtime._aot_style_identity_guard(session)
        if session.get('terminal') or _arm_elapsed(session)>=ARM_TRAIN_SECONDS:
            raise BoundaryStop('arm_training1800_complete:'+label)
    _shared_guard(shared,label+' after validation')


def _status(shared):
    return dict(format='spades-lr-ab-expansion-status-v2',campaign_id=shared['authorization']['campaign_id'],run_id=shared['authorization']['run_id'],
        phase=shared['phase'],terminal=shared.get('terminal',False),stop_reason=shared.get('stop_reason'),
        lr_ab_clock=shared['clock'],elapsed_seconds=_elapsed(shared),source_cfg=shared['source']['source_cfg'],
        training_profile=shared.get('training_profile'),
        arm_configs=shared['arm_configs'],arm_records=copy.deepcopy(shared['arm_records']),
        planned_rollouts_per_arm=MAX_ROLLOUTS,arm_training_seconds=ARM_TRAIN_SECONDS,
        training_enabled=not shared.get('terminal',False),promotion=False,playing_strength_claim=False,
        evaluation=shared.get('evaluation'),error=shared.get('error'),exports=shared['exports'],
        external_verification_required_before_new_run=True)


def _checkpoint_definition(payload,name):
    """Extract one definition from the immutable production notebook."""
    cell=''.join(_read_json(Path(payload)/'production_source.ipynb')['cells'][18]['source'])
    matches=[node for node in ast.parse(cell).body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(matches)!=1:raise RuntimeError('Unique pinned checkpoint definition required:'+name)
    return ast.get_source_segment(cell,matches[0])+'\n'


def _expansion_namespace_binding(ns):
    names=('TRAINING_REVISION','CHECKPOINT_TRAINING_REVISION','OPPONENT_DIVERSITY_RECORD',
        'FROZEN_OPPONENT_BENCHMARK_RECORD','DIVERSITY_OPPONENT_MANIFEST','DIVERSITY_TRAIN_STYLE_IDS',
        'DIVERSITY_HOLDOUT_STYLE_IDS','EXPANSION_OPPONENT_REVISION','EXPANSION_OPPONENT_MANIFEST',
        'EXPANSION_PROFILE_KEY','EXPANSION_PROFILE_VALUE','EXPANSION_TRAIN_STYLE_IDS',
        'EXPANSION_HOLDOUT_STYLE_IDS')
    return hashlib.sha256(_canonical({name:ns[name] for name in names}).encode()).hexdigest()


def _install_training_expansion(ns,runtime,contract,payload,identities,cfg,guard):
    """Post-restore namespace/schema migration; never modify a state leaf."""
    if cfg.get(PROFILE_KEY)!=PROFILE_VALUE:
        raise RuntimeError('Exact new expansion configuration required.')
    source_revision=ns['CHECKPOINT_TRAINING_REVISION']
    if source_revision!=contract.SOURCE_NATIVE_REVISION or ns.get('_training_expansion_installed'):
        raise RuntimeError('Install expansion once, after exact adaptive source restoration.')
    guard('before expansion definitions')
    before_callables={name:value for name,value in ns.items() if callable(value)}
    inherited_values={name:copy.deepcopy(value) for name,value in ns.items()
        if name.startswith('DIVERSITY_') or name=='HEURISTIC_NAMES'}
    frozen_record=copy.deepcopy(ns['OPPONENT_DIVERSITY_RECORD'])
    source_validator=ns['_validate_checkpoint_manifest']
    # Snapshot globals preserve the original source validator after revisions
    # and checkpoint metadata become expanded in the live training namespace.
    source_validator=types.FunctionType(source_validator.__code__,dict(ns),source_validator.__name__,
        source_validator.__defaults__,source_validator.__closure__)
    raw=(Path(payload)/EXPANSION_SOURCE).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identities[EXPANSION_SOURCE]['sha256']:
        raise RuntimeError('Expansion definition bytes changed before installation.')
    exec(compile(raw,str(Path(payload)/EXPANSION_SOURCE),'exec'),ns)
    allowed={'diverse_heuristic_action','decode_opponent_style_id','is_train_opponent_style_id',
        'is_holdout_opponent_style_id','sample_train_opponent_style_ids'}
    if any(ns.get(name) is not value for name,value in before_callables.items() if name not in allowed):
        raise RuntimeError('Expansion modified an original rule/model/teacher/checkpoint callable.')
    if any(_canonical(ns.get(name))!=_canonical(value) for name,value in inherited_values.items()):
        raise RuntimeError('Expansion modified inherited opponent IDs/parameters/benchmark metadata.')
    manifest=_read_json(Path(payload)/EXPANSION_MANIFEST)
    if (_canonical(ns['EXPANSION_OPPONENT_MANIFEST'])!=_canonical(manifest)
            or ns.get('EXPANSION_PROFILE_KEY')!=PROFILE_KEY or ns.get('EXPANSION_PROFILE_VALUE')!=PROFILE_VALUE
            or len(ns['EXPANSION_TRAIN_STYLE_IDS'])!=205 or len(ns['EXPANSION_HOLDOUT_STYLE_IDS'])!=64):
        raise RuntimeError('Exact expanded training manifest/profile/style domain required.')
    record=contract.expansion_opponent_record()
    if (_canonical(record['training_style_ids'])!=_canonical(list(ns['EXPANSION_TRAIN_STYLE_IDS']))
            or _canonical(record['heldout_style_ids'])!=_canonical(list(ns['EXPANSION_HOLDOUT_STYLE_IDS']))):
        raise RuntimeError('Expanded namespace and native contract domains disagree.')
    base_learning_config=ns['learning_config']
    def learning_config(overrides=None):
        if not isinstance(overrides,dict) or PROFILE_KEY not in overrides:
            return base_learning_config(overrides)
        if overrides[PROFILE_KEY]!=PROFILE_VALUE:raise ValueError('Exact reviewed expansion profile required.')
        original={key:value for key,value in overrides.items() if key!=PROFILE_KEY}
        out=base_learning_config(original);out[PROFILE_KEY]=PROFILE_VALUE
        return out
    def validate_manifest(metadata):
        if metadata.get('training_revision')==source_revision:
            return source_validator(metadata)
        saved=metadata.get('cfg')
        if (metadata.get('training_revision')!=contract.NATIVE_REVISION or not isinstance(saved,dict)
                or saved.get(PROFILE_KEY)!=PROFILE_VALUE
                or _canonical(metadata.get('opponent_diversity'))!=_canonical(record)
                or _canonical(metadata.get('frozen_opponent_benchmark'))!=_canonical(frozen_record)
                or _canonical(metadata.get('opponent_training_profile'))!=_canonical(contract.training_profile_metadata(saved))):
            raise ValueError('Exact expanded native revision/profile/frozen benchmark required.')
        original=copy.deepcopy(metadata);original['training_revision']=source_revision
        original['opponent_diversity']=copy.deepcopy(frozen_record)
        original['cfg']={key:value for key,value in saved.items() if key!=PROFILE_KEY}
        original.pop('frozen_opponent_benchmark',None);original.pop('opponent_training_profile',None)
        source_validator(original)
        learning_config(saved)
        return saved
    def validate_styles(array,n_env,legacy=False):
        values=ns['np'].asarray(array)
        permitted=tuple(range(13)) if legacy else tuple(record['training_style_ids'])
        if (values.shape!=(int(n_env),) or str(values.dtype)!='int32'
                or not ns['np'].isin(values,permitted).all()):
            raise ValueError('Expanded native style IDs outside exact training allowlist; held-out IDs forbidden.')
    ns['learning_config']=learning_config
    ns['_validate_checkpoint_manifest']=validate_manifest
    ns['_validate_diversity_native_styles']=validate_styles
    ns['_EXPANSION_SOURCE_NATIVE_REVISION']=source_revision
    ns['CHECKPOINT_TRAINING_REVISION']=contract.NATIVE_REVISION
    ns['TRAINING_REVISION']+='+'+contract.NATIVE_REVISION.rsplit('+',1)[-1]
    ns['OPPONENT_DIVERSITY_RECORD']=copy.deepcopy(record)
    ns['FROZEN_OPPONENT_BENCHMARK_RECORD']=copy.deepcopy(frozen_record)
    ns['_expansion_training_profile_metadata']=contract.training_profile_metadata
    source=_checkpoint_definition(payload,'save_checkpoint')
    target="        'opponent_diversity': _json_safe(OPPONENT_DIVERSITY_RECORD),\n"
    insertion=target+"        'frozen_opponent_benchmark': _json_safe(FROZEN_OPPONENT_BENCHMARK_RECORD),\n"+"        'opponent_training_profile': _json_safe(_expansion_training_profile_metadata(cfg)),\n"
    if source.count(target)!=1:raise RuntimeError('Pinned native opponent metadata insertion differs.')
    source=source.replace(target,insertion,1)
    load_source=_checkpoint_definition(payload,'load_checkpoint')
    target='revision not in (TRAINING_REVISION, CHECKPOINT_TRAINING_REVISION)'
    if load_source.count(target)!=1:raise RuntimeError('Pinned native revision loader insertion differs.')
    load_source=load_source.replace(target,'revision not in (TRAINING_REVISION, CHECKPOINT_TRAINING_REVISION, _EXPANSION_SOURCE_NATIVE_REVISION)',1)
    exec(compile(source+load_source,'reviewed_expansion_checkpoint_schema_v2.py','exec'),ns)
    runtime._aot_diversity_contract=lambda:copy.deepcopy(record)
    runtime._aot_validate_style_array=validate_styles
    ns['_training_expansion_installed']=True
    guard('after expansion definitions')
    return dict(format='spades-training-expansion-install-v2',source_native_revision=source_revision,
        native_revision=contract.NATIVE_REVISION,training_profile=contract.training_profile_metadata(cfg),
        expansion_source_sha256=identities[EXPANSION_SOURCE]['sha256'],
        expansion_manifest_sha256=identities[EXPANSION_MANIFEST]['sha256'],
        checkpoint_definition_sha256=hashlib.sha256((source+load_source).encode()).hexdigest(),
        state_arrays_modified=False,optimizer_state_reset=False,rng_state_reset=False,
        existing_league_assignments_preserved_until_match_done=True)


def _prepare_arm(shared,arm):
    """Independent full-state restore; no state, optimizer or RNG reinitialization."""
    payload=shared['payload'];ids=shared['identities'];root=Path(shared['root'])/arm;root.mkdir(exist_ok=False)
    (root/'checkpoints').mkdir()
    runtime=_load(payload/'spades_legacy_aot.py','spades_lr_ab_legacy_'+arm+'_'+uuid.uuid4().hex,expected_sha256=ids['spades_legacy_aot.py']['sha256'])
    helpers=shared['helpers'];platform=shared['platform'];guard=lambda label:_shared_guard(shared,arm+':'+label)
    guard('before framework import')
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax
    from flax import linen as nn
    import optax
    stack=dict(jax='0.11.1',jaxlib='0.11.1',flax='0.12.10',optax='0.2.8')
    if ({'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),'flax':flax.__version__,'optax':optax.__version__}!=stack
            or jax.config.jax_enable_x64 or jax.config.jax_disable_jit or jax.config.jax_default_matmul_precision is not None):
        raise RuntimeError('Exact imported stack/config/precision required.')
    guard('device discovery');devices=jax.devices()
    if len(devices)!=1 or jax.default_backend()!='gpu' or 'T4' not in str(devices[0].device_kind):
        raise RuntimeError('Exactly one selected physical T4 required.')
    original=_read_json(payload/'original_source.ipynb');production=_read_json(payload/'production_source.ipynb')
    ns=dict(__name__='spades_lr_ab_'+arm,Path=Path,os=os,time=time,json=json,math=math,np=np,jax=jax,jnp=jnp,
        flax=flax,nn=nn,optax=optax,_SPADES_KAGGLE_PLATFORM=copy.deepcopy(platform),
        _fresh_frozen_evaluation_path=str(payload/'diversity_evaluation.py'))
    guard('memory definitions');exec(compile(''.join(production['cells'][12]['source']),'pinned_memory_cell12.py','exec'),ns)
    opts=dict(ns['GPU_TUNING']);_memory_options_guard(opts)
    snapshot=ns['_gpu_memory_snapshot']()
    if any(snapshot.get(k)!=platform[v] for k,v in (('gpu_uuid','selected_gpu_uuid'),('gpu_pci_bus_id','selected_gpu_pci_bus_id'),('board_total_bytes','selected_gpu_total_bytes'))):
        raise RuntimeError('Selected physical UUID/PCI/capacity mapping failed.')
    def admit(estimate,label):
        guard(label+' before memory admission');_memory_options_guard(opts)
        record=ns['_gpu_memory_admission'](estimate,opts,label)
        details=record.get('budget_details',{});snap=record.get('snapshot',{})
        if (record.get('status')!='admitted' or record.get('overlap_credit_bytes')!=0 or snap.get('platform')!='gpu'
                or details.get('pool_counter_source')!='pool_bytes' or details.get('pool_counter_status')!='validated_current_total'
                or 'pool_bytes' not in snap.get('jax_memory_stats_raw',{})):
            raise RuntimeError('Actual current validated pool/board memory admission required.')
        guard(label+' after memory admission');return record
    admit(1,'immutable definition constants')
    for index in (6,8,10,18):
        guard('definition cell'+str(index));notebook=original if index==6 else production
        exec(compile(''.join(notebook['cells'][index]['source']),'pinned_cell%d.py'%index,'exec'),ns)
    if ns['TRAINING_REVISION']!='strong-teacher-hand-gae-v3' or len(ns['HEURISTIC_NAMES'])!=13:
        raise RuntimeError('Original thirteen styles and training source required.')
    sampling=_load(payload/'sampling_policy.py','lr_ab_sampling_'+uuid.uuid4().hex,expected_sha256=ids['sampling_policy.py']['sha256'])
    ns['sample_train_opponent_style_ids']=sampling.build_train_sampler(jax,jnp,ns['sample_train_opponent_style_ids'],sampling.identity_sampling_policy())
    plasticity=_load(payload/'plasticity.py','lr_ab_plasticity_'+uuid.uuid4().hex,expected_sha256=ids['plasticity.py']['sha256'])
    plasticity.install_training(ns,str(payload/'production_source.ipynb'))
    # Source1201 already has this revision and three adaptive settings. Activate
    # before loading; never call candidate_config again or migrate them twice.
    plasticity.activate_candidate_revision(ns)
    source_cfg=copy.deepcopy(shared['source']['source_cfg']);cfg=copy.deepcopy(shared['arm_configs'][arm])
    original_load=ns['gpu_admit_checkpoint_load']
    def load_admit(saved,size):
        guard('native transfer before admission');outcome=original_load(saved,size);guard('native transfer after admission');return outcome
    ns['gpu_admit_checkpoint_load']=load_admit
    guard('exact source1201 native load')
    state,loaded_cfg,history,metadata=ns['load_checkpoint'](shared['source_native']['path'])
    runtime._aot_sync(dict(ns=ns),state)
    if _canonical(loaded_cfg)!=_canonical(source_cfg) or _canonical(history)!=_canonical(shared['source']['source1201']['native_history']):
        raise RuntimeError('Exact source full config/history changed during restore.')
    equality=runtime._aot_assert_state_matches_source906(ns,state,shared['source_native']['path'],guard)
    # Inspect same-signature source health BEFORE candidate lr changes. Existing
    # Adam/RNG/counters and all historical rows remain intact in each branch.
    history=copy.deepcopy(history)
    source_journal=ns['_optimizer_resume_journal'](history,source_cfg,int(state.updates))
    source_health=ns['_optimizer_health'](source_journal)
    if source_health.get('should_stop'):raise RuntimeError('Original source configuration health forbids LR migration.')
    migration=shared['contract'].arm_migration_record(source_cfg,cfg,arm)
    expansion=_install_training_expansion(ns,runtime,shared['contract'],payload,ids,cfg,guard)
    # The namespace migration changes future assignments only. Verify every
    # restored array again before AOT tracing can bind the new training sampler.
    post_expansion_equality=runtime._aot_assert_state_matches_source906(ns,state,shared['source_native']['path'],guard)
    history.append(dict(kind='lr_ab_branch_start',arm=arm,authorization_sha256=hashlib.sha256(_canonical(shared['authorization']).encode()).hexdigest(),
        lr_ab_clock=copy.deepcopy(shared['clock']),source_identities=copy.deepcopy(shared['source']['source_identities']),migration=copy.deepcopy(migration)))
    trainer=ns['make_train_step'](cfg)
    guard('abstract template');abstract=jax.eval_shape(lambda:ns['initialize_training'](source_cfg))
    after_collect,batch,_=jax.eval_shape(trainer.collect,abstract)
    S,B=ns['_gpu_tree_bytes'](abstract),ns['_gpu_tree_bytes'](batch)
    admit(S,'collect compile preparation');guard('collect lowering');lowered=trainer.collect.lower(abstract)
    guard('collect compile');collect=lowered.compile();del lowered
    C=ns['_gpu_compiled_bytes'](collect.memory_analysis())
    admit(C['total_bytes']+S+65536,'optimizer compile preparation');guard('optimizer lowering');lowered=trainer.optimize.lower(after_collect,batch)
    guard('optimizer compile');optimize=lowered.compile();del lowered
    O=ns['_gpu_compiled_bytes'](optimize.memory_analysis());estimate=max(C['total_bytes'],O['total_bytes'])+S+65536
    admit(estimate,'sequential AOT selected')
    profile=dict(device=devices[0],optimize_input_specs=runtime._aot_input_specs(jax,np,((after_collect,batch),{})),
        options=opts,execution_handles=dict(collect=collect,optimize=optimize),state_bytes=S,batch_bytes=B,
        collect=C,optimize=O,estimated_peak_bytes=estimate,retained_rollback_states=1)
    journal=ns['_optimizer_resume_journal'](history,cfg,int(state.updates))
    session=dict(arm=arm,shared=shared,ns=ns,runtime=runtime,state=state,cfg=cfg,saved_cfg=copy.deepcopy(cfg),
        profile=profile,options=opts,handles=dict(profile['execution_handles']),saved_handles=dict(profile['execution_handles']),
        platform=copy.deepcopy(platform),saved_platform=copy.deepcopy(platform),runtime_identity=ns['_gpu_runtime_identity'](),
        root=str(root),history=history,journal_rows=journal,accepted_update=int(state.updates),accepted_env_steps=state.env_steps,
        recent_calls=[],recovery_sink={},terminal=False,status='prepared',source_leaf_specs=copy.deepcopy(metadata['leaves']),
        candidate_revision=ns['CHECKPOINT_TRAINING_REVISION'],training_started_monotonic=None,training_stopped_monotonic=None,
        migration=migration,expansion=expansion,equality=equality,post_expansion_equality=post_expansion_equality,
        expansion_namespace_binding=_expansion_namespace_binding(ns),
        source_health=source_health,next_export_elapsed=900.,last_metrics=None)
    session['runtime_callable_identity']=runtime._aot_eval_namespace_identity(ns)
    original_finite=runtime._aot_host_finite
    def host_finite(s,tree,stage,expected_leaves,payload_guard_bytes):
        if stage=='optimizer':
            if expected_leaves!=238:raise RuntimeError('Original optimizer leaf schema changed.')
            expected_leaves=242
        return original_finite(s,tree,stage,expected_leaves,payload_guard_bytes)
    runtime._aot_host_finite=host_finite
    runtime._aot_boundary=lambda s,label:_arm_guard(s,label)
    runtime._aot_elapsed=lambda s:_arm_elapsed(s)
    runtime._aot_limit_reason=lambda s:('arm_training1800_complete' if _arm_elapsed(s)>=ARM_TRAIN_SECONDS else None)
    runtime._aot_begin_cycle_at_launch=_begin_training_clock
    runtime._aot_write=lambda s:_write_arm_status(s)
    runtime._aot_event=lambda s,kind,**values:_append_event(root,kind,values)
    runtime._aot_heartbeat=lambda s,event,**values:print(json.dumps(dict(arm=arm,event=event,update=s['accepted_update'],arm_elapsed_seconds=_arm_elapsed(s),**values),allow_nan=False),flush=True)
    session['operator_callable_identity']=runtime._aot_eval_namespace_identity(vars(runtime))
    session['recovery_sink'].update(state=state,cfg=copy.deepcopy(cfg),history=history,session=session)
    _ACTIVE_RECOVERY[arm]=session['recovery_sink']
    shared['arm_records'][arm]=dict(cfg=cfg,updates=int(state.updates),env_steps=state.env_steps,native=None,
        equality=equality,post_expansion_equality=post_expansion_equality,source_health=source_health,
        migration=migration,expansion=expansion,training_clock=dict(start_monotonic=None,
        stop_monotonic=None,elapsed_seconds=0.,maximum_seconds=ARM_TRAIN_SECONDS),health=source_health,last_metrics=None,
        completed_rollouts=0,durable_updates=None,durable_env_steps=None,accepted_state_saved=False,memory_profile=dict(collect=C,optimize=O,state_bytes=S,batch_bytes=B,estimated_peak_bytes=estimate,
        initial_board_snapshot=snapshot,explicit_AOT_compiles=2))
    return session


def _begin_training_clock(s,begin):
    if s['training_started_monotonic'] is None:
        s['training_started_monotonic']=begin
        _write_new_json(Path(s['root'])/'training_clock.json',dict(start_monotonic=begin,
            maximum_seconds=ARM_TRAIN_SECONDS,deadline_monotonic=begin+ARM_TRAIN_SECONDS,
            scope='Continuous elapsed wall time from first collect; includes validation and exports.'))
    _arm_guard(s,'first collect after arm clock IO')
    return False


def _update_arm_record(s):
    record=s['shared']['arm_records'][s['arm']]
    record.update(updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
        completed_rollouts=s['accepted_update']-s['shared']['source_update'],last_metrics=s.get('last_metrics'),
        health=s['ns']['_optimizer_health'](s['journal_rows']),training_clock=dict(start_monotonic=s['training_started_monotonic'],
        stop_monotonic=s['training_stopped_monotonic'],elapsed_seconds=_arm_elapsed(s),maximum_seconds=ARM_TRAIN_SECONDS))
    record['accepted_state_saved']=(record.get('native') is not None and record.get('durable_updates')==s['accepted_update']
        and record.get('durable_env_steps')==s['accepted_env_steps'])
    return record


def _write_arm_status(s):
    record=_update_arm_record(s)
    _atomic_status(Path(s['root']),record)
    return _atomic_status(Path(s['shared']['root']),_status(s['shared']))


def _save_arm(s,reason,*,terminal=False):
    """No new GPU work; preserve last accepted state even after a boundary."""
    shared=s['shared'];nonce=uuid.uuid4().hex
    _update_arm_record(s)  # Preserve latest accepted counters before any save IO.
    s['history'].append(dict(kind='lr_ab_checkpoint',arm=s['arm'],reason=reason,nonce=nonce,
        updates=s['accepted_update'],env_steps=s['accepted_env_steps'],terminal=bool(terminal),lr_ab_clock=copy.deepcopy(shared['clock'])))
    s['recovery_sink'].update(state=s['state'],cfg=copy.deepcopy(s['saved_cfg']),history=s['history'])
    native=s['ns']['save_checkpoint'](s['state'],s['saved_cfg'],s['history'],Path(s['root'])/'checkpoints',
        cumulative_seconds=shared['source']['source1201']['native_manifest']['cumulative_seconds']+_elapsed(shared),
        tag='lr_ab_'+s['arm']+'_'+reason+'_'+nonce,keep=None)
    shared['helpers']._native_local_audit(s,native)
    shared['contract'].validate_arm_native(Path(native).read_bytes(),source_artifacts=shared['source'],
        arm=s['arm'],expected_config=s['saved_cfg'])
    info=_identity(native);shared['arm_records'][s['arm']].update(native=info,durable_updates=s['accepted_update'],
        durable_env_steps=s['accepted_env_steps'],accepted_state_saved=True)
    s['last_native']=info;_write_arm_status(s)
    return info


def _export_bundle(shared,reason):
    """Immutable portable snapshot, including actual exact evaluation checkpoints."""
    nonce=uuid.uuid4().hex;root=Path(shared['root']);data={}
    def snapshot(info):
        raw=Path(info['path']).read_bytes()
        if len(raw)!=info['bytes'] or hashlib.sha256(raw).hexdigest()!=info['sha256']:
            raise IOError('Actual checkpoint snapshot differs from verified identity:'+info['path'])
        return raw
    data['checkpoints/source1201_native.zip']=snapshot(shared['source_native'])
    for arm,record in shared['arm_records'].items():
        if record.get('native') is not None:
            data['checkpoints/'+arm+'_native.zip']=snapshot(record['native'])
    status=_status(shared);status.update(nonce=nonce,reason=reason,external_verification_complete=False)
    provenance=dict(format='spades-lr-ab-expansion-provenance-v2',source_identities=shared['source']['source_identities'],
        source_history_sha256=shared['source']['source_history_sha256'],source_pins=shared['code_identities'],
        source_health=shared['source']['source_health'],platform=shared['platform'],lr_ab_clock=shared['clock'],
        branch_214_equalities={arm:r['equality'] for arm,r in shared['arm_records'].items()},
        source_cfg=shared['source']['source_cfg'],arm_configs=shared['arm_configs'],
        training_profile=shared.get('training_profile'),
        historical_ledgers_unchanged=True,optimizer_and_rng_reset=False,
        original_combined_JIT_continuity='failed; unresolved',playing_strength_claim=False)
    data.update({'status.json':(json.dumps(status,indent=2,allow_nan=False)+'\n').encode(),
        'authorization.json':(json.dumps(shared['authorization'],indent=2,allow_nan=False)+'\n').encode(),
        'provenance.json':(json.dumps(provenance,indent=2,allow_nan=False)+'\n').encode()})
    for directory in ('control','candidate','evaluation'):
        parent=root/directory
        if parent.is_dir():
            for path in sorted(parent.rglob('*')):
                if path.is_file() and not path.is_symlink() and (directory=='evaluation' or path.name=='events.jsonl'):
                    data['evidence/'+str(path.relative_to(root))]=path.read_bytes()
    evaluated=dict(shared.get('evaluated_native_sha256',{}))
    manifest=dict(format='spades-lr-ab-expansion-bundle-v2',campaign_id=shared['authorization']['campaign_id'],
        run_id=shared['authorization']['run_id'],nonce=nonce,reason=reason,phase=shared['phase'],
        evaluated_native_sha256=evaluated,members={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in data.items()})
    data['manifest.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    destination=root/('lr_ab_'+reason+'_'+nonce+'.zip');partial=destination.with_suffix('.zip.partial')
    with partial.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for name,raw in data.items():archive.writestr(name,raw)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:raise IOError('LR A/B export CRC failure; accepted natives retained.')
    partial.rename(destination);info=_identity(destination)
    snapshot=root/('pending_'+nonce+'.json');_write_new_json(snapshot,dict(status,bundle=info))
    export=dict(bundle=info,status=_identity(snapshot),nonce=nonce,reason=reason)
    shared['exports'].append(export);_atomic_status(root,_status(shared))
    print(json.dumps(dict(event='immutable_lr_ab_bundle',**export),allow_nan=False),flush=True)
    return export


def _release_arm(s):
    """Release only a safely saved, stopped branch; original bytes stay on disk."""
    if s.get('last_native') is None:raise RuntimeError('Cannot release an unsaved accepted branch.')
    if not s.get('terminal'):raise RuntimeError('Cannot release a running branch.')
    record=s['shared']['arm_records'][s['arm']]
    if (record.get('accepted_state_saved') is not True or record.get('durable_updates')!=s['accepted_update']
            or record.get('durable_env_steps')!=s['accepted_env_steps'] or record.get('native')!=s['last_native']
            or _identity(s['last_native']['path'])!=s['last_native']):
        raise RuntimeError('Cannot release latest accepted state without exact verified durable save.')
    # No active job is terminated; all launches synchronize before this point.
    # Explicitly release device leaves/handles to avoid charging armA residency
    # against armB. Host/native copies, not armA state, initialize armB.
    jax=s['ns']['jax'];leaves=jax.tree_util.tree_leaves(s['state'])
    seen=set()
    for leaf in leaves:
        if id(leaf) in seen:continue
        seen.add(id(leaf));deleted=getattr(leaf,'is_deleted',None)
        if callable(deleted) and deleted():continue
        delete=getattr(leaf,'delete',None)
        if callable(delete):delete()
    s.pop('active_device_result',None);s['state']=None;s['recovery_sink'].pop('state',None)
    s['recovery_sink'].pop('active_device_result',None)
    s['handles'].clear();s['saved_handles'].clear();s['profile']['execution_handles'].clear()
    s['ns'].clear();gc.collect();jax.clear_caches();gc.collect()


def _train_arm(shared,arm):
    session=None
    try:
        shared['phase']='prepare_'+arm
        session=_prepare_arm(shared,arm)
        session['status']='training';shared['phase']='train_'+arm
        while session['accepted_update']-shared['source_update']<MAX_ROLLOUTS:
            _arm_guard(session,arm+' next rollout')
            health_stop=session['runtime']._aot_step(session)
            completed=session['accepted_update']-shared['source_update']
            if completed==1:
                _save_arm(session,'qa');_export_bundle(shared,arm+'_qa')
            elif _arm_elapsed(session)>=session['next_export_elapsed']:
                _save_arm(session,'supplemental15');_export_bundle(shared,arm+'_supplemental15')
                session['next_export_elapsed']=_arm_elapsed(session)+900.
            if health_stop:raise BoundaryStop(arm+':'+health_stop)
        session['status']='planned_rollouts_complete'
    finally:
        if session is not None:
            session['training_stopped_monotonic']=(time.monotonic() if session['training_started_monotonic'] is not None else None);session['terminal']=True
            _save_arm(session,'final',terminal=True)
    return session


def _evaluate(shared,session):
    shared['phase']='evaluation';_shared_guard(shared,'prepare evaluation')
    records=shared['arm_records']
    if set(records)!={'control','candidate'} or any(r['completed_rollouts']!=MAX_ROLLOUTS for r in records.values()):
        raise BoundaryStop('incomplete_equal_rollout_budget; evaluation not launched')
    evaluator=_load(shared['payload']/'lr_ab_evaluation.py','lr_ab_eval_'+uuid.uuid4().hex,
        expected_sha256=shared['code_identities']['lr_ab_evaluation.py']['sha256'])
    runtime=session['runtime'];ns=session['ns']
    # All model trees are reconstructed as immutable host copies from the exact
    # saved native files, never live mutable ending-state references.
    source_params=runtime._aot_frozen_baseline_params(ns,session['state'],shared['source_native']['path'],
        lambda label:_arm_guard(session,label,training=False))
    control_params=runtime._aot_frozen_baseline_params(ns,session['state'],records['control']['native']['path'],
        lambda label:_arm_guard(session,label,training=False))
    candidate_params=runtime._aot_frozen_baseline_params(ns,session['state'],records['candidate']['native']['path'],
        lambda label:_arm_guard(session,label,training=False))
    evaluated=dict(source=shared['source_native']['sha256'],control=records['control']['native']['sha256'],candidate=records['candidate']['native']['sha256'])
    shared['evaluated_native_sha256']=evaluated
    # Persist all three evaluated bytes BEFORE launching any evaluation kernel.
    _export_bundle(shared,'pre_evaluation')
    def admission(estimate,label):
        _arm_guard(session,label,training=False)
        outcome=ns['_gpu_memory_admission'](estimate,session['options'],label)
        details=outcome.get('budget_details',{});snap=outcome.get('snapshot',{})
        if (outcome.get('status')!='admitted' or outcome.get('overlap_credit_bytes')!=0 or snap.get('platform')!='gpu'
                or details.get('pool_counter_source')!='pool_bytes' or details.get('pool_counter_status')!='validated_current_total'
                or 'pool_bytes' not in snap.get('jax_memory_stats_raw',{})):
            raise RuntimeError('Current validated pool/board evaluation admission required.')
        _arm_guard(session,label+' after admission',training=False);return outcome
    report_path=Path(shared['root'])/'evaluation'/'evaluation_report.json'
    try:
        report=evaluator.evaluate_lr_ab_pair(ns=ns,source_params=source_params,source_cfg=shared['source']['source_cfg'],
            source_checkpoint=shared['source_native'],control_params=control_params,lr_params=candidate_params,
            control_cfg=shared['arm_configs']['control'],lr_cfg=shared['arm_configs']['candidate'],
            control_checkpoint=records['control']['native'],lr_checkpoint=records['candidate']['native'],
            output_dir=report_path.parent,precompile_estimate_bytes=session['profile']['estimated_peak_bytes'],
            admit=admission,launch_guard=lambda label:_arm_guard(session,label,training=False),
            experiment_id=shared['authorization']['campaign_id'],run_id=shared['authorization']['run_id'],
            source_native_identity={k:shared['source_native'][k] for k in ('bytes','sha256')},
            arm_config_signatures={arm:hashlib.sha256(_canonical(cfg).encode()).hexdigest() for arm,cfg in shared['arm_configs'].items()})
        return report
    finally:
        if report_path.exists():
            archive=report_path.parent/'portable_lr_ab_evidence.zip'
            evaluator.package_lr_ab_evaluation_evidence(report_path,archive,source_checkpoint=shared['source_native']['path'],
                control_checkpoint=records['control']['native']['path'],lr_checkpoint=records['candidate']['native']['path'])
            shared['evaluation']=dict(report=_identity(report_path),archive=_identity(archive),
                complete=_read_json(report_path).get('complete',False),evaluated_native_sha256=evaluated)


def _start_shared(config,payload,identities,contract,source,arms,entry,root):
    # Reverify the bytes while fixing stat baselines before executing any helper.
    stat_pins=_pin_sources(identities)
    runtime=_load(payload/'spades_legacy_aot.py','lr_ab_inventory_'+uuid.uuid4().hex,
        expected_sha256=identities['spades_legacy_aot.py']['sha256'])
    platform_record=_choose_device(runtime,config.get('device_uuid'));platform=platform_record['platform']
    context=_context(platform);context.update(utc_now=entry['utc'],monotonic_now=entry['monotonic'])
    clock=contract.make_lr_ab_clock(config['authorization'],context)
    shared=dict(root=str(root),entry=entry,authorization=copy.deepcopy(config['authorization']),payload=payload,
        code_identities=copy.deepcopy(identities),identities=dict(identities),stat_pins=stat_pins,
        contract=contract,clock=clock,highwater=context,platform=platform,source=source,arm_configs=copy.deepcopy(arms),
        training_profile=contract.training_profile_metadata(source['source_cfg']),
        arm_records={},phase='host_prepared',terminal=False,exports=[],source_update=source['source1201']['native_manifest']['updates'])
    _write_new_json(root/'lr_ab_clock.json',clock)
    _shared_guard(shared,'source materialization')
    native=root/'source1201_native.zip'
    with native.open('xb') as stream:stream.write(source['source1201']['native_bytes'])
    info=_identity(native);expected=source['source1201']['native_identity']
    if any(info[k]!=expected[k] for k in ('bytes','sha256')):raise IOError('Exact source1201 native bytes changed on materialization.')
    shared['source_native']=info;shared['identities']['source1201_native.zip']=info
    shared['stat_pins'].update(_pin_sources({'source1201_native.zip':info}))
    helpers=_load(payload/'pinned_fresh_helpers.py','lr_ab_frozen_helpers_'+uuid.uuid4().hex,
        expected_sha256=identities['pinned_fresh_helpers.py']['sha256'])
    shared['helpers']=helpers
    for key in ('XLA_FLAGS','CUBLAS_WORKSPACE_CONFIG','TF_DETERMINISTIC_OPS','JAX_DEFAULT_MATMUL_PRECISION'):
        if os.environ.get(key) is not None:raise RuntimeError('Original unset compiler/determinism/precision environment required:'+key)
    for key,allowed,default in (('XLA_PYTHON_CLIENT_PREALLOCATE',('false',),'false'),('XLA_PYTHON_CLIENT_MEM_FRACTION',('0.96','.96'),'0.96')):
        if key in os.environ and os.environ[key] not in allowed:raise RuntimeError('Allocator environment mismatch:'+key)
        os.environ.setdefault(key,default)
    _no_jax();_shared_guard(shared,'selected UUID before JAX')
    os.environ['CUDA_VISIBLE_DEVICES']=platform['selected_gpu_uuid']
    _write_new_json(root/'authorization.json',shared['authorization'])
    _atomic_status(root,_status(shared));return shared


def run_once(config_path,payload_dir):
    """One experiment only. Driver6600s includes imports/compile/train/eval/IO.

    The frontend separately limits setup to600s from its first code cell, so
    setup plus this driver permits at most7200s before new work is forbidden.
    Each arm independently restores source1201, then performs exactly128 planned
    rollouts within its continuous1800s training ceiling. No extension or retry.
    In-flight work may finish late; accepted-state preservation remains allowed.
    """
    global _RUN_INVOKED
    if _RUN_INVOKED:raise RuntimeError('A/B entrypoint already invoked; no automatic retry or second experiment.')
    _RUN_INVOKED=True;entry=dict(utc=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic())
    raw=_read_json(config_path);config_identity=_identity(Path(config_path).absolute());root=Path(raw['output_dir']);run_id=raw['authorization']['run_id']
    if (not root.is_absolute() or root.is_symlink() or '..' in root.parts or not isinstance(run_id,str)
            or not run_id or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-' for c in run_id)):
        raise ValueError('New absolute output root and safe authorized run identity required.')
    registry=(Path('/kaggle/working') if Path('/kaggle/working').is_dir() else root.parent)/'.spades_lr_ab_expansion_v2_starts'
    registry.mkdir(parents=True,exist_ok=True)
    campaign_id=raw['authorization'].get('campaign_id')
    if not isinstance(campaign_id,str) or not campaign_id or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-' for c in campaign_id):
        raise ValueError('Safe fixed campaign identity required.')
    _write_new_json(registry/('campaign_'+campaign_id+'.json'),dict(campaign_id=campaign_id,run_id=run_id,entry=entry,output_dir=str(root),restart_permitted=False))
    _write_new_json(registry/(run_id+'.json'),dict(run_id=run_id,entry=entry,output_dir=str(root),restart_permitted=False))
    root.mkdir(parents=True,exist_ok=False)
    _write_new_json(root/'start_journal.json',dict(format='spades-lr-ab-expansion-start-v2',run_id=run_id,
        start_utc=entry['utc'],start_monotonic=entry['monotonic'],maximum_seconds=TOTAL_SECONDS,
        planned_rollouts_per_arm=MAX_ROLLOUTS,arm_training_seconds=ARM_TRAIN_SECONDS,restart_permitted=False,
        config_identity=_identity(Path(config_path).absolute())))
    shared=None;control=None;candidate=None
    try:
        config,payload,identities,contract,source,arms=_host_inputs(config_path,payload_dir,started_root=root)
        if _canonical(config)!=_canonical(raw) or _identity(Path(config_path).absolute())!=config_identity:
            raise RuntimeError('Configuration changed after exclusive experiment claim.')
        identities['lr_ab_config.json']=config_identity
        shared=_start_shared(config,payload,identities,contract,source,arms,entry,root)
        control=_train_arm(shared,'control')
        _export_bundle(shared,'control_final')
        _release_arm(control)
        candidate=_train_arm(shared,'candidate')
        _export_bundle(shared,'candidate_final')
        report=_evaluate(shared,candidate)
        shared['phase']='complete' if report.get('complete') is True else 'evaluation_incomplete'
        shared['stop_reason']='bounded_lr_ab_complete' if report.get('complete') is True else 'evaluation_incomplete'
    except BaseException as exc:
        import traceback
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),retry_attempted=False)
        if shared is None:
            _write_new_json(root/'preparation_failure.json',dict(status='failed',error=error,entry=entry,
                source_preserved=True,restart_permitted=False,gpu_training_started=False))
            return dict(status='preparation_failed',final_bundle_path=None,output_dir=str(root),error=error,
                restart_permitted=False,playing_strength_claim=False)
        shared['error']=error;shared['stop_reason']=str(exc)
        shared['phase']='bounded_stop' if isinstance(exc,BoundaryStop) else 'failed_accepted_state_preserved'
    finally:
        if shared is not None:
            shared['terminal']=True
            try:
                final=_export_bundle(shared,'final');shared['final_bundle_path']=final['bundle']['path']
            except BaseException as exc:
                shared['final_bundle_path']=None
                shared['error']=dict(previous=shared.get('error'),final_export_type=type(exc).__name__,
                    final_export_message=str(exc),accepted_natives_and_active_recovery_retained=True)
                _atomic_status(root,_status(shared))
    result=_status(shared);result.update(final_bundle_path=shared['final_bundle_path'],output_dir=str(root),
        restart_permitted=False,active_recovery_retained=True)
    _write_new_json(root/'lr_ab_result.json',result)
    return result
