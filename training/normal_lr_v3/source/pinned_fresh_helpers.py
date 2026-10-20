"""Definitions-only fresh, bounded Kaggle continuation of durable Spades906.

Host preflight imports no numerical framework. run_once owns one nonrenewable
clock; checkpoints are immutable and independently audited after it returns.
Historical ledgers and source checkpoints are never modified. No provider calls.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone

FORMAT = 'spades-fresh-onehour-config-v1'
CORE_PINS = {
    'spades_legacy_aot.py': '66f00e9a245d85d236eea4cda545bce1fef4578d44f7d591d36c7a803e782db1',
    'original_source.ipynb': '3236d2c95d4d1a95bf31a5cb7299853be6871783691829ee775fbb38366d1389',
    'production_source.ipynb': 'ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8',
    'diversity_evaluation.py': '1751557e800d48a596b7a783595aa1bd42ecf6c20b130c97a9a498e583e16ecc',
    'sampling_policy.py': '824c41ee9318509901af82a1ca99d4814c88e2ac8112efcbc03cf9e13cc709a0',
    'audit_export_original.py': 'a354a362dad26b68dc8ad3030654adc9d7646a362b2a0ca024b7977ce04b9fb7',
    'opponent_manifest.json': '20e4727fdf4d37d035c9ec1bc5d770eb3510f93c9bd20c680c512ceee5f78f1d',
    'heldout_protocol.json': '0b5970eb566bfcf3db55ffc5559a80e27c072babffe5e1c64f92d3aa5ed2942d',
}
_RUN_INVOKED = False
_ACTIVE_RECOVERY = None


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


def _host_inputs(config_path, payload_dir, *, started_root=None):
    payload = Path(payload_dir).absolute()
    config_info = _identity(Path(config_path).absolute())
    config = _read_json(config_path)
    required = {'format', 'source906_package_path', 'baseline674_package_path', 'output_dir', 'authorization'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'device_uuid'}:
        raise ValueError('Exact fresh configuration keys required.')
    if config['format'] != FORMAT:
        raise ValueError('Unknown fresh configuration format.')
    root = Path(config['output_dir'])
    if not root.is_absolute() or root.is_symlink() or (root.exists() and root != started_root):
        raise ValueError('New absolute output directory required; never reuse a campaign root.')
    identities = {}
    for name, expected in CORE_PINS.items():
        info = _identity(payload / name)
        if info['sha256'] != expected:
            raise ValueError('Pinned payload source differs: ' + name)
        identities[name] = info
    # New campaign components are bound by the reviewed payload manifest, which
    # the notebook loader also verifies before loading this entry point.
    manifest_path = payload / 'fresh_payload_manifest.json'
    manifest = _read_json(manifest_path)
    if manifest.get('format') != 'spades-fresh-payload-manifest-v1':
        raise ValueError('Reviewed fresh payload manifest required.')
    for name in ('spades_fresh_run.py', 'fresh_run_contract.py', 'plasticity.py', 'fresh_evaluation.py'):
        info = _identity(payload / name)
        expected = manifest.get('members', {}).get(name)
        if expected != dict(bytes=info['bytes'], sha256=info['sha256']):
            raise ValueError('Fresh campaign payload binding differs: ' + name)
        identities[name] = info
    contract = _load(payload / 'fresh_run_contract.py', 'spades_fresh_contract_' + uuid.uuid4().hex, expected_sha256=identities['fresh_run_contract.py']['sha256'])
    pin_names = dict(original_sha256='original_source.ipynb', production_sha256='production_source.ipynb',
                     operator_sha256='spades_fresh_run.py', contract_sha256='fresh_run_contract.py',
                     evaluator_sha256='fresh_evaluation.py', sampling_sha256='sampling_policy.py',
                     plasticity_sha256='plasticity.py', legacy_operator_sha256='spades_legacy_aot.py',
                     frozen_evaluator_sha256='diversity_evaluation.py')
    expected_pins = {key: identities[name]['sha256'] for key, name in pin_names.items()}
    contract.validate_fresh_authorization(config['authorization'], expected_source_pins=expected_pins)
    source = contract.validate_source_artifacts(config['source906_package_path'], config['baseline674_package_path'])
    plasticity = _load(payload / 'plasticity.py', 'spades_host_plasticity_' + uuid.uuid4().hex, expected_sha256=identities['plasticity.py']['sha256'])
    candidate_cfg = plasticity.candidate_config(source['source_cfg'])
    if hashlib.sha256(_canonical(candidate_cfg).encode()).hexdigest() != config['authorization']['configuration_sha256']:
        raise ValueError('Authorized candidate configuration differs from exact bounded migration.')
    return config, config_info, payload, identities, contract, source


def preflight(config_path, payload_dir):
    """Read-only host content checks. No JAX import, GPU inventory or allocation."""
    if any(name in ('jax', 'jaxlib', 'jax_plugins') or name.startswith(('jax.', 'jaxlib.', 'jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Run CPU preflight in a fresh process before JAX import.')
    config, config_info, payload, identities, contract, source = _host_inputs(config_path, payload_dir)
    if any(name in ('jax', 'jaxlib', 'jax_plugins') or name.startswith(('jax.', 'jaxlib.', 'jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Host-only payload imported JAX during preflight; GPU run forbidden.')
    return dict(format='spades-fresh-host-preflight-v1', status='PASS',
                config=config_info, payload=identities, output_dir=config['output_dir'],
                authorization=config['authorization'], source_artifacts=_source_summary(source),
                jax_imported=False, gpu_work_started=False, training_authorized_by_preflight=False)


def _source_summary(source):
    return {arm: {key: value for key, value in source[arm].items() if key not in ('native_bytes', 'native_history')}
            for arm in ('source906', 'baseline674')}


def _context(platform):
    return dict(utc_now=datetime.now(timezone.utc).isoformat(), monotonic_now=time.monotonic(),
                host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                selected_gpu_uuid=platform['selected_gpu_uuid'],
                selected_gpu_pci_bus_id=platform['selected_gpu_pci_bus_id'])


class FreshBoundaryStop(RuntimeError):
    """A clean workload boundary; accepted state still must be exported."""


def _choose_device(runtime, device_uuid):
    inventory = runtime.inventory_kaggle_nojax()
    boards = inventory['physical_gpu_inventory']
    if device_uuid is None:
        boards = sorted((row for row in boards if 'T4' in row['name']), key=lambda row: row['pci_bus_id'])
        if not boards:
            raise RuntimeError('No physical free-provider T4 visible; no CPU or paid fallback.')
        device_uuid = boards[0]['uuid']
    return runtime.preflight_kaggle_nojax(device_uuid)


def _fresh_export(session, reason, *, terminal=False):
    """Save accepted state and a unique, matching portable evidence snapshot."""
    s = session
    runtime = s['runtime']
    ns = s['ns']
    root = Path(s['root'])
    nonce = uuid.uuid4().hex
    s['history'].append(dict(kind='fresh_checkpoint', reason=reason, nonce=nonce,
                             updates=s['accepted_update'], env_steps=s['accepted_env_steps'],
                             fresh_run_clock=copy.deepcopy(s['fresh_clock']), terminal=bool(terminal)))
    s['recovery_sink'].update(state=s['state'], cfg=copy.deepcopy(s['cfg']), history=s['history'])
    # Saving an already accepted result remains permitted after a deadline. It
    # starts no training/evaluation workload and cannot erase older artifacts.
    native = ns['save_checkpoint'](s['state'], s['cfg'], s['history'], root / 'checkpoints',
                                  cumulative_seconds=s['historical_cumulative_seconds'] + _fresh_elapsed(s),
                                  tag='fresh_' + reason + '_' + nonce, keep=None)
    _native_local_audit(s, native)
    native_info = _identity(native)
    s['last_native'] = native_info
    status = _status(s)
    status.update(status='external_verification_pending', nonce=nonce, reason=reason,
                  terminal=bool(terminal), native=native_info, local_content_verified=True,
                  external_verification_complete=False)
    evidence = {}
    for folder in sorted(root.glob('evaluation_*')):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not path.is_symlink():
                evidence[str(path.relative_to(root))] = path
    if (root / 'events.jsonl').is_file():
        evidence['events.jsonl'] = root / 'events.jsonl'
    data = {'native_checkpoint.zip': native.read_bytes(),
            'status.json': (json.dumps(status, indent=2, allow_nan=False) + '\n').encode(),
            'authorization.json': (json.dumps(s['authorization'], indent=2, allow_nan=False) + '\n').encode(),
            'provenance.json': (json.dumps(s['provenance'], indent=2, allow_nan=False) + '\n').encode()}
    for name, path in evidence.items():
        data['evidence/' + name] = path.read_bytes()
    manifest = dict(format='spades-fresh-onehour-bundle-v1', campaign_id=s['authorization']['campaign_id'], run_id=s['run_id'], checkpoint_role=reason, nonce=nonce,
                    reason=reason, updates=s['accepted_update'], env_steps=s['accepted_env_steps'],
                    RUN_NAME='modified_spades_run_001', fresh_run_clock=s['fresh_clock'],
                    members={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                             for name, raw in data.items()})
    data['manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    bundle = root / ('update%09d_%s_%s.zip' % (s['accepted_update'], reason, nonce))
    with bundle.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for name, raw in data.items():
                archive.writestr(name, raw)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise IOError('Fresh bundle CRC failure; accepted state and native retained.')
    bundle_info = _identity(bundle)
    pending = dict(status, bundle=bundle_info)
    snapshot = root / ('pending_%s.json' % nonce)
    _write_new_json(snapshot, pending)
    s['exports'].append(dict(bundle=bundle_info, status=_identity(snapshot), native=native_info,
                             nonce=nonce, reason=reason))
    s['next_export_elapsed'] = _fresh_elapsed(s) + 900.0
    _atomic_status(root, _status(s))
    print(json.dumps(dict(event='immutable_checkpoint', **s['exports'][-1]), allow_nan=False), flush=True)
    return s['exports'][-1]


def _fresh_elapsed(s):
    return max(0.0, time.monotonic() - s['entry_monotonic'])


def _status(s):
    return dict(format='spades-fresh-run-status-v1', run_id=s['run_id'], campaign_id=s['authorization']['campaign_id'], status=s['status'],
                RUN_NAME='modified_spades_run_001', phase=s.get('phase'),
                updates=s['accepted_update'], env_steps=s['accepted_env_steps'],
                fresh_run_clock=s['fresh_clock'], elapsed_seconds=_fresh_elapsed(s),
                one_bounded_cycle_only=True, training_enabled=not s.get('terminal', False),
                terminal=s.get('terminal', False), stop_reason=s.get('stop_reason'),
                phase_results=s.get('phase_results', {}), last_metrics=s.get('last_metrics'),
                exports=s['exports'], promotion=False, playing_strength_claim=False,
                external_verification_required_before_new_run=True,
                historical_ledger_unchanged=True, cfg=s['saved_cfg'],
                migration=s['migration'], error=s.get('error'))


def _memory_options_guard(options):
    if any(options.get(key) != value for key, value in dict(
            target_gib=14., reserve_mib=1024, safety_factor=1.45, extra_guard_mib=256).items()):
        raise RuntimeError('Reviewed exact memory options changed; no launch.')


def _native_local_audit(s, path):
    import io
    np = s['ns']['np']
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {'manifest.json', 'history.json', 'state.npz'} or archive.testzip() is not None:
            raise IOError('Exact native container/CRC required.')
        manifest = json.loads(archive.read('manifest.json'))
        history = json.loads(archive.read('history.json'))
        payload = archive.read('state.npz')
    if (manifest['cfg'] != s['saved_cfg'] or manifest['updates'] != s['accepted_update']
            or manifest['env_steps'] != s['accepted_env_steps'] or history != s['history']
            or manifest['sha256'] != hashlib.sha256(payload).hexdigest()
            or manifest['training_revision'] != s['candidate_revision']
            or manifest.get('opponent_diversity') != s['runtime']._aot_diversity_contract()
            or manifest.get('training_state_schema') != 2
            or manifest.get('versions') != {'jax': '0.11.1', 'flax': '0.12.10', 'optax': '0.2.8'}):
        raise IOError('Fresh native identity/config/history/counter/revision mismatch.')
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        names = [f'leaf_{i:05d}' for i in range(214)]
        if sorted(arrays.files) != names or manifest['leaves'] != s['source_leaf_specs']:
            raise IOError('Fresh native exact214 schema differs.')
        for name, spec in zip(names, manifest['leaves']):
            value = arrays[name]
            if (list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or value.dtype.hasobject
                    or np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all()):
                raise IOError('Fresh native nonfinite/schema failure: ' + name)
        s['runtime']._aot_validate_style_array(arrays['leaf_00156'], s['saved_cfg']['n_env'])
        vals = [arrays[f'leaf_{i:05d}'] for i in (207, 208, 209, 210, 211, 212, 213)]
        types = ('int32', 'int32', 'int32', 'bool', 'uint32', 'uint32', 'int32')
        if any(v.shape != () or str(v.dtype) != typ for v, typ in zip(vals, types)):
            raise IOError('Fresh native scalar counter schema differs.')
        update, decisions, boot, complete, lo, hi, base = [v.item() for v in vals]
        lifetime = ((int(hi) << 32) | int(lo)) + (int(update) - int(base)) * int(decisions)
        if (update != s['accepted_update'] or decisions != s['saved_cfg']['n_env'] * s['saved_cfg']['rollout_len']
                or boot != 0 or complete is not True or not 0 <= base <= update or lifetime != s['accepted_env_steps']):
            raise IOError('Fresh native scalar counters do not bind the accepted state.')
    return manifest


def _prepare(config, payload, identities, contract, source, entry, root):
    """Restore source214 and compile the pinned sequential AOT memory route."""
    tracked = dict(identities)
    def pin_stat(info):
        path = Path(info['path'])
        before = path.stat()
        if _identity(path) != info:
            raise RuntimeError('Payload changed between host validation and preparation: ' + str(path))
        after = path.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise RuntimeError('Source changed while fixing stat identity: ' + str(path))
        return tuple(getattr(after, key) for key in fields)
    stat_pins = {name: pin_stat(info) for name, info in tracked.items()}
    runtime = _load(payload / 'spades_legacy_aot.py', 'spades_fresh_legacy_' + uuid.uuid4().hex, expected_sha256=identities['spades_legacy_aot.py']['sha256'])
    platform_preflight = _choose_device(runtime, config.get('device_uuid'))
    platform = platform_preflight['platform']
    context = _context(platform)
    context.update(utc_now=entry['utc'], monotonic_now=entry['monotonic'])
    clock = contract.make_fresh_run_clock(config['authorization'], context)
    _write_new_json(root / 'fresh_run_clock.json', clock)
    highwater = context
    def guard(label):
        nonlocal highwater
        now = _context(platform)
        try:
            contract.validate_fresh_context(config['authorization'], clock, now, previous_context=highwater)
        except ValueError as exc:
            raise FreshBoundaryStop(str(exc) + ': ' + label) from exc
        highwater = now
        for name, info in tracked.items():
            path = Path(info['path'])
            stat = path.stat()
            actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if path.is_symlink() or actual != stat_pins[name]:
                raise RuntimeError('Reviewed source changed before ' + label + ': ' + name)
        if time.monotonic() - entry['monotonic'] >= 3600:
            raise FreshBoundaryStop('one_hour_complete:' + label)
    guard('host artifact materialization')
    source906 = source['source906']
    baseline674 = source['baseline674']
    cfg = copy.deepcopy(source['source_cfg'])
    source_native = root / 'source906_native.zip'
    baseline_native = root / 'baseline674_native.zip'
    for path, record in ((source_native, source906), (baseline_native, baseline674)):
        with path.open('xb') as stream:
            stream.write(record['native_bytes'])
        info = _identity(path)
        expected = record['native_identity']
        if info['bytes'] != expected['bytes'] or info['sha256'] != expected['sha256']:
            raise IOError('Pinned native materialization differs.')
        tracked[path.name] = info
        stat_pins[path.name] = pin_stat(info)
    source906.pop('native_bytes')
    baseline674.pop('native_bytes')
    _write_new_json(root / 'source_identities.json', source['source_identities'])
    stack = dict(jax='0.11.1', jaxlib='0.11.1', flax='0.12.10', optax='0.2.8')
    if any(os.environ.get(key) is not None for key in ('XLA_FLAGS', 'CUBLAS_WORKSPACE_CONFIG',
                'TF_DETERMINISTIC_OPS', 'JAX_DEFAULT_MATMUL_PRECISION')):
        raise RuntimeError('Original unset compiler/determinism/precision environment required.')
    for key, allowed, default in (('XLA_PYTHON_CLIENT_PREALLOCATE', ('false',), 'false'),
                                   ('XLA_PYTHON_CLIENT_MEM_FRACTION', ('0.96', '.96'), '0.96')):
        if key in os.environ and os.environ[key] not in allowed:
            raise RuntimeError('Allocator environment mismatch:' + key)
        os.environ.setdefault(key, default)
    guard('selected UUID before JAX import')
    if any(name in ('jax', 'jaxlib', 'jax_plugins') or name.startswith(('jax.', 'jaxlib.', 'jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Fresh kernel required; no implicit reset or reentry.')
    os.environ['CUDA_VISIBLE_DEVICES'] = platform['selected_gpu_uuid']
    guard('JAX import')
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax
    from flax import linen as nn
    import optax
    if ({'jax': jax.__version__, 'jaxlib': importlib.metadata.version('jaxlib'),
         'flax': flax.__version__, 'optax': optax.__version__} != stack
            or jax.config.jax_enable_x64 or jax.config.jax_disable_jit
            or jax.config.jax_default_matmul_precision is not None):
        raise RuntimeError('Imported exact stack/config/precision mismatch.')
    guard('device discovery')
    devices = jax.devices()
    if len(devices) != 1 or jax.default_backend() != 'gpu' or 'T4' not in str(devices[0].device_kind):
        raise RuntimeError('Exactly one selected physical T4 required.')
    original = _read_json(payload / 'original_source.ipynb')
    production = _read_json(payload / 'production_source.ipynb')
    ns = dict(__name__='spades_fresh_isolated', _fresh_frozen_evaluation_path=str(payload / 'diversity_evaluation.py'), Path=Path, os=os, time=time, json=json, math=math,
              np=np, jax=jax, jnp=jnp, flax=flax, nn=nn, optax=optax,
              _SPADES_KAGGLE_PLATFORM=copy.deepcopy(platform))
    guard('UUID bound memory definitions')
    exec(compile(''.join(production['cells'][12]['source']), 'pinned_runtime_cell12.py', 'exec'), ns)
    snapshot = ns['_gpu_memory_snapshot']()
    if any(snapshot.get(key) != platform[field] for key, field in (
            ('gpu_uuid', 'selected_gpu_uuid'), ('gpu_pci_bus_id', 'selected_gpu_pci_bus_id'),
            ('board_total_bytes', 'selected_gpu_total_bytes'))):
        raise RuntimeError('Selected physical UUID/PCI board mapping failed.')
    opts = dict(ns['GPU_TUNING'])
    if any(opts.get(key) != val for key, val in dict(target_gib=14., reserve_mib=1024,
                                                   safety_factor=1.45, extra_guard_mib=256).items()):
        raise RuntimeError('Pinned exact memory settings changed.')
    def admit(estimate, label):
        guard(label + ' before admission')
        _memory_options_guard(opts)
        record = ns['_gpu_memory_admission'](estimate, opts, label)
        details = record.get('budget_details', {})
        snap = record.get('snapshot', {})
        if (record.get('status') != 'admitted' or record.get('overlap_credit_bytes') != 0
                or snap.get('platform') != 'gpu' or details.get('pool_counter_source') != 'pool_bytes'
                or details.get('pool_counter_status') != 'validated_current_total'
                or 'pool_bytes' not in snap.get('jax_memory_stats_raw', {})):
            raise RuntimeError('Actual current validated pool/board memory admission required.')
        guard(label + ' after admission')
        return record
    admit(1, 'immutable definition constants')
    for index in (6, 8, 10, 18):
        guard('definition cell ' + str(index))
        notebook = original if index == 6 else production
        exec(compile(''.join(notebook['cells'][index]['source']), 'pinned_cell%d.py' % index, 'exec'), ns)
    if ns['TRAINING_REVISION'] != 'strong-teacher-hand-gae-v3' or len(ns['HEURISTIC_NAMES']) != 13:
        raise RuntimeError('Original training revision/thirteen heuristic styles required.')
    sampling = _load(payload / 'sampling_policy.py', 'spades_fresh_sampling_' + uuid.uuid4().hex, expected_sha256=identities['sampling_policy.py']['sha256'])
    policy = sampling.identity_sampling_policy()
    ns['sample_train_opponent_style_ids'] = sampling.build_train_sampler(jax, jnp,
                                                    ns['sample_train_opponent_style_ids'], policy)
    plasticity = _load(payload / 'plasticity.py', 'spades_fresh_plasticity_' + uuid.uuid4().hex, expected_sha256=identities['plasticity.py']['sha256'])
    candidate_cfg = plasticity.candidate_config(cfg)
    migration = plasticity.migration_record(cfg, candidate_cfg)
    plasticity.install_training(ns, str(payload / 'production_source.ipynb'))
    guard('abstract native template')
    trainer = ns['make_train_step'](candidate_cfg)
    abstract = jax.eval_shape(lambda: ns['initialize_training'](cfg))
    after_collect, batch, _ = jax.eval_shape(trainer.collect, abstract)
    state_bytes, batch_bytes = ns['_gpu_tree_bytes'](abstract), ns['_gpu_tree_bytes'](batch)
    admit(state_bytes, 'AOT collect preparation')
    guard('collect lowering')
    lowered = trainer.collect.lower(abstract)
    guard('collect compile')
    collect_handle = lowered.compile()
    del lowered
    collect_profile = ns['_gpu_compiled_bytes'](collect_handle.memory_analysis())
    admit(collect_profile['total_bytes'] + state_bytes + 65536, 'AOT optimize preparation')
    guard('optimizer lowering')
    lowered = trainer.optimize.lower(after_collect, batch)
    guard('optimizer compile')
    optimize_handle = lowered.compile()
    del lowered
    optimize_profile = ns['_gpu_compiled_bytes'](optimize_handle.memory_analysis())
    estimated_peak = max(collect_profile['total_bytes'], optimize_profile['total_bytes']) + state_bytes + 65536
    admit(estimated_peak, 'sequential AOT selected')
    original_load_admit = ns['gpu_admit_checkpoint_load']
    def load_admit(saved_cfg, size):
        guard('native transfer before admission')
        result = original_load_admit(saved_cfg, size)
        guard('native transfer after admission')
        return result
    ns['gpu_admit_checkpoint_load'] = load_admit
    guard('source906 exact load')
    state, loaded_cfg, loaded_history, metadata = ns['load_checkpoint'](source_native)
    runtime._aot_sync(dict(ns=ns), state)
    if loaded_cfg != cfg or loaded_history != source906['native_history']:
        raise RuntimeError('Source906 full config/history changed during native restore.')
    equality = runtime._aot_assert_state_matches_source906(ns, state, source_native, guard)
    plasticity.activate_candidate_revision(ns)
    profile = dict(device=devices[0], optimize_input_specs=runtime._aot_input_specs(jax, np, ((after_collect, batch), {})),
                   options=opts, execution_handles=dict(collect=collect_handle, optimize=optimize_handle),
                   state_bytes=state_bytes, batch_bytes=batch_bytes, collect=collect_profile,
                   optimize=optimize_profile, estimated_peak_bytes=estimated_peak, retained_rollback_states=1)
    history = copy.deepcopy(loaded_history)
    history.append(dict(kind='fresh_campaign_authorization', authorization=copy.deepcopy(config['authorization']),
                        source_identities=copy.deepcopy(source['source_identities']),
                        fresh_run_clock=copy.deepcopy(clock), historical_ledger_unchanged=True))
    history.append(copy.deepcopy(migration))
    journal = ns['_optimizer_resume_journal'](history, candidate_cfg, int(state.updates))
    sink = {}
    s = dict(ns=ns, runtime=runtime, state=state, cfg=copy.deepcopy(candidate_cfg), saved_cfg=copy.deepcopy(candidate_cfg),
             source_cfg=cfg, source_leaf_specs=copy.deepcopy(source906['native_manifest']['leaves']),
             source906_native=_identity(source_native), baseline674_native=_identity(baseline_native),
             candidate_revision=ns['CHECKPOINT_TRAINING_REVISION'], history=history, journal_rows=journal,
             saved_platform=copy.deepcopy(platform), platform=platform, profile=profile, options=opts,
             handles=dict(profile['execution_handles']), saved_handles=dict(profile['execution_handles']),
             runtime_identity=ns['_gpu_runtime_identity'](),
             root=str(root), run_id=config['authorization']['run_id'], authorization=copy.deepcopy(config['authorization']),
             fresh_clock=clock, entry_monotonic=entry['monotonic'], migration=migration,
             historical_cumulative_seconds=source906['native_manifest']['cumulative_seconds'],
             accepted_update=int(state.updates), accepted_env_steps=state.env_steps,
             status='prepared', phase='qa', phase_results=dict(initial_214_array_equality=equality),
             recent_calls=[], exports=[], next_export_elapsed=900., recovery_sink=sink,
             terminal=False, training_enabled=True, last_native=None)
    s['phase_results']['memory_profile'] = dict(collect=collect_profile, optimize=optimize_profile,
        state_bytes=state_bytes, batch_bytes=batch_bytes, estimated_peak_bytes=estimated_peak,
        initial_board_snapshot=snapshot, explicit_AOT_compiles=2)
    s['provenance'] = dict(format='spades-fresh-run-provenance-v1', source_identities=source['source_identities'],
                           source_history_sha256=source['source_history_sha256'],
                           fresh_run_clock=clock, config_migration=migration,
                           source_pins=copy.deepcopy(identities), source_cfg=cfg, candidate_cfg=candidate_cfg,
                           source214_equality=equality, platform=platform,
                           memory_profile=copy.deepcopy(s['phase_results']['memory_profile']),
                           historical_clock=copy.deepcopy(source['historical_clock']),
                           historical_ledger_unchanged=True, original_combined_JIT_continuity='failed; unresolved')
    s['guard'] = guard
    s['admit'] = admit
    s['runtime_callable_identity'] = runtime._aot_eval_namespace_identity(ns)
    sink.update(state=state, cfg=copy.deepcopy(candidate_cfg), history=history, session=s)
    # Frozen model trees stay on host; optimizer/state/RNG are never reset.
    s['baseline674_params'] = runtime._aot_frozen_baseline_params(ns, state, baseline_native, guard)
    s['source906_params'] = runtime._aot_frozen_baseline_params(ns, state, source_native, guard)
    # Keep exact _aot_admit/_aot_launch/_aot_step bytecode. Replace only their
    # campaign boundary/report callbacks, which previously enforced old ledger.
    def boundary(session, label):
        guard(label)
        _memory_options_guard(session['options'])
        runtime._aot_kaggle_identity_guard(session)
        if (runtime._aot_eval_namespace_identity(vars(runtime)) != session['operator_callable_identity']
                or runtime._aot_eval_namespace_identity(ns) != session['runtime_callable_identity']
                or ns['_gpu_runtime_identity']() != session['runtime_identity']
                or session['cfg'] != session['saved_cfg']
                or session['handles'] != session['saved_handles']):
            raise RuntimeError('Reviewed fresh callable/config/AOT identity changed.')
        runtime._aot_style_identity_guard(session)
        if session.get('terminal'):
            raise FreshBoundaryStop('Campaign stopped: ' + label)
        guard(label + ' after style validation')
    original_host_finite = runtime._aot_host_finite
    def fresh_host_finite(session, tree, stage, expected_leaves, payload_guard_bytes):
        if stage == 'optimizer':
            if expected_leaves != 238:
                raise RuntimeError('Unexpected original optimizer leaf count.')
            expected_leaves = 242  # Four reviewed gate diagnostics; state remains214.
        return original_host_finite(session, tree, stage, expected_leaves, payload_guard_bytes)
    runtime._aot_host_finite = fresh_host_finite
    runtime._aot_boundary = boundary
    runtime._aot_write = lambda session: _atomic_status(root, _status(session))
    runtime._aot_elapsed = _fresh_elapsed
    runtime._aot_limit_reason = lambda session: ('one_hour_complete' if _fresh_elapsed(session) >= 3600 else None)
    runtime._aot_begin_cycle_at_launch = lambda session, begin: False
    runtime._aot_event = lambda session, kind, **values: _append_event(root, kind, values)
    runtime._aot_heartbeat = lambda session, event, **values: print(json.dumps(
        dict(event=event, updates=session['accepted_update'], elapsed_seconds=_fresh_elapsed(session), **values),
        allow_nan=False), flush=True)
    s['operator_callable_identity'] = runtime._aot_eval_namespace_identity(vars(runtime))
    guard('ready for QA')
    return s


def _append_event(root, kind, values):
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(kind=kind, utc=datetime.now(timezone.utc).isoformat(), **values), allow_nan=False) + '\n')


def _evaluate(s, comparison):
    evaluator = _load(Path(s['payload_dir']) / 'fresh_evaluation.py', 'spades_fresh_eval_' + uuid.uuid4().hex, expected_sha256=s['provenance']['source_pins']['fresh_evaluation.py']['sha256'])
    s['phase'] = comparison
    s['guard']('begin ' + comparison)
    candidate = comparison == 'candidate_vs_source906'
    if candidate:
        left_params, right_params = s['source906_params'], s['state'].params
        left_cfg, right_cfg = s['source_cfg'], s['saved_cfg']
        left_checkpoint, right_checkpoint = s['source906_native'], s['last_native']
    else:
        left_params, right_params = s['baseline674_params'], s['source906_params']
        left_cfg = right_cfg = s['source_cfg']
        left_checkpoint, right_checkpoint = s['baseline674_native'], s['source906_native']
    report_path = Path(s['root']) / ('evaluation_' + comparison) / 'evaluation_report.json'
    portable_path = report_path.parent / 'portable_evaluation_evidence.zip'
    try:
        result = evaluator.evaluate_fresh_pair(ns=s['ns'], baseline_params=left_params,
            candidate_params=right_params, baseline_cfg=left_cfg, candidate_cfg=right_cfg,
            baseline_checkpoint=left_checkpoint, candidate_checkpoint=right_checkpoint,
            output_dir=report_path.parent,
            precompile_estimate_bytes=s['profile']['estimated_peak_bytes'], admit=s['admit'],
            launch_guard=lambda label: _evaluation_guard(s, label), comparison_name=comparison)
        return result
    finally:
        if report_path.is_file():
            evaluator.package_fresh_evaluation_evidence(report_path, portable_path)
            report = _read_json(report_path)
            s['phase_results'][comparison] = dict(complete=report.get('complete', False),
                report=_identity(report_path), portable_archive=_identity(portable_path),
                playing_strength_claim=False)



def _evaluation_guard(s, label):
    s['guard'](label)
    _memory_options_guard(s['options'])
    s['runtime']._aot_kaggle_identity_guard(s)
    if (s['cfg'] != s['saved_cfg'] or s['ns']['_gpu_runtime_identity']() != s['runtime_identity']
            or s['runtime']._aot_eval_namespace_identity(vars(s['runtime'])) != s['operator_callable_identity']
            or s['runtime']._aot_eval_namespace_identity(s['ns']) != s['runtime_callable_identity']):
        raise RuntimeError('Fresh evaluation config/runtime source changed.')
    s['guard'](label + ' after device/config validation')


def run_once(config_path, payload_dir):
    """One fresh3600s compile/eval/train cap. Never retries or waits for ACK.

    The persistent exclusive start record is created before framework/helper
    imports. A second call, kernel restart, or output-root reuse cannot reset
    this campaign's budget. External verification precedes any new campaign.
    """
    global _RUN_INVOKED, _ACTIVE_RECOVERY
    if _RUN_INVOKED:
        raise RuntimeError('This entrypoint was already invoked; no automatic second run or retry.')
    _RUN_INVOKED = True
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic())
    raw_config = _read_json(config_path)
    root = Path(raw_config['output_dir'])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError('New absolute output directory required.')
    # A second output path cannot renew the same authorized run on this host.
    # Provider runtime loss remains a new explicit parent decision, never retry.
    run_id = raw_config['authorization']['run_id']
    if not isinstance(run_id, str) or not run_id or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-' for c in run_id):
        raise ValueError('Safe authorized run identity required before start journal.')
    registry = (Path('/kaggle/working') if Path('/kaggle/working').is_dir() else root.parent) / '.spades_fresh_campaign_starts'
    registry.mkdir(parents=True, exist_ok=True)
    _write_new_json(registry / (run_id + '.json'), dict(run_id=run_id, start=entry, output_dir=str(root), restart_permitted=False))
    root.mkdir(parents=True, exist_ok=False)
    _write_new_json(root / 'fresh_start_journal.json', dict(format='spades-fresh-start-journal-v1',
        start_utc=entry['utc'], start_monotonic=entry['monotonic'], duration_seconds=3600,
        config_identity=_identity(Path(config_path).absolute()), restart_permitted=False,
        run_id=raw_config['authorization']['run_id'], campaign_id=raw_config['authorization']['campaign_id']))
    s = None
    try:
        config, config_info, payload, identities, contract, source = _host_inputs(
            config_path, payload_dir, started_root=root)
        s = _prepare(config, payload, identities, contract, source, entry, root)
        _ACTIVE_RECOVERY = s['recovery_sink']
        s['payload_dir'] = str(payload)
        _write_new_json(root / 'authorization.json', config['authorization'])
        _write_new_json(root / 'provenance.json', s['provenance'])
        _atomic_status(root, _status(s))
        # One real rollout/update first: existing numerical/legality/work/health
        # checks run unchanged. Commit precedes I/O and accepted rollback stays.
        s['status'] = 'running'
        s['guard']('QA first update')
        health_stop = s['runtime']._aot_step(s)
        s['phase_results']['qa'] = dict(status='PASS' if not health_stop else 'HEALTH_STOP',
            updates=s['accepted_update'], env_steps=s['accepted_env_steps'], actual_metrics=s['last_metrics'])
        _fresh_export(s, 'qa')
        if health_stop:
            raise FreshBoundaryStop(health_stop)
        _evaluate(s, 'source906_vs_baseline674')
        # Reserve the final third for a matched candidate/source906 DEV attempt.
        # No promise that either1024-game benchmark fits the fixed wall cap.
        s['phase'] = 'training'
        while _fresh_elapsed(s) < 2400.0:
            s['guard']('training update')
            health_stop = s['runtime']._aot_step(s)
            if _fresh_elapsed(s) >= s['next_export_elapsed']:
                _fresh_export(s, 'supplemental15')
            if health_stop:
                raise FreshBoundaryStop(health_stop)
        _fresh_export(s, 'pre_evaluation')
        _evaluate(s, 'candidate_vs_source906')
        s['status'] = 'bounded_cycle_complete'
        s['stop_reason'] = 'single_cycle_complete_report_before_extension'
    except BaseException as exc:
        import traceback
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                     retry_attempted=False)
        if s is None:
            _write_new_json(root / 'preparation_failure.json', dict(status='failed', error=error,
                start=entry, source906_preserved=True, provider_action_attempted=False,
                restart_permitted=False))
            return dict(status='preparation_failed', output_dir=str(root), final_bundle_path=None,
                        error=error, restart_permitted=False, playing_strength_claim=False)
        s['error'] = error
        s['stop_reason'] = str(exc)
        s['status'] = 'bounded_stop' if isinstance(exc, FreshBoundaryStop) else 'failed_accepted_state_retained'
    finally:
        if s is not None:
            s['terminal'] = True
            s['training_enabled'] = False
            # No clock gate here: checkpoint the most recent fully accepted
            # update even when the immediately preceding launch crossed its cap.
            try:
                final = _fresh_export(s, 'final', terminal=True)
                _ACTIVE_RECOVERY = s['recovery_sink']
                s['final_bundle_path'] = final['bundle']['path']
            except BaseException as save_error:
                s['error'] = dict(previous=s.get('error'), final_save_type=type(save_error).__name__,
                                  final_save_message=str(save_error), accepted_state_retained=True)
                s['final_bundle_path'] = None
                _atomic_status(root, _status(s))
    result = _status(s)
    result.update(output_dir=str(root), final_bundle_path=s['final_bundle_path'],
                  restart_permitted=False, active_recovery_retained=True)
    _write_new_json(root / 'fresh_run_result.json', result)
    return result
