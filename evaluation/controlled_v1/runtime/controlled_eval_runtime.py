"""Bounded read-only Spades policy comparison. Import has no JAX/provider actions.

Only reviewed engine, opponent, model inference and memory definitions are loaded.
No optimizer, TrainingState, training initializer, trainer, native saver or journal
is constructed. Production execution always uses one selected physical free T4.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import tempfile
import time
import types
import zipfile
from datetime import datetime, timezone

FORMAT = 'spades-controlled-winrate-eval-config-v1'
REPORT_FORMAT = 'spades-controlled-winrate-evaluation-v1'
CHECKPOINTS = {
    'baseline': dict(bytes=43309324, sha256='e125a9a946bbd50185829cde5c4e577d4198d39b3ecede06da9aa44a77e496f6', updates=1329),
    'candidate': dict(bytes=63547896, sha256='15c6ca425eee1298ddd50f01215b6be7011cab1031f0a62ff6cdbc5bd4ae0cde', updates=8351),
}
CANDIDATE_CONFIG_SHA256 = '822c9046158d6161344d3da7d73b115a74a943219d16329633a07cf416720085'
BASE_REVISION = 'strong-teacher-hand-gae-v3+opponent-diversity-v1+advantage-gated-anchor-v1+opponent-training-expansion-v2'
SOURCE_NAMES = ('original_source.ipynb', 'production_source.ipynb', 'diversity_evaluation.py', 'spades_legacy_aot.py')
PROTOCOL_SHA256 = 'da9a8c987c94e299d54b2778599097cc286027b08cd596f1f6a287fb1e32c1b7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    canonical(result)
    return result


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
            size += len(part)
    return dict(path=str(path.resolve()), bytes=size, sha256=h.hexdigest())


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Explicit UTC offset required.')
    return parsed.astimezone(timezone.utc)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_module(path, expected):
    raw = Path(path).read_bytes()
    require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], 'Frozen source changed: ' + str(path))
    module = types.ModuleType('_controlled_eval_' + Path(path).stem)
    module.__file__ = str(Path(path).resolve())
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


def source_pins(payload_dir):
    require(identity(Path(__file__).with_name('source_pins.json'))['sha256'] ==
            '4da260bb4698354f56985c0c05253e7577ae4330baab6f41e72f9e7df0edf854', 'Reviewed source-pin manifest changed.')
    pins = read_json(Path(__file__).with_name('source_pins.json'))
    require(set(pins) == set(SOURCE_NAMES), 'Exact four-source closure required.')
    for name, expected in pins.items():
        actual = identity(Path(payload_dir) / name)
        require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'Source closure mismatch: ' + name)
    return pins


def candidate_selection(config):
    """Future candidates require a separately frozen parent evidence manifest."""
    mode = config.get('assessment_mode', 'current8351')
    require(mode in ('current8351', 'future_final'), 'Unknown assessment mode.')
    if mode == 'current8351':
        require(config.get('candidate_manifest_path') is None and config.get('candidate_manifest_sha256') is None,
                'Current8351 selection cannot be overridden.')
        return CHECKPOINTS.copy(), None
    path = config.get('candidate_manifest_path')
    require(isinstance(path, str) and path, 'Future final candidate manifest is required.')
    require(identity(path)['sha256'] == config.get('candidate_manifest_sha256'), 'Future candidate manifest commitment changed.')
    manifest = read_json(path)
    require(manifest.get('format') == 'spades-controlled-final-candidate-manifest-v1'
            and manifest.get('selection_frozen') is True, 'Explicit frozen future candidate selection required.')
    candidate = manifest.get('candidate', {})
    require(type(candidate.get('bytes')) is int and candidate['bytes'] > 0
            and type(candidate.get('updates')) is int and candidate['updates'] >= 8351
            and isinstance(candidate.get('sha256'), str) and len(candidate['sha256']) == 64
            and all(x in '0123456789abcdef' for x in candidate['sha256'])
            and candidate['sha256'] != CHECKPOINTS['candidate']['sha256']
            and candidate.get('cfg_sha256') == CANDIDATE_CONFIG_SHA256,
            'Future candidate must be an explicit later saved state with unchanged reviewed configuration.')
    audit = manifest.get('cpu_audit', {})
    saved = manifest.get('parent_saved_state', {})
    require(audit.get('status') == 'PASS' and type(audit.get('bytes')) is int and audit['bytes'] > 0
            and isinstance(audit.get('sha256'), str) and len(audit['sha256']) == 64
            and all(x in '0123456789abcdef' for x in audit['sha256'])
            and isinstance(audit.get('reference'), str) and audit['reference'].strip(),
            'Actual referenced and hashed passing CPU audit required for future candidate.')
    require(saved.get('accepted') is True and saved.get('saved_native_sha256') == candidate['sha256']
            and isinstance(saved.get('reference'), str) and saved['reference'].strip(),
            'Parent-accepted externally saved state provenance required.')
    return dict(baseline=CHECKPOINTS['baseline'], candidate={k:candidate[k] for k in ('bytes','sha256','updates')}), manifest


def validate_config(config, protocol):
    require(config.get('format') == FORMAT, 'Unknown execution configuration.')
    require(config.get('protocol_sha256') == PROTOCOL_SHA256, 'Exact reviewed protocol commitment required.')
    require(isinstance(config.get('run_id'), str) and config['run_id'] and len(config['run_id']) <= 128,
            'Explicit run identity required.')
    require(config.get('max_wall_seconds') == 5400 and config.get('export_reserve_seconds') == 600,
            'Exactly 90 minutes total and 10 minutes export reserve required.')
    start, launch, end = [utc(config[k]) for k in ('session_started_utc', 'launch_deadline_utc', 'platform_hard_end_utc')]
    require(0 < (end-start).total_seconds() <= 5400 and (end-launch).total_seconds() == 600 and start < launch,
            'Whole-session wall budget or export reserve differs.')
    require(type(config.get('session_started_monotonic')) in (float, int)
            and math.isfinite(config['session_started_monotonic']) and config['session_started_monotonic'] > 0,
            'First-cell monotonic clock required.')
    require(config.get('host_boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'Session must remain on the first-cell host boot.')
    age = time.monotonic() - config['session_started_monotonic']
    require(age >= 0 and abs(age - (datetime.now(timezone.utc)-start).total_seconds()) < 10,
            'First-cell wall/monotonic clock anchors disagree.')
    evidence = config.get('execution_evidence', {})
    require(evidence.get('provider') == 'kaggle' and all(evidence.get(k) is True for k in
            ('free_compute_only', 'no_active_jobs', 'same_account')), 'Fresh same-account free/no-active-job evidence required.')
    require(type(evidence.get('available_gpu_quota_seconds')) in (int, float)
            and math.isfinite(evidence['available_gpu_quota_seconds']) and evidence['available_gpu_quota_seconds'] >= 5400,
            'At least 90 minutes of currently verified free GPU quota required.')
    observed = utc(evidence['observed_utc'])
    require(0 <= (start-observed).total_seconds() <= 3600 and datetime.now(timezone.utc) < utc(evidence['valid_until_utc'])
            and end <= utc(evidence['provider_hard_end_utc']) and isinstance(evidence.get('reference'), str)
            and evidence['reference'].strip(), 'Fresh referenced quota validity/provider hard end required.')
    require(set(protocol.get('checkpoints', {})) == set(CHECKPOINTS) and all(
            all(protocol['checkpoints'][arm].get(k) == v for k, v in expected.items())
            for arm, expected in CHECKPOINTS.items()), 'Protocol checkpoint selection differs.')
    candidate_selection(config)
    require(protocol.get('budget', {}).get('session_wall_seconds') == 5400
            and protocol['budget'].get('export_reserve_seconds') == 600
            and protocol['budget'].get('workload_deadline_seconds') == 4800, 'Protocol wall policy differs.')
    return dict(start=start, launch=launch, end=end,
                monotonic_launch=config['session_started_monotonic'] + (launch-start).total_seconds())


def model_parameter_layout(width=256, obs_dim=753, critic_dim=1173):
    """Explicit Flax sorted-dict leaf order from the exact reviewed two towers.

    This constructs a schema only, never calls Module.init or an optimizer.
    """
    layout = []
    for tower, inputs, outputs in (('actor_network', obs_dim, 68), ('critic_network', critic_dim, 1)):
        blocks = {**{f'hidden_{i}': (width, width) for i in range(3)},
                  'input': (inputs, width), 'output': (width, outputs),
                  **{f'residual_{i}': (width, width) for i in range(3)}}
        for block in sorted(set(blocks) | {f'norm_{i}' for i in range(3)}):
            if block.startswith('norm_'):
                fields = [('bias', (width,)), ('scale', (width,))]
            else:
                dim_in, dim_out = blocks[block]
                fields = [('bias', (dim_out,)), ('kernel', (dim_in, dim_out))]
            for field, shape in fields:
                layout.append(((tower, block, field), dict(shape=list(shape), dtype='float32')))
    require(len(layout) == 44, 'Policy schema must contain exactly 44 leaves.')
    return layout


def tree_from_leaves(leaves, specs, *, width=256, obs_dim=753, critic_dim=1173):
    layout = model_parameter_layout(width, obs_dim, critic_dim)
    require(len(leaves) == len(specs) == 44, 'Exactly 44 native policy leaves required.')
    tree = {}
    for leaf, actual, (keys, expected) in zip(leaves, specs, layout):
        require(actual == expected and list(leaf.shape) == expected['shape'] and str(leaf.dtype) == expected['dtype'],
                'Native policy architecture differs at ' + '/'.join(keys))
        tree.setdefault(keys[0], {}).setdefault(keys[1], {})[keys[2]] = leaf
    return tree


def inspect_native(path, arm, *, load_params=False, expected=None):
    """Read and validate native bytes on the host; do not deserialize TrainState."""
    import numpy as np
    expected = CHECKPOINTS[arm] if expected is None else expected
    native = Path(path).read_bytes()
    actual = dict(path=str(Path(path).resolve()), bytes=len(native), sha256=sha(native))
    require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'Exact admitted native differs: ' + arm)
    with zipfile.ZipFile(io.BytesIO(native)) as archive:
        require(sorted(archive.namelist()) == ['history.json', 'manifest.json', 'state.npz'] and archive.testzip() is None,
                'Exact native members and valid CRC required.')
        manifest = json.loads(archive.read('manifest.json'))
        history = json.loads(archive.read('history.json'))
        payload = archive.read('state.npz')
    canonical(manifest)
    require(isinstance(history, list) and all(isinstance(x, dict) for x in history), 'Native history must be a record list.')
    revision = BASE_REVISION + ('+formula-opponent-v3' if arm == 'candidate' else '')
    require(manifest.get('training_state_schema') == 2 and manifest.get('training_revision') == revision
            and manifest.get('ruleset') == 'modified-spades-500-blind-swap-team-failure-v1'
            and manifest.get('updates') == expected['updates'], 'Native architecture/revision/counters differ.')
    require(manifest.get('env_steps') == expected['updates'] * 458752, 'Native lifetime decision count differs.')
    require(manifest.get('sha256') == sha(payload), 'Native state payload SHA mismatch.')
    specs = manifest.get('leaves')
    require(isinstance(specs, list) and len(specs) == 214, 'Exactly 214 native state schema leaves required.')
    digest, policy = hashlib.sha256(), []
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        require(sorted(arrays.files) == [f'leaf_{i:05d}' for i in range(214)], 'Exactly 214 unique native arrays required.')
        for i, spec in enumerate(specs):
            leaf = arrays[f'leaf_{i:05d}']
            require(list(leaf.shape) == spec['shape'] and str(leaf.dtype) == spec['dtype'] and not leaf.dtype.hasobject,
                    'Native leaf schema failure: ' + str(i))
            require(not np.issubdtype(leaf.dtype, np.inexact) or bool(np.isfinite(leaf).all()), 'Nonfinite native state leaf: ' + str(i))
            if i < 44:
                digest.update(canonical(spec).encode())
                digest.update(leaf.tobytes(order='C'))
                value = np.array(leaf, copy=True)
                value.setflags(write=False)
                policy.append(value)
        require(arrays['leaf_00207'].shape == () and int(arrays['leaf_00207']) == expected['updates'],
                'Actual native update scalar differs.')
    tree = tree_from_leaves(policy, specs[:44])
    record = dict(native=actual, updates=manifest['updates'], env_steps=manifest['env_steps'],
                  cfg=manifest['cfg'], cfg_sha256=sha(canonical(manifest['cfg']).encode()),
                  training_revision=manifest['training_revision'], ruleset=manifest['ruleset'],
                  native_schema_leaves=214, native_finite_leaves=214, parameter_leaves=44,
                  parameter_digest_sha256=digest.hexdigest(), parameter_bytes_match_native=True,
                  policy_schema_sha256=sha(canonical(specs[:44]).encode()),
                  full_schema_sha256=sha(canonical(specs).encode()), optimizer_loaded_to_device=False,
                  train_state_constructed=False, native_state_modified=False)
    return record, tree if load_params else None


def validate_native_pair(records):
    a, b = records['baseline'], records['candidate']
    require(a['full_schema_sha256'] == b['full_schema_sha256'], 'Full native schemas differ.')
    require(b['cfg_sha256'] == CANDIDATE_CONFIG_SHA256, 'Exact candidate saved full configuration differs.')
    require(a['cfg'].get('opponent_training_profile') == 'training_expansion_v2', 'Original CONTROL1329 profile required.')
    expected = dict(a['cfg'], opponent_training_profile='normal_lr_formula_longrun_v3')
    require(canonical(b['cfg']) == canonical(expected), 'Saved configs may differ only in the documented formula profile.')
    require(all(a['cfg'].get(k) == v for k, v in dict(width=256, n_env=3584, rollout_len=128, lr=.0002).items()),
            'Reviewed shared inference architecture and original settings required.')


def validate_plans(protocol, frozen, assessment_mode='current8351'):
    stages = protocol.get('stages')
    require(isinstance(stages, list) and len(stages) == 2
            and [s.get('stage') for s in stages] == ['development', 'holdout_primary'], 'Exactly separate DEV then primary holdout required.')
    require(stages[0]['plan'] == frozen.evaluation_plan('development', 'development'), 'Frozen development plan changed.')
    expected = frozen.evaluation_plan('holdout', 'primary')
    plan = stages[1]['plan']
    require(set(plan) == set(expected) and all(plan[k] == v for k, v in expected.items() if k not in ('seed', 'bootstrap_seed')),
            'Holdout may change only precommitted fresh root/bootstrap seeds.')
    for key in ('seed', 'bootstrap_seed'):
        require(type(plan[key]) is int and 0 <= plan[key] < 2**32, 'Explicit uint32 fresh holdout seed required.')
    require(plan['seed'] not in (2026101901, 2026101907, 2026101921) and plan['bootstrap_seed'] not in (2026101902, 2026101908, 2026101922),
            'Fresh primary seeds must not reuse previous/reserved seeds.')
    require(plan['seed'] != plan['bootstrap_seed'], 'Independent deal/bootstrap seeds required.')
    if assessment_mode == 'future_final':
        future = protocol['seed_adapter']['future_confirmation']
        require(future.get('execute_in_this_session') is False and future.get('requires_new_predeclared_final_checkpoint_binding') is True,
                'Original reserved future confirmation commitment required.')
        final_plan = dict(plan, seed_name='confirmation', seed=future['deal_seed'], bootstrap_seed=future['bootstrap_seed'])
        require(final_plan['seed'] == 748298380 and final_plan['bootstrap_seed'] == 1923345019,
                'Exact reserved final confirmation seeds required.')
        return [stages[0], dict(stage='holdout_confirmation', plan=final_plan)]
    require(assessment_mode == 'current8351', 'Unknown plan assessment mode.')
    return stages


def preflight(config_path, payload_dir):
    """Host-only exact identity/schema/protocol checks; never imports JAX."""
    require(not any(n == 'jax' or n.startswith(('jax.', 'jaxlib', 'jax_plugins.')) for n in sys.modules),
            'Fresh process required; no JAX import before physical UUID selection.')
    config = read_json(config_path)
    protocol_path = Path(config['protocol_path'])
    require(identity(protocol_path)['sha256'] == config['protocol_sha256'], 'Protocol commitment changed.')
    protocol = read_json(protocol_path)
    clocks = validate_config(config, protocol)
    require(datetime.now(timezone.utc) < clocks['launch'], 'No start after workload deadline.')
    pins = source_pins(payload_dir)
    frozen = load_module(Path(payload_dir)/'diversity_evaluation.py', pins['diversity_evaluation.py'])
    validate_plans(protocol, frozen, config.get('assessment_mode', 'current8351'))
    selection, manifest = candidate_selection(config)
    records = {arm: inspect_native(config[arm+'_native_path'], arm, expected=selection[arm])[0] for arm in CHECKPOINTS}
    validate_native_pair(records)
    return dict(status='PASS', scope='Host content/recorded quota only; not GPU execution or provider attestation.',
                checkpoints=records, source_pins=pins, protocol_sha256=config['protocol_sha256'],
                checkpoint_selection=selection, future_candidate_manifest=manifest,
                config_sha256=identity(config_path)['sha256'], training=False, gpu_allocated=False)


def definitions_namespace(payload_dir, *, platform=None, cpu_fixture=False, memory_guard=None):
    """Extract only exact reviewed rules/opponents/inference/memory definitions."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    pins = source_pins(payload_dir)
    original = read_json(Path(payload_dir)/'original_source.ipynb')
    production = read_json(Path(payload_dir)/'production_source.ipynb')
    module_name = '_read_only_spades_inference_' + str(time.monotonic_ns())
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns = module.__dict__
    ns.update(np=np, jax=jax, jnp=jnp, nn=nn)
    if platform is not None:
        ns['_SPADES_KAGGLE_PLATFORM'] = platform
    def execute(source, filename, names=None, exclude_expressions=False):
        tree = ast.parse(source, filename)
        if names is not None:
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
            require({node.name for node in tree.body} == set(names), 'Exact inference definition selection differs.')
        elif exclude_expressions:
            tree.body = [node for node in tree.body if not isinstance(node, ast.Expr)]
        exec(compile(ast.fix_missing_locations(tree), filename, 'exec'), ns)
    if not cpu_fixture:
        execute(''.join(production['cells'][12]['source']), 'frozen_memory_definitions.py', exclude_expressions=True)
        require(callable(memory_guard), 'Production definition allocations need actual memory admission.')
        memory_guard(ns)
    execute(''.join(original['cells'][6]['source']), 'frozen_engine_definitions.py', exclude_expressions=True)
    execute(''.join(production['cells'][8]['source']), 'frozen_opponent_definitions.py', exclude_expressions=True)
    execute(''.join(production['cells'][10]['source']), 'frozen_model_inference.py',
            names=('_network_features', 'ResidualTower', 'ActorCritic', 'model_for_config', 'policy_logits'))
    require((ns['OBS_DIM'], ns['CRITIC_DIM'], ns['N_ACTIONS']) == (753, 1173, 68), 'Reviewed rule observation/action dimensions differ.')
    require(not any(k in ns for k in ('TrainingState', 'initialize_training', 'make_train_step', '_optimizer', 'load_checkpoint', 'save_checkpoint')),
            'Training/native mutation entrypoints must be absent.')
    ns['_frozen_evaluator'] = load_module(Path(payload_dir)/'diversity_evaluation.py', pins['diversity_evaluation.py'])
    return ns


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def workload_guard(config, clocks, *, signatures=None, platform=None, jax=None):
    """Admission guard; the supervisor separately interrupts in-flight work."""
    require(datetime.now(timezone.utc) < clocks['launch'] and time.monotonic() < clocks['monotonic_launch'],
            'Workload deadline reached; no new compile, transfer or kernel.')
    require(config['host_boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Host boot changed.')
    for path, expected in (signatures or {}).items():
        require(_stat(path) == expected, 'Bound input changed during evaluation: ' + path)
    if platform is not None:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == platform['selected_gpu_uuid'], 'Selected physical UUID visibility changed.')
    if jax is not None:
        devices = jax.devices()
        require(len(devices) == 1 and devices[0].platform == 'gpu' and 'T4' in str(devices[0].device_kind)
                and jax.default_backend() == 'gpu', 'Exactly one selected physical T4 required; no CPU fallback.')


def _admit(ns, estimate, label, guard, report, root):
    guard()
    options = dict(ns['GPU_TUNING'])
    require(all(options.get(k) == v for k, v in dict(target_gib=14., reserve_mib=1024,
            safety_factor=1.45, extra_guard_mib=256).items()), 'Frozen memory options changed.')
    record = ns['_gpu_memory_admission'](int(estimate), options, label)
    details, snapshot = record.get('budget_details', {}), record.get('snapshot', {})
    require(record.get('status') == 'admitted' and record.get('overlap_credit_bytes') == 0
            and snapshot.get('platform') == 'gpu' and details.get('pool_counter_source') == 'pool_bytes'
            and details.get('pool_counter_status') == 'validated_current_total'
            and 'pool_bytes' in snapshot.get('jax_memory_stats_raw', {}), 'Validated current pool/physical-board admission required.')
    report.setdefault('memory_admissions', []).append(record)
    atomic_json(root/'evaluation_report.json', report)
    guard()
    return record


def verify_device_parameters(ns, parameters, records):
    """Digest the actual device arrays used by both compiled-policy calls."""
    result = {}
    for arm, tree in parameters.items():
        leaves = ns['jax'].tree_util.tree_leaves(ns['jax'].device_get(tree))
        layout = model_parameter_layout()
        require(len(leaves) == 44, 'Device policy must contain all 44 native leaves.')
        digest = hashlib.sha256()
        for value, (_, spec) in zip(leaves, layout):
            value = ns['np'].asarray(value)
            require(list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype']
                    and bool(ns['np'].isfinite(value).all()), 'Actual device parameter schema/nonfinite failure.')
            digest.update(canonical(spec).encode())
            digest.update(value.tobytes(order='C'))
        require(digest.hexdigest() == records[arm]['parameter_digest_sha256'], 'Actual device parameters changed from exact native.')
        result[arm] = dict(parameter_leaves=44, parameter_digest_sha256=digest.hexdigest(),
                           all_device_parameter_bytes_match_native=True)
    return result


def _threefry_pair(key, counts):
    """Pure integer reference for JAX's Threefry2x32-20 round function."""
    mask = 2**32-1
    keys = [key[0], key[1], key[0]^key[1]^0x1BD11BDA]
    x, y = (counts[0]+keys[0])&mask, (counts[1]+keys[1])&mask
    rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
    for group in range(5):
        for rotation in rotations[group%2]:
            x = (x+y)&mask
            y = (((y<<rotation)&mask)|(y>>(32-rotation)))^x
        x = (x+keys[(group+1)%3])&mask
        y = (y+keys[(group+2)%3]+group+1)&mask
    return [x, y]


def reference_pair_roots(seed, style, pairs, partitionable):
    """Independent host recipe for fold_in(PRNGKey(seed),style), then split."""
    require(type(seed) is int and 0 <= seed < 2**32 and type(style) is int and 0 <= style < 2**32
            and type(pairs) is int and pairs > 0 and type(partitionable) is bool, 'Explicit Threefry root recipe arguments required.')
    folded = _threefry_pair([0, seed], [0, style])
    if partitionable:
        return [_threefry_pair(folded, [0, i]) for i in range(pairs)]
    hashes = [_threefry_pair(folded, [i, pairs+i]) for i in range(pairs)]
    flat = [h[0] for h in hashes]+[h[1] for h in hashes]
    return [flat[i:i+2] for i in range(0, len(flat), 2)]


def _worker(config_path, payload_dir):
    config = read_json(config_path)
    root = Path(config['output_dir'])
    report = read_json(root/'evaluation_report.json')
    try:
        audit = preflight(config_path, payload_dir)
        report['host_preflight'] = audit
        protocol = read_json(config['protocol_path'])
        clocks = validate_config(config, protocol)
        watched = [str(Path(config_path).resolve()), config['protocol_path'],
                   config['baseline_native_path'], config['candidate_native_path'], __file__,
                   str(Path(__file__).with_name('source_pins.json'))]
        watched += [str(Path(payload_dir)/name) for name in SOURCE_NAMES]
        if config.get('candidate_manifest_path'):
            watched.append(config['candidate_manifest_path'])
        signatures = {path: _stat(path) for path in watched}
        guard = lambda: workload_guard(config, clocks, signatures=signatures)
        guard()
        runtime = load_module(Path(payload_dir)/'spades_legacy_aot.py', audit['source_pins']['spades_legacy_aot.py'])
        inventory = runtime.inventory_kaggle_nojax()
        requested = config.get('selected_gpu_uuid')
        boards = sorted((b for b in inventory['physical_gpu_inventory'] if 'T4' in b['name']), key=lambda b:b['pci_bus_id'])
        require(boards, 'No actual free-provider physical T4 available; no fallback.')
        selected = requested if requested is not None else boards[0]['uuid']
        host = runtime.preflight_kaggle_nojax(selected)
        platform = host['platform']
        os.environ['CUDA_VISIBLE_DEVICES'] = selected
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['JAX_PLATFORMS'] = 'cuda'
        require(os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR') in (None, 'default', 'bfc'), 'BFC pool telemetry allocator required.')
        report.update(status='framework_setup', platform=platform, host_inventory=host,
                      physical_uuid_selection='Explicit UUID' if requested else 'First T4 by physical PCI order')
        atomic_json(root/'evaluation_report.json', report)
        guard()
        import importlib.metadata
        import numpy as np
        import jax
        import flax
        report['library_versions'] = dict(python='.'.join(map(str, sys.version_info[:3])),
                numpy=np.__version__, jax=jax.__version__,
                jaxlib=importlib.metadata.version('jaxlib'), flax=flax.__version__)
        require(jax.__version__ == '0.11.1' and importlib.metadata.version('jaxlib') == '0.11.1'
                and flax.__version__ == '0.12.10' and not jax.config.jax_enable_x64
                and not jax.config.jax_disable_jit and jax.config.jax_default_matmul_precision is None,
                'Exact reviewed production inference stack and precision required.')
        require(jax.config.jax_default_prng_impl == 'threefry2x32', 'Reviewed Threefry PRNG implementation required.')
        report['prng'] = dict(implementation='threefry2x32', partitionable=bool(jax.config.jax_threefry_partitionable))
        guard = lambda: workload_guard(config, clocks, signatures=signatures, platform=platform, jax=jax)
        guard()
        ns = definitions_namespace(payload_dir, platform=platform,
                memory_guard=lambda mem:_admit(mem, 2**20, 'immutable rules constants', guard, report, root))
        frozen = ns['_frozen_evaluator']
        stages = validate_plans(protocol, frozen, config.get('assessment_mode', 'current8351'))
        records, parameters = {}, {}
        for arm in CHECKPOINTS:
            guard()
            record, tree = inspect_native(config[arm+'_native_path'], arm, load_params=True,
                                          expected=audit['checkpoint_selection'][arm])
            require(record == audit['checkpoints'][arm], 'Native changed after preflight.')
            leaves = jax.tree_util.tree_leaves(tree)
            estimate = sum(x.nbytes for x in leaves)
            _admit(ns, estimate, arm+' policy-only transfer', guard, report, root)
            device_tree = jax.device_put(tree)
            for expected, actual in zip(leaves, jax.tree_util.tree_leaves(jax.device_get(device_tree))):
                require(expected.shape == actual.shape and expected.dtype == actual.dtype
                        and expected.tobytes(order='C') == actual.tobytes(order='C'), 'Device policy bytes differ from exact native.')
            parameters[arm], records[arm] = device_tree, record
        validate_native_pair(records)
        report['parameter_native_bindings'] = records
        report['device_parameter_checks_before'] = verify_device_parameters(ns, parameters, records)
        kernel = frozen.build_diversity_evaluation_kernel(ns, 256)
        first = stages[0]['plan']
        jnp = ns['jnp']
        roots = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(first['seed']), first['style_ids'][0]), first['pairs_per_style'])
        keys = jnp.repeat(roots, 2, axis=0)
        teams = jnp.tile(jnp.asarray([0, 1], jnp.int32), first['pairs_per_style'])
        # A fixed 512 MiB provisional inference compile allowance, then actual
        # executable buffer accounting for every launch. No training profile.
        _admit(ns, 512*2**20, 'inference lowering', guard, report, root)
        guard()
        abstract = jax.tree_util.tree_map(lambda x:jax.ShapeDtypeStruct(x.shape, x.dtype), parameters['baseline'])
        lowered = kernel.lower(abstract, abstract, keys, teams, jnp.int32(first['style_ids'][0]), jnp.int32(first['max_hands']))
        report['status'] = 'compiling'
        atomic_json(root/'evaluation_report.json', report)
        _admit(ns, 512*2**20, 'inference compilation', guard, report, root)
        guard()
        handle = lowered.compile()
        del lowered
        profile = ns['_gpu_compiled_bytes'](handle.memory_analysis())
        estimate = profile['total_bytes']
        require(type(estimate) is int and estimate > 0, 'Actual compiled memory profile required.')
        report.update(status='evaluating', compiler_profile=profile, deterministic_actions=True,
                      no_training=True, policy_team_controls_both_partners=True)
        atomic_json(root/'evaluation_report.json', report)
        campaign_roots = set()
        for stage in stages:
            name, plan = stage['stage'], stage['plan']
            stage_report = dict(stage=name, plan=plan, status='incomplete', scored=False,
                                raw_files=[], transport_files=[], checkpoint_bindings=records)
            report['stages'].append(stage_report)
            raw = {'baseline':{}, 'candidate':{}}
            seen = set()
            atomic_json(root/'evaluation_report.json', report)
            for style in plan['style_ids']:
                guard()
                roots = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(plan['seed']), style), plan['pairs_per_style'])
                keys = jnp.repeat(roots, 2, axis=0)
                root_values = np.asarray(jax.device_get(roots))
                require(root_values.tolist() == reference_pair_roots(plan['seed'], style, plan['pairs_per_style'],
                        report['prng']['partitionable']), 'Actual JAX roots differ from independent host Threefry recipe.')
                tuples = {tuple(map(int, r)) for r in root_values}
                require(len(tuples) == plan['pairs_per_style'] and not tuples & seen and not tuples & campaign_roots, 'Duplicate root clusters within/across stages.')
                seen.update(tuples)
                campaign_roots.update(tuples)
                for arm in ('baseline', 'candidate'):
                    label = f'{name}_{style}_{arm}'
                    _admit(ns, estimate, label, guard, report, root)
                    guard()
                    kernel_started = time.monotonic()
                    outputs = jax.device_get(handle(parameters[arm], parameters[arm], keys, teams,
                                                  jnp.int32(style), jnp.int32(plan['max_hands'])))
                    kernel_wall_seconds = time.monotonic()-kernel_started
                    # Always preserve numeric outputs before reliability validation.
                    transport = frozen._preserve_kernel_outputs(root/('transport_'+label+'.npz'), outputs)
                    stage_report['transport_files'].append(transport)
                    atomic_json(root/'evaluation_report.json', report)
                    row = frozen.validate_output(outputs, np.asarray(jax.device_get(teams)), max_hands=plan['max_hands'])
                    row.update(stage=name, mode=plan['mode'], seed_name=plan['seed_name'], seed=plan['seed'],
                               style_id=style, arm=arm, pair_root_keys=root_values.tolist(),
                               checkpoint=records[arm]['native'], transport_snapshot=transport,
                               kernel_wall_seconds=kernel_wall_seconds, preserved_utc=now())
                    raw_path = root/('raw_'+label+'.json')
                    atomic_json(raw_path, row)
                    stage_report['raw_files'].append(identity(raw_path))
                    raw[arm][style] = row
                    atomic_json(root/'evaluation_report.json', report)
                    require(row['complete'] and row['legal'] and row['finite'],
                            'Incomplete/illegal/nonfinite full match: preserve evidence, no retry or partial inference.')
                    guard()
            stage_report.update(status='complete', scored=True,
                                analysis=frozen.analyze_paired_raw(raw, plan),
                                win_draw_loss=win_draw_loss(raw))
            stage_report['raw_manifest_sha256'] = sha(canonical(stage_report['raw_files']).encode())
            stage_report['transport_manifest_sha256'] = sha(canonical(stage_report['transport_files']).encode())
            atomic_json(root/'evaluation_report.json', report)
        report['device_parameter_checks_after'] = verify_device_parameters(ns, parameters, records)
        for arm in CHECKPOINTS:
            require(identity(config[arm+'_native_path']) == records[arm]['native'], 'Native input changed during comparison.')
        report.update(status='complete', complete=True, finished_utc=now(),
                      claim_scope='Fixed checkpoint comparison against these scripted opponents and deal roots only; no general population or training-seed uncertainty claim.')
        atomic_json(root/'evaluation_report.json', report)
        return 0
    except BaseException as exc:
        report.update(status='incomplete_unscored', complete=False, finished_utc=now(),
                      error=dict(type=type(exc).__name__, message=str(exc)))
        atomic_json(root/'evaluation_report.json', report)
        return 1


def package_evidence(root, config, *, payload_dir, final=False):
    """Small atomic evidence snapshot; exact native inputs remain external.

    The native identity/digest bindings are retained. Independent rebinding uses
    the already durably saved exact native files, without duplicating ~107 MB.
    """
    root = Path(root)
    target = root/('final_evidence.zip' if final else 'latest_partial_evidence.zip')
    temporary = target.with_name(target.name+'.tmp')
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.npz', '.csv')]
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            archive.write(path, path.name)
        archive.write(__file__, 'runtime/controlled_eval_runtime.py')
        archive.write(Path(__file__).with_name('source_pins.json'), 'runtime/source_pins.json')
        for name in SOURCE_NAMES:
            archive.write(Path(payload_dir)/name, 'frozen_sources/'+name)
        manifest = {info.filename:dict(bytes=info.file_size, sha256=sha(archive.read(info.filename)))
                    for info in archive.infolist()}
        archive.writestr('archive_manifest.json', json.dumps(dict(format='spades-controlled-evidence-manifest-v1',
            attempt_snapshot='final' if final else 'incomplete_forensic_snapshot', members=manifest,
            external_native_inputs={arm:config[arm+'_native_path'] for arm in CHECKPOINTS},
            native_inputs_included=False), indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        require(archive.testzip() is None, 'Portable evaluation evidence CRC failure.')
    os.replace(temporary, target)
    return identity(target)


def supervise_process(process, *, deadline_monotonic, on_poll=None, poll_seconds=.25):
    """Terminate the worker process group at deadline; no automatic retry.

    Injectable deadline/process interface supports an isolated CPU watchdog test.
    """
    outcome = {'watchdog_terminated':False}
    done = threading.Event()
    def deadline_watch():
        if done.wait(max(0., deadline_monotonic-time.monotonic())):
            return
        if process.poll() is not None:
            return
        outcome['watchdog_terminated'] = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        except ProcessLookupError:
            pass
    watcher = threading.Thread(target=deadline_watch, name='evaluation-hard-deadline', daemon=True)
    watcher.start()
    while process.poll() is None:
        if on_poll is not None:
            on_poll()
        time.sleep(min(poll_seconds, max(0., deadline_monotonic-time.monotonic())))
    done.set()
    watcher.join(timeout=7)
    return dict(exit_code=process.returncode, watchdog_terminated=outcome['watchdog_terminated'])


def run_once(config_path, payload_dir):
    """One exclusive run, supervised worker, partial evidence and final bundle.

    Start the session clock in the notebook's first cell, before dependency setup.
    This function never allocates a provider, installs a package or retries work.
    """
    config_path, payload_dir = Path(config_path).resolve(), Path(payload_dir).resolve()
    config = read_json(config_path)
    protocol = read_json(config['protocol_path'])
    clocks = validate_config(config, protocol)
    require(datetime.now(timezone.utc) < clocks['launch'], 'No run starts after workload deadline.')
    root = Path(config['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root/'execution_config.json', config)
    (root/'evaluation_plan.json').write_bytes(Path(config['protocol_path']).read_bytes())
    selection, candidate_manifest = candidate_selection(config)
    if candidate_manifest is not None:
        (root/'candidate_manifest.json').write_bytes(Path(config['candidate_manifest_path']).read_bytes())
    report = dict(format=REPORT_FORMAT, run_id=config['run_id'], status='preparing', complete=False,
                  created_utc=now(), stages=[], protocol_sha256=config['protocol_sha256'],
                  checkpoint_selection=selection, assessment_mode=config.get('assessment_mode', 'current8351'),
                  workload_deadline_utc=config['launch_deadline_utc'],
                  platform_hard_end_utc=config['platform_hard_end_utc'], export_reserve_seconds=600,
                  optimizer_loaded_to_device=False, train_state_constructed=False, training=False,
                  automatic_retry=False, automatic_paid_compute=False, production_gpu_execution_verified=False)
    if config.get('assessment_mode') == 'future_final':
        report['future_confirmation_exposure_note'] = protocol['seed_adapter']['future_confirmation']['warning']
    atomic_json(root/'evaluation_report.json', report)
    process = None
    try:
        # Native audit runs inside the supervised process: even CPU decompression
        # is included in the first-cell budget and can be stopped by the parent.
        with (root/'worker.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                    '--config', str(config_path), '--payload-dir', str(payload_dir)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            last_package = [0.]
            def snapshot():
                # Atomically checkpoint every completed raw chunk before a later
                # operation can be terminated. Export is done by the supervisor.
                if time.monotonic()-last_package[0] >= 120 and time.monotonic()+15 < clocks['monotonic_launch']:
                    package_evidence(root, config, payload_dir=payload_dir)
                    last_package[0] = time.monotonic()
            outcome = supervise_process(process, deadline_monotonic=clocks['monotonic_launch'], on_poll=snapshot)
        report = read_json(root/'evaluation_report.json')
        report['supervisor'] = outcome
        if outcome['watchdog_terminated'] or outcome['exit_code'] != 0:
            report.update(status='incomplete_unscored', complete=False)
            if outcome['watchdog_terminated']:
                report['error'] = dict(type='WorkloadDeadline', message='Worker stopped at the 80-minute workload boundary; preserved completed raw files, no retry.')
        if report.get('complete'):
            report['production_gpu_execution_verified'] = True
        report['finished_utc'] = now()
        atomic_json(root/'evaluation_report.json', report)
        if report.get('complete'):
            audit = audit_report(root/'evaluation_report.json', payload_dir, protocol_path=config['protocol_path'])
            atomic_json(root/'content_audit.json', audit)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        report = read_json(root/'evaluation_report.json')
        report.update(status='incomplete_unscored', complete=False, finished_utc=now(),
                      supervisor_error=dict(type=type(exc).__name__, message=str(exc)))
        atomic_json(root/'evaluation_report.json', report)
    export_outcomes_csv(root, report)
    manifest = dict(status=report['status'], complete=report['complete'],
                    report=identity(root/'evaluation_report.json'), outcomes_csv=identity(root/'outcomes.csv'),
                    content_audit=identity(root/'content_audit.json') if (root/'content_audit.json').exists() else None,
                    exact_native_inputs_external=True, native_selection=selection,
                    training=False, assessment_mode=config.get('assessment_mode', 'current8351'))
    atomic_json(root/'run_manifest.json', manifest)
    bundle = package_evidence(root, config, payload_dir=payload_dir, final=True)
    result = dict(status=report['status'], complete=report['complete'],
                  report=manifest['report'], content_audit=manifest['content_audit'], outcomes_csv=manifest['outcomes_csv'], evidence=bundle,
                  training=False, final_checkpoint_testing_reserved=True)
    atomic_json(root/'run_result.json', result)
    return result


def win_draw_loss(raw):
    result = {}
    for arm in ('baseline', 'candidate'):
        outcomes = [x for row in raw[arm].values() for x in row['outcomes']]
        require(all(x in (0., .5, 1.) for x in outcomes), 'Only complete win/draw/loss outcomes may be summarized.')
        result[arm] = dict(wins=outcomes.count(1.), draws=outcomes.count(.5), losses=outcomes.count(0.),
                           games=len(outcomes), win_credit=(outcomes.count(1.)+.5*outcomes.count(.5))/len(outcomes))
    return result


def export_outcomes_csv(root, report):
    root = Path(root)
    headers = ['stage', 'style_id', 'pair_index', 'root_key_0', 'root_key_1', 'team', 'arm',
               'completed', 'winner', 'team0_score', 'team1_score', 'win_credit', 'hands', 'illegal_actions', 'checkpoint_sha256']
    temporary = root/'outcomes.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for stage in report.get('stages', []):
            for info in stage.get('raw_files', []):
                row = read_json(info['path'])
                for i, team in enumerate(row['learner_teams']):
                    key = row['pair_root_keys'][i//2]
                    writer.writerow(dict(stage=row['stage'], style_id=row['style_id'], pair_index=i//2,
                        root_key_0=key[0], root_key_1=key[1], team=team, arm=row['arm'], completed=row['completed'][i],
                        winner=row['winners'][i], team0_score=row['scores'][i][0], team1_score=row['scores'][i][1],
                        win_credit=row['outcomes'][i], hands=row['hands'][i], illegal_actions=row['illegal_actions'][i],
                        checkpoint_sha256=row['checkpoint']['sha256']))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root/'outcomes.csv')


def audit_report(report_path, payload_dir, *, protocol_path=None, baseline_native=None, candidate_native=None):
    """Host-only independent raw scoring/statistics and native content audit.

    Exported raw identities preserve their original paths. A separate recorded
    path map resolves relocated evidence without rewriting those identities.
    This checks content; it is not an independent GPU/provider attestation.
    """
    import numpy as np
    report_path = Path(report_path)
    root = report_path.parent
    report = read_json(report_path)
    require(report.get('format') == REPORT_FORMAT and report.get('complete') is True
            and report.get('status') == 'complete', 'Complete campaign required; incomplete evidence is unscored.')
    versions = report.get('library_versions', {})
    require(isinstance(versions, dict) and all(isinstance(versions.get(name), str) and versions[name].strip()
            for name in ('python', 'numpy', 'jax', 'jaxlib', 'flax')),
            'Actual Python/NumPy/JAX/JAXlib/Flax versions required for inference and bootstrap reproducibility.')
    config = read_json(root/'execution_config.json')
    mode = config.get('assessment_mode', 'current8351')
    require(report.get('assessment_mode') == mode, 'Report/config assessment modes differ.')
    protocol_path = Path(protocol_path) if protocol_path is not None else root/'evaluation_plan.json'
    require(identity(protocol_path)['sha256'] == report.get('protocol_sha256') == config.get('protocol_sha256') == PROTOCOL_SHA256,
            'Exact protocol commitment differs.')
    protocol = read_json(protocol_path)
    pins = source_pins(payload_dir)
    frozen = load_module(Path(payload_dir)/'diversity_evaluation.py', pins['diversity_evaluation.py'])
    expected_stages = validate_plans(protocol, frozen, mode)
    if mode == 'future_final' and not Path(config['candidate_manifest_path']).exists():
        config = dict(config, candidate_manifest_path=str(root/'candidate_manifest.json'))
    selection, manifest = candidate_selection(config)
    require(report['checkpoint_selection'] == selection, 'Report selection differs from admitted candidate manifest.')
    inputs = dict(baseline=baseline_native or config['baseline_native_path'],
                  candidate=candidate_native or config['candidate_native_path'])
    actual_records = {arm:inspect_native(path, arm, expected=selection[arm])[0] for arm, path in inputs.items()}
    validate_native_pair(actual_records)
    bindings = report.get('parameter_native_bindings', {})
    require(set(bindings) == set(CHECKPOINTS), 'Exact paired native binding coverage required.')
    path_map = {}
    for arm, actual in actual_records.items():
        saved = bindings[arm]
        require(set(actual) == set(saved) and all(canonical(actual[k]) == canonical(v) for k, v in saved.items() if k != 'native')
                and all(actual['native'][k] == saved['native'][k] for k in ('bytes', 'sha256')),
                'Recomputed native parameter/fullcfg binding differs: ' + arm)
        path_map[saved['native']['path']] = actual['native']['path']
        for phase in ('before', 'after'):
            proof = report.get('device_parameter_checks_'+phase, {}).get(arm, {})
            require(proof == dict(parameter_leaves=44, parameter_digest_sha256=actual['parameter_digest_sha256'],
                                  all_device_parameter_bytes_match_native=True), 'Recorded actual device parameter digest differs.')
    require(len(report.get('stages', [])) == len(expected_stages), 'Exact planned stage coverage required.')
    all_roots = set()
    raw_paths = set()
    transport_paths = set()
    total_games = 0
    def resolve(info):
        require(isinstance(info, dict) and set(info) == {'path', 'bytes', 'sha256'}, 'Exact artifact identity required.')
        original = Path(info['path'])
        path = original if original.exists() else root/original.name
        actual = identity(path)
        require(all(actual[k] == info[k] for k in ('bytes', 'sha256')), 'Preserved artifact bytes changed: ' + str(path))
        path_map[info['path']] = actual['path']
        return path
    summaries = []
    for expected, actual_stage in zip(expected_stages, report['stages']):
        stage, plan = expected['stage'], expected['plan']
        require(actual_stage.get('stage') == stage and actual_stage.get('plan') == plan
                and actual_stage.get('status') == 'complete' and actual_stage.get('scored') is True,
                'Complete exact stage plan required.')
        files, transports = actual_stage.get('raw_files'), actual_stage.get('transport_files')
        require(isinstance(files, list) and isinstance(transports, list)
                and len(files) == len(transports) == 2*len(plan['style_ids']), 'Exact raw/transport stage coverage required.')
        require(actual_stage.get('raw_manifest_sha256') == sha(canonical(files).encode())
                and actual_stage.get('transport_manifest_sha256') == sha(canonical(transports).encode()), 'Raw/transport identity manifests changed.')
        raw = {'baseline':{}, 'candidate':{}}
        for info in files:
            require(info['path'] not in raw_paths, 'Duplicate raw artifact path.')
            raw_paths.add(info['path'])
            row = read_json(resolve(info))
            arm, style = row.get('arm'), row.get('style_id')
            require(arm in raw and type(style) is int and style in plan['style_ids'] and style not in raw[arm],
                    'Duplicate/unknown raw arm/style.')
            require(row.get('stage') == stage and row.get('mode') == plan['mode']
                    and row.get('seed_name') == plan['seed_name'] and row.get('seed') == plan['seed']
                    and row.get('checkpoint') == bindings[arm]['native'], 'Raw plan/checkpoint binding differs.')
            teams = row.get('learner_teams')
            require(teams == [0, 1]*plan['pairs_per_style'] and all(type(x) is int for x in teams), 'Exact duplicated team roles required.')
            transport = row.get('transport_snapshot')
            require(transport in transports and transport['path'] not in transport_paths, 'Exact unique referenced kernel transport required.')
            transport_paths.add(transport['path'])
            with np.load(resolve(transport), allow_pickle=False) as arrays:
                require(sorted(arrays.files) == [f'output_{i:02d}' for i in range(7)], 'Exactly seven preserved kernel outputs required.')
                scored = frozen.validate_output(tuple(arrays[f'output_{i:02d}'] for i in range(7)),
                                                np.asarray(teams, np.int32), max_hands=plan['max_hands'])
            require(all(canonical(row.get(k)) == canonical(v) for k, v in scored.items())
                    and all(row.get(k) is True for k in ('complete', 'legal', 'finite')), 'Raw scores/reliability differ from actual preserved arrays.')
            roots = row.get('pair_root_keys')
            require(isinstance(roots, list) and len(roots) == plan['pairs_per_style'] and all(
                    isinstance(key, list) and len(key) == 2 and all(type(x) is int and 0 <= x < 2**32 for x in key)
                    for key in roots), 'Exact uint32 duplicate-pair roots required.')
            prng = report.get('prng', {})
            require(prng.get('implementation') == 'threefry2x32' and type(prng.get('partitionable')) is bool
                    and roots == reference_pair_roots(plan['seed'], style, plan['pairs_per_style'], prng['partitionable']),
                    'Raw roots differ from independent host JAX Threefry fold-in/split recipe.')
            raw[arm][style] = row
        for style in plan['style_ids']:
            roots = [tuple(x) for x in raw['baseline'][style]['pair_root_keys']]
            require(len(set(roots)) == plan['pairs_per_style'] and not set(roots)&all_roots,
                    'Root collision within/across strata or stages.')
            require(raw['baseline'][style]['pair_root_keys'] == raw['candidate'][style]['pair_root_keys'], 'Policies must share every exact deal root.')
            all_roots.update(roots)
        analysis = frozen.analyze_paired_raw(raw, plan)
        require(canonical(analysis) == canonical(actual_stage.get('analysis'))
                and canonical(win_draw_loss(raw)) == canonical(actual_stage.get('win_draw_loss')),
                'Independent recomputed paired statistics/WDL differ.')
        games = 2*len(plan['style_ids'])*plan['games_per_style']
        total_games += games
        summaries.append(dict(stage=stage, games=games, root_clusters=analysis['root_clusters'],
                              aggregate_difference=analysis['aggregate_difference']))
    require(report.get('optimizer_loaded_to_device') is False and report.get('train_state_constructed') is False
            and report.get('training') is False and report.get('automatic_retry') is False, 'Read-only/no-retry record differs.')
    return dict(status='PASS', scope='CONTENT_ONLY', report=identity(report_path), stages=summaries,
                games=total_games, root_clusters=len(all_roots), resolved_artifact_paths=path_map,
                root_generation_independently_recomputed=True,
                root_audit_scope='Pure host Threefry2x32-20 recomputes the recorded JAX fold-in/split recipe and verifies unique paired roots across stages.',
                gpu_execution_independently_attested=False, provider_state_independently_attested=False,
                exact_external_native_inputs_required=True, source_closure_included_in_bundle=True)


def audit_bundle(bundle_path, *, baseline_native, candidate_native):
    """Audit a relocated small archive against separately preserved natives."""
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None,
                'Unique archive members with valid CRC required.')
        require(all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names), 'Unsafe archive member path.')
        manifest = json.loads(archive.read('archive_manifest.json'))
        require(manifest.get('format') == 'spades-controlled-evidence-manifest-v1'
                and manifest.get('native_inputs_included') is False
                and set(manifest.get('members', {})) == set(names)-{'archive_manifest.json'}, 'Exact small archive manifest coverage required.')
        for name, expected in manifest['members'].items():
            raw = archive.read(name)
            require(dict(bytes=len(raw), sha256=sha(raw)) == expected, 'Archive member hash mismatch: ' + name)
        with tempfile.TemporaryDirectory(prefix='spades_readonly_evidence_audit_') as directory:
            root = Path(directory)
            for name in names:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            audit = audit_report(root/'evaluation_report.json', root/'frozen_sources',
                                 baseline_native=baseline_native, candidate_native=candidate_native)
            # Temporary resolved paths are forensic mappings, not persistent output.
            audit['resolved_path_lifetime'] = 'Temporary extracted archive; original recorded paths remain unmodified.'
            audit['bundle'] = identity(bundle_path)
            return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run', '_worker', 'audit', 'audit-bundle'))
    parser.add_argument('--config')
    parser.add_argument('--payload-dir')
    parser.add_argument('--bundle')
    parser.add_argument('--report')
    parser.add_argument('--protocol')
    parser.add_argument('--baseline-native')
    parser.add_argument('--candidate-native')
    args = parser.parse_args()
    if args.action == 'audit-bundle':
        require(all((args.bundle, args.baseline_native, args.candidate_native)),
                '--bundle and both exact external native paths are required.')
        result = audit_bundle(args.bundle, baseline_native=args.baseline_native, candidate_native=args.candidate_native)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    require(args.payload_dir is not None, '--payload-dir is required.')
    if args.action == '_worker':
        return _worker(args.config, args.payload_dir)
    if args.action == 'audit':
        result = audit_report(args.report, args.payload_dir, protocol_path=args.protocol,
                              baseline_native=args.baseline_native, candidate_native=args.candidate_native)
    else:
        require(args.config is not None, '--config is required.')
        result = (preflight if args.action == 'preflight' else run_once)(args.config, args.payload_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('status') in ('PASS', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
