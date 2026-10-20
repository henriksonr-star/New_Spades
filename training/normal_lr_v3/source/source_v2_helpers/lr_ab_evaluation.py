"""Expanded-training learning-rate A/B binding over the frozen DEV evaluator.

Definitions only. No provider, JAX, training, ACK or network action at import.
The original evaluator's code object, kernels, seeds, rules, scoring and launch
checks are reused. Only its parameter/native binder and evidence writer differ.
"""
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import types
import zipfile

FROZEN_SHA256 = '1751557e800d48a596b7a783595aa1bd42ecf6c20b130c97a9a498e583e16ecc'
FORMAT = 'spades-lr-ab-expansion-per-arm-evaluation-v2'
ARCHIVE_FORMAT = 'spades-lr-ab-expansion-per-arm-evidence-v2'
SOURCE_REVISION = 'strong-teacher-hand-gae-v3+opponent-diversity-v1+advantage-gated-anchor-v1'
REVISION = SOURCE_REVISION + '+opponent-training-expansion-v2'
TRAINING_PROFILE = 'training_expansion_v2'
TRAINING_PROFILE_KEY = 'opponent_training_profile'
EXPANSION_MANIFEST_SHA256 = '2497c7fd54ee54ed26a3131ece87addbf4c2032170c7b1a7434528e6ea2781d4'
TRAIN_STYLE_IDS = tuple(range(13)) + tuple(13 + 64 * f + v for f in (0, 1, 2, 5, 6, 7) for v in range(32))
FROZEN_TRAIN_STYLE_IDS = tuple(range(13)) + tuple(13 + 64 * f + v for f in range(3) for v in range(32))
HELDOUT_STYLE_IDS = tuple(13 + 64 * f + v for f in (3, 4) for v in range(32, 64))
FROZEN_WRAPPER_SHA256 = '0e9b0ab0b956ade7ea08634cbb1c931110e69a07960ec84a0b223d518af31ae3'
BENCHMARK_SCOPE = dict(training_profile=TRAINING_PROFILE,
    frozen_development_style_count=16, new_expansion_families_in_strength_benchmark=False,
    additional_evaluation_games=0, heldout_protocol_unchanged=True,
    opponent_namespace='Frozen evaluator reconstructs the original five-family opponent namespace before compiling.')
COMPARISON = 'control_vs_lr150'
SOURCE_NATIVE = dict(bytes=42929088, sha256='93504f4d9dfb7080ff08b6eacf544aa55971997c9a45a1d06e8d111307fbf68e')
SOURCE_UPDATES = 1201
SOURCE_ENV_STEPS = 550961152
TARGET_ROLLOUTS = 128
TARGET_UPDATES = SOURCE_UPDATES + TARGET_ROLLOUTS
BRANCH_LABELS = dict(baseline='control', candidate='candidate', source='source')
NATIVE_MEMBERS = dict(baseline='control_native.zip', candidate='candidate_native.zip', source='source_native.zip')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=_sha(raw))


def _same_content(actual, recorded):
    return actual['bytes'] == recorded.get('bytes') and actual['sha256'] == recorded.get('sha256')


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key.')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token.')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _canonical(value)
    return value


def load_frozen(source_path=None):
    path = Path(source_path) if source_path is not None else Path(__file__).with_name('diversity_evaluation.py')
    raw = path.read_bytes()
    _require(_sha(raw) == FROZEN_SHA256, 'Frozen evaluation module bytes differ.')
    # Compile the exact verified snapshot, never a second path read or stale pyc.
    module = types.ModuleType('_lr_ab_exact_frozen_evaluation')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def _validate_configs(control_cfg, lr_cfg):
    _require(isinstance(control_cfg, dict) and isinstance(lr_cfg, dict), 'Each actual native config is required.')
    _require(control_cfg.get(TRAINING_PROFILE_KEY) == TRAINING_PROFILE, 'Actual continued arm training-expansion profile required.')
    _require(type(control_cfg.get('lr')) in (int, float) and control_cfg['lr'] == .0002, 'Actual control lr must be .0002.')
    expected = dict(control_cfg, lr=.0003)
    _require(_canonical(lr_cfg) == _canonical(expected), 'Full actual configs may differ only by lr=.0002 to .0003.')
    _require((control_cfg.get('width'), control_cfg.get('n_env'), control_cfg.get('rollout_len')) == (256, 3584, 128), 'Exact saved model and rollout configuration required.')
    return dict(baseline=dict(control_cfg), candidate=dict(lr_cfg))


def _validate_source_config(source_cfg, control_cfg):
    _require(isinstance(source_cfg, dict) and TRAINING_PROFILE_KEY not in source_cfg, 'Source1201 must retain its original unexpanded cfg.')
    expected = dict(control_cfg)
    _require(expected.pop(TRAINING_PROFILE_KEY, None) == TRAINING_PROFILE, 'Continued control expansion profile required.')
    _require(_canonical(source_cfg) == _canonical(expected), 'Source1201 cfg may differ from continued control only by the new training profile.')
    return dict(source_cfg)


def _expected_revision(arm):
    _require(arm in BRANCH_LABELS, 'Unknown native checkpoint arm.')
    return SOURCE_REVISION if arm == 'source' else REVISION


def frozen_opponent_record():
    return dict(protocol='spades-opponent-diversity-v1',
                manifest_sha256='20e4727fdf4d37d035c9ec1bc5d770eb3510f93c9bd20c680c512ceee5f78f1d',
                heldout_protocol_sha256='0b5970eb566bfcf3db55ffc5559a80e27c072babffe5e1c64f92d3aa5ed2942d',
                training_style_ids=list(FROZEN_TRAIN_STYLE_IDS), heldout_style_ids=list(HELDOUT_STYLE_IDS),
                native_league_style_leaf=156, legacy_style_ids_unchanged=True,
                holdout_used_for_training=False, strength_evaluated=False)


def expansion_opponent_record():
    return dict(protocol='spades-training-opponent-expansion-v2', profile=TRAINING_PROFILE,
                manifest_sha256=EXPANSION_MANIFEST_SHA256, training_style_ids=list(TRAIN_STYLE_IDS),
                heldout_style_ids=list(HELDOUT_STYLE_IDS), native_league_style_leaf=156,
                old_families_unchanged=True, holdout_used_for_training=False, strength_evaluated=False)


def training_profile_metadata(cfg):
    """Derive scripted proportions from actual saved outer assignment settings."""
    _require(isinstance(cfg, dict) and all(type(cfg.get(k)) in (int, float) and math.isfinite(cfg[k]) and 0 <= cfg[k] <= 1 for k in ('selfplay_prob', 'snapshot_prob')), 'Actual saved outer assignment probabilities required; no defaults.')
    selfplay, snapshot = cfg['selfplay_prob'], cfg['snapshot_prob']
    _require(selfplay + snapshot <= 1, 'Actual saved outer assignment probabilities must sum to at most one.')
    scripted = 1. - selfplay - snapshot
    return dict(format='spades-opponent-training-profile-v2', profile=TRAINING_PROFILE,
                opponent_record=expansion_opponent_record(),
                scripted_sampling=dict(legacy_probability=.25, old_train_probability=.25,
                    new_probability=.5, legacy_uniform_count=13, old_uniform_family_count=3,
                    new_uniform_family_count=3, uniform_variants_per_family=32, assignment_scope='match'),
                saved_outer_configuration=dict(selfplay_prob=selfplay, snapshot_prob=snapshot),
                effective_assignment_probabilities=dict(selfplay=selfplay, historical_snapshot=snapshot,
                    scripted=scripted, legacy_scripted=scripted*.25,
                    old_family_scripted=scripted*.25, new_family_scripted=scripted*.5,
                    each_legacy_style=scripted/52, each_old_family=scripted/12,
                    each_old_family_variant=scripted/384, each_new_family=scripted/6,
                    each_new_family_variant=scripted/192),
                historical_snapshot_unavailable_fallback='selfplay',
                assignment_scope='match; not decision or time fractions',
                frozen_benchmark_unchanged=True, playing_strength_claim=False)


def _validate_native_training_metadata(manifest, cfg, arm):
    if arm == 'source':
        _require(TRAINING_PROFILE_KEY not in cfg and _canonical(manifest.get('opponent_diversity')) == _canonical(frozen_opponent_record()), 'Source1201 native must retain its original opponent metadata.')
        _require(TRAINING_PROFILE_KEY not in manifest and 'frozen_opponent_benchmark' not in manifest, 'Source1201 native metadata cannot be relabeled as expanded.')
        return None
    _require(cfg.get(TRAINING_PROFILE_KEY) == TRAINING_PROFILE, 'Expanded native own profile cfg required.')
    expected = training_profile_metadata(cfg)
    _require(_canonical(manifest.get('opponent_diversity')) == _canonical(expansion_opponent_record()) and _canonical(manifest.get('frozen_opponent_benchmark')) == _canonical(frozen_opponent_record()) and _canonical(manifest.get(TRAINING_PROFILE_KEY)) == _canonical(expected), 'Truthful expanded native opponents/profile and separate frozen benchmark required.')
    return _sha(_canonical(expected).encode())


def _experiment_binding(experiment_id, run_id, source_native_identity, arm_config_signatures, configs):
    for name, value in (('experiment_id', experiment_id), ('run_id', run_id)):
        _require(isinstance(value, str) and 0 < len(value) <= 128 and all(char.isalnum() and char.isascii() or char in '_.-' for char in value), 'Safe explicit ' + name + ' required.')
    _require(isinstance(source_native_identity, dict) and _same_content(source_native_identity, SOURCE_NATIVE), 'Exact source1201 native identity required.')
    signatures = {BRANCH_LABELS[arm]: _sha(_canonical(cfg).encode()) for arm, cfg in configs.items() if arm != 'source'}
    _require(_canonical(arm_config_signatures) == _canonical(signatures), 'Caller-authorized actual arm configuration signatures differ.')
    return dict(experiment_id=experiment_id, run_id=run_id, source_native=dict(SOURCE_NATIVE),
                source_updates=SOURCE_UPDATES, source_env_steps=SOURCE_ENV_STEPS,
                arm_config_signatures=signatures)


def _native_info(path, recorded, cfg, arm):
    """Read real checkpoint bytes without rewriting their cfg or identities."""
    import numpy as np
    raw_native = path.getvalue() if isinstance(path, io.BytesIO) else Path(path).read_bytes()
    actual = dict(bytes=len(raw_native), sha256=_sha(raw_native))
    _require(_same_content(actual, recorded), 'Native checkpoint content identity differs: ' + arm)
    # Hash and parse one immutable snapshot; never reopen a native path.
    with zipfile.ZipFile(io.BytesIO(raw_native)) as archive:
        _require(len(archive.namelist()) == 3 and set(archive.namelist()) == {'manifest.json', 'history.json', 'state.npz'} and archive.testzip() is None, 'Exact native checkpoint container/CRC required.')
        manifest = _json(archive.read('manifest.json'))
        payload = archive.read('state.npz')
    _require(manifest.get('training_revision') == _expected_revision(arm) and manifest.get('training_state_schema') == 2 and isinstance(manifest.get('leaves'), list) and len(manifest['leaves']) == 214, 'Native revision/schema differs: ' + arm)
    _require(_canonical(manifest.get('cfg')) == _canonical(cfg), 'Native own full config differs: ' + arm)
    profile_signature = _validate_native_training_metadata(manifest, cfg, arm)
    _require(manifest.get('sha256') == _sha(payload), 'Native array payload hash differs.')
    _require(type(manifest.get('updates')) is int and type(manifest.get('env_steps')) is int, 'Native exact counters required.')
    expected_updates = SOURCE_UPDATES if arm == 'source' else TARGET_UPDATES
    expected_steps = SOURCE_ENV_STEPS + (expected_updates - SOURCE_UPDATES) * 3584 * 128
    _require(manifest['updates'] == expected_updates and manifest['env_steps'] == expected_steps, 'Incomplete A/B: exact source1201 or trained1329/128-rollout endpoint required.')
    if arm == 'source':
        _require(_same_content(actual, SOURCE_NATIVE), 'Exact source1201 native bytes differ.')
    digest = hashlib.sha256()
    params = []
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        _require(set(arrays.files) == {f'leaf_{i:05d}' for i in range(214)}, 'Exact 214 native leaves required.')
        for index in range(44):
            leaf = arrays[f'leaf_{index:05d}']
            spec = manifest['leaves'][index]
            _require(list(leaf.shape) == spec.get('shape') and str(leaf.dtype) == spec.get('dtype') and not leaf.dtype.hasobject and bool(np.isfinite(leaf).all()), 'Native policy leaf specification/finite failure.')
            digest.update(_canonical(spec).encode())
            digest.update(leaf.tobytes(order='C'))
            params.append(leaf.copy())
    return manifest, params, dict(native=dict(recorded), parameter_leaves=44, parameter_digest_sha256=digest.hexdigest(), parameter_bytes_match_native=True, native_config_signature=_sha(_canonical(cfg).encode()), native_training_revision=manifest['training_revision'], native_opponent_record_sha256=_sha(_canonical(manifest['opponent_diversity']).encode()), native_training_profile_sha256=profile_signature, updates=manifest['updates'], env_steps=manifest['env_steps'], completed_rollouts=manifest['updates'] - SOURCE_UPDATES)


def _group_analysis(frozen, raw, plan):
    """Reuse frozen paired bootstrap unchanged within the two declared groups."""
    groups = {}
    for name, predicate in (('legacy_styles', lambda style: style < 13), ('original_train_styles', lambda style: style >= 13)):
        subset = dict(plan, style_ids=[style for style in plan['style_ids'] if predicate(style)])
        grouped_raw = {arm: {style: raw[arm][style] for style in subset['style_ids']} for arm in raw}
        groups[name] = dict(style_ids=subset['style_ids'], scope='Frozen development only: four legacy styles or twelve variants from the original three TRAIN families; excludes heldout and expansion families.', analysis=frozen.analyze_paired_raw(grouped_raw, subset))
    return groups


def evaluate_lr_ab_pair(*, ns, control_params, lr_params, source_params,
                        control_cfg, lr_cfg, source_cfg, control_checkpoint,
                        lr_checkpoint, source_checkpoint, output_dir,
                        precompile_estimate_bytes, admit, launch_guard,
                        experiment_id, run_id, source_native_identity,
                        arm_config_signatures):
    """One fixed1536-game three-policy DEV campaign; never a second campaign.

    The primary control/candidate1024 games execute the exact frozen function
    code object. A transparent lower/compile proxy retains its actual handle;
    source1201 then plays512 games on that SAME handle/keys without recompiling
    or replaying control/candidate. A timeout preserves partial evidence and
    makes the entire three-policy campaign incomplete, with no selection.
    """
    import numpy as np
    configs = _validate_configs(control_cfg, lr_cfg)
    _validate_source_config(source_cfg, control_cfg)
    experiment = _experiment_binding(experiment_id, run_id, source_native_identity, arm_config_signatures, configs)
    frozen = load_frozen(ns.get('_lr_ab_frozen_evaluation_path', ns.get('_fresh_frozen_evaluation_path')))
    records = dict(baseline=dict(control_checkpoint), candidate=dict(lr_checkpoint), source=dict(source_checkpoint))
    configs['source'] = dict(source_cfg)
    signatures = {arm: _sha(_canonical(cfg).encode()) for arm, cfg in configs.items()}
    source_identity = _identity(__file__)
    bindings, native_specs, captured = {}, {}, {}
    raw = {'baseline': {}, 'candidate': {}, 'source': {}}
    root = Path(output_dir)

    def bind_arm(actual_ns, params, arm):
        _require(tuple(actual_ns['TrainingState']._fields[:1]) == ('params',), 'TrainingState.params must be first native subtree.')
        record = records[arm]
        manifest, native_params, result = _native_info(record['path'], record, configs[arm], arm)
        leaves = actual_ns['jax'].tree_util.tree_leaves(params)
        _require(len(leaves) == 44, 'Exact44 policy parameter leaves required.')
        for index, (leaf, native) in enumerate(zip(leaves, native_params)):
            actual = np.asarray(actual_ns['jax'].device_get(leaf))
            _require(actual.shape == native.shape and actual.dtype == native.dtype and actual.tobytes(order='C') == native.tobytes(order='C'), 'Actual policy/native parameter bytes differ: ' + arm + ':' + str(index))
        bindings[arm], native_specs[arm] = result, manifest['leaves']
        _require(all(_canonical(value) == _canonical(manifest['leaves']) for value in native_specs.values()), 'Architecture/state leaf specifications differ between policies.')
        return result

    def bind(actual_ns, params, record, ignored_cfg, *, baseline):
        arm = 'baseline' if baseline else 'candidate'
        _require(_canonical(record) == _canonical(records[arm]), 'Primary native role differs.')
        return bind_arm(actual_ns, params, arm)

    original_write = frozen._write_json
    def metadata(value):
        value = dict(value)
        value.pop('cfg', None)
        value.pop('config_signature', None)
        value.update(format=FORMAT, comparison_name=COMPARISON,
                     experiment_binding=experiment, branch_labels=BRANCH_LABELS,
                     per_arm_configs=configs, per_arm_config_signatures=signatures,
                     checkpoint_records=records, parameter_native_bindings=dict(bindings),
                     per_arm_training_revisions={arm: _expected_revision(arm) for arm in records},
                     benchmark_scope=dict(BENCHMARK_SCOPE),
                     adapted_frozen_wrapper_sha256=FROZEN_WRAPPER_SHA256,
                     target_rollouts_per_trained_arm=TARGET_ROLLOUTS,
                     target_updates_per_trained_arm=TARGET_UPDATES,
                     config_difference=dict(lr=dict(control=.0002, candidate=.0003, multiplier=1.5)),
                     config_difference_scope='Between the two continued arms; both use the same expanded training profile.',
                     source_to_trained_config_difference={TRAINING_PROFILE_KEY: dict(source=None, trained=TRAINING_PROFILE)},
                     shared_continued_arm_training_profile=training_profile_metadata(control_cfg),
                     wrapper_source=source_identity, wrapper_source_sha256=source_identity['sha256'],
                     heldout_used=False, training_authorized=False, ack_authorized=False,
                     frozen_evaluator_code_reused=True, primary_frozen_code_object=True,
                     source_reuses_same_compiled_handle=True, training_seed_count=1,
                     training_seed_uncertainty_included=False, lr_generalization_claim=False,
                     playing_strength_claim=False, selection_authorized=False,
                     inference_scope='One conditional training-seed comparison. Equal128 rollout updates include any treatment-induced KL early-stop/minibatch work differences; actual work is bound by the outer runtime journal.')
        return value

    def write(path, value):
        value = dict(value)
        if value.get('format') == 'spades-diversity-evaluation-report-v1':
            return original_write(root / 'primary_pair_report.json', metadata(value))
        if value.get('arm') in ('baseline', 'candidate'):
            arm = value['arm']
            value.update(comparison_name=COMPARISON, branch_label=BRANCH_LABELS[arm],
                         experiment_id=experiment_id, run_id=run_id,
                         native_config_signature=signatures[arm])
            raw[arm][value['style_id']] = value
        return original_write(path, value)

    # This proxy forwards unchanged operands and returns the actual compiled
    # handle to the frozen function. It captures, never wraps, kernel execution.
    original_factory = frozen.build_diversity_evaluation_kernel
    def factory(actual_ns, width):
        kernel = original_factory(actual_ns, width)
        class KernelProxy:
            def lower(self, *args, **kwargs):
                lowered = kernel.lower(*args, **kwargs)
                class LoweredProxy:
                    def compile(self):
                        handle = lowered.compile()
                        captured['handle'] = handle
                        return handle
                return LoweredProxy()
        return KernelProxy()

    launch_guard('source1201 parameter/native binding')
    bind_arm(ns, source_params, 'source')
    namespace = dict(frozen.evaluate_diversity_pair.__globals__,
                     _bind_params_to_native=bind, _write_json=write,
                     build_diversity_evaluation_kernel=factory)
    function = types.FunctionType(frozen.evaluate_diversity_pair.__code__, namespace,
                                  'evaluate_lr_ab_primary_frozen_code', frozen.evaluate_diversity_pair.__defaults__)
    function.__kwdefaults__ = dict(frozen.evaluate_diversity_pair.__kwdefaults__ or {})
    report = None
    try:
        function(ns=ns, baseline_params=control_params, candidate_params=lr_params,
                 cfg=dict(lr_cfg), mode='development', seed_name='development',
                 output_dir=root,
                 checkpoint_records={arm: records[arm] for arm in ('baseline', 'candidate')},
                 precompile_estimate_bytes=precompile_estimate_bytes, admit=admit,
                 launch_guard=launch_guard, cycle_number=1)
        report = _json((root / 'primary_pair_report.json').read_bytes())
        report.update(status='collecting_source', complete=False, legal=False, finite=False,
                      primary_analysis=report.pop('analysis'), primary_aggregate_difference=report.pop('aggregate_difference'),
                      primary_complete=True)
        jax, jnp = ns['jax'], ns['jnp']
        plan = report['plan']
        handle = captured['handle']
        estimate = report['compiler_profile']['total_bytes']
        for style in plan['style_ids']:
            roots = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(plan['seed']), style), plan['pairs_per_style'])
            keys = jnp.repeat(roots, 2, axis=0)
            teams = jnp.tile(jnp.asarray([0, 1], jnp.int32), plan['pairs_per_style'])
            root_values = np.asarray(jax.device_get(roots))
            _require(root_values.tolist() == raw['baseline'][style]['pair_root_keys'], 'Source/primary root generation differs.')
            label = f'evaluation development development {style} source'
            launch_guard(label + ' before admission')
            admitted = admit(estimate, label)
            _require(isinstance(admitted, dict) and admitted.get('status') == 'admitted' and admitted.get('overlap_credit_bytes') == 0 and admitted.get('snapshot', {}).get('platform') == 'gpu' and admitted.get('budget_details', {}).get('pool_counter_source') == 'pool_bytes' and admitted.get('budget_details', {}).get('pool_counter_status') == 'validated_current_total', 'Actual validated GPU pool/board admission required.')
            launch_guard(label + ' after admission')
            launch_guard(label + ' kernel')
            outputs = jax.device_get(handle(source_params, source_params, keys, teams, jnp.int32(style), jnp.int32(plan['max_hands'])))
            transport = frozen._preserve_kernel_outputs(root / f'transport_development_development_{style}_source.npz', outputs)
            report['transport_files'].append(transport)
            row = frozen.validate_output(outputs, np.asarray(jax.device_get(teams)), max_hands=plan['max_hands'])
            row.update(mode='development', seed_name='development', seed=plan['seed'],
                       style_id=style, arm='source', pair_root_keys=root_values.tolist(),
                       checkpoint=records['source'], transport_snapshot=transport,
                       comparison_name=COMPARISON, branch_label='source', experiment_id=experiment_id,
                       run_id=run_id, native_config_signature=signatures['source'])
            report['raw_files'].append(original_write(root / f'raw_development_development_{style}_source.json', row))
            raw['source'][style] = row
            launch_guard(label + ' after fence and raw preservation')
            _require(row['complete'] and row['legal'] and row['finite'], 'Source evaluation invalid/incomplete; raw retained, no inference/retry.')
        analyses = _all_analyses(frozen, raw, plan)
        report.update(status='complete', complete=True, legal=True, finite=True,
                      analysis=analyses['primary'], aggregate_difference=analyses['primary']['aggregate_difference'],
                      comparisons=analyses, games=1536, unique_policy_games=1536,
                      development_groups=analyses['primary_development_groups'])
    except BaseException as exc:
        if report is None and (root / 'primary_pair_report.json').is_file():
            report = _json((root / 'primary_pair_report.json').read_bytes())
        if report is not None:
            report.update(status='incomplete', complete=False, legal=False, finite=False,
                          error=dict(type=type(exc).__name__, message=str(exc)),
                          primary_complete=bool(report.get('primary_complete', False)),
                          selection_authorized=False, playing_strength_claim=False)
        raise
    finally:
        if report is not None:
            report = metadata(report)
            report['raw_manifest_sha256'] = _sha(_canonical(report['raw_files']).encode())
            report['transport_manifest_sha256'] = _sha(_canonical(report['transport_files']).encode())
            original_write(root / 'evaluation_report.json', report)
    return report


def _all_analyses(frozen, raw, plan):
    def pair(left, right):
        return dict(baseline=raw[left], candidate=raw[right])
    primary = pair('baseline', 'candidate')
    control_source = pair('source', 'baseline')
    candidate_source = pair('source', 'candidate')
    return dict(primary=frozen.analyze_paired_raw(primary, plan),
                control_vs_source=frozen.analyze_paired_raw(control_source, plan),
                candidate_vs_source=frozen.analyze_paired_raw(candidate_source, plan),
                primary_development_groups=_group_analysis(frozen, primary, plan),
                control_vs_source_development_groups=_group_analysis(frozen, control_source, plan),
                candidate_vs_source_development_groups=_group_analysis(frozen, candidate_source, plan))


def _archive_reader(path):
    """Return unchanged report, logical artifact reader and embedded native bytes."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        report = _json(path.read_bytes())
        def read(info):
            raw = Path(info['path']).read_bytes()
            _require(len(raw) == info['bytes'] and _sha(raw) == info['sha256'], 'Live evaluation artifact hash differs.')
            return raw
        return report, read, None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)) and archive.testzip() is None, 'Evaluation archive duplicate/CRC failure.')
        manifest = _json(archive.read('evaluation_archive_manifest.json'))
        _require(manifest.get('format') == ARCHIVE_FORMAT and isinstance(manifest.get('members'), dict), 'LR A/B archive format differs.')
        _require(set(names) == set(manifest['members']) | {'evaluation_archive_manifest.json'}, 'Exact archive member coverage required.')
        content = {}
        for name, info in manifest['members'].items():
            _require(isinstance(name, str) and '/' not in name and '\\' not in name and name not in ('.', '..'), 'Unsafe archive member name.')
            raw = archive.read(name)
            _require(set(info) == {'bytes', 'sha256'} and len(raw) == info['bytes'] and _sha(raw) == info['sha256'], 'Evaluation archive content hash differs.')
            content[name] = raw
    report = _json(content['evaluation_report.json'])
    expected_members = {'evaluation_report.json', *NATIVE_MEMBERS.values()} | {f'raw_{i:03d}.json' for i in range(len(report.get('raw_files', [])))} | {f'transport_{i:03d}.npz' for i in range(len(report.get('transport_files', [])))}
    _require(set(content) == expected_members, 'Portable raw/transport/native exact member coverage differs.')
    by_path = {}
    for key, prefix, suffix in (('raw_files', 'raw', 'json'), ('transport_files', 'transport', 'npz')):
        for index, info in enumerate(report[key]):
            _require(info['path'] not in by_path, 'Duplicate logical artifact path.')
            by_path[info['path']] = content[f'{prefix}_{index:03d}.{suffix}']
    def read(info):
        raw = by_path[info['path']]
        _require(len(raw) == info['bytes'] and _sha(raw) == info['sha256'], 'Portable artifact differs from original recorded identity.')
        return raw
    for info in report.get('raw_files', []) + report.get('transport_files', []):
        read(info)
    natives = {arm: content[member] for arm, member in NATIVE_MEMBERS.items()}
    for arm, raw in natives.items():
        recorded = report['checkpoint_records'][arm]
        _require(len(raw) == recorded['bytes'] and _sha(raw) == recorded['sha256'], 'Exact evaluated native bytes are not enclosed: ' + arm)
    return report, read, natives


def audit_lr_ab_evaluation(report_path, *, control_checkpoint=None, lr_checkpoint=None, source_checkpoint=None,
                           expected_experiment_id=None, expected_run_id=None,
                           expected_arm_config_signatures=None, frozen_source_path=None):
    """Host-only audit; native files may be relocated, provenance stays unchanged.

    For live reports, three native paths are explicit trusted operator inputs. Only their exact
    bytes/hash bind to the recorded identities. No runtime absolute raw/native
    path is required when auditing the portable evidence archive.
    """
    import numpy as np
    frozen = load_frozen(frozen_source_path)
    report, read, enclosed_natives = _archive_reader(report_path)
    comparison = report.get('comparison_name')
    _require(comparison == COMPARISON and report.get('branch_labels') == BRANCH_LABELS, 'Explicit control/candidate role binding differs.')
    configs = report.get('per_arm_configs')
    _require(isinstance(configs, dict) and set(configs) == {'baseline', 'candidate', 'source'}, 'Exact per-arm config coverage required.')
    _validate_configs(configs['baseline'], configs['candidate'])
    _validate_source_config(configs['source'], configs['baseline'])
    signatures = {arm: _sha(_canonical(cfg).encode()) for arm, cfg in configs.items()}
    _require(report.get('per_arm_config_signatures') == signatures and 'cfg' not in report and 'config_signature' not in report, 'Per-arm config signature differs or ambiguous shared config exists.')
    _require(report.get('per_arm_training_revisions') == {arm: _expected_revision(arm) for arm in configs}, 'Per-arm source/expanded training revisions differ.')
    _require(report.get('benchmark_scope') == BENCHMARK_SCOPE and report.get('adapted_frozen_wrapper_sha256') == FROZEN_WRAPPER_SHA256, 'Frozen benchmark scope or wrapper lineage differs.')
    _require(report.get('config_difference') == dict(lr=dict(control=.0002, candidate=.0003, multiplier=1.5)) and report.get('config_difference_scope') == 'Between the two continued arms; both use the same expanded training profile.', 'Declared between-arm LR-only config difference differs.')
    _require(report.get('source_to_trained_config_difference') == {TRAINING_PROFILE_KEY: dict(source=None, trained=TRAINING_PROFILE)} and _canonical(report.get('shared_continued_arm_training_profile')) == _canonical(training_profile_metadata(configs['baseline'])), 'Actual shared continued-arm profile or source-to-trained config difference differs.')
    experiment = report.get('experiment_binding')
    _require(isinstance(experiment, dict), 'Explicit experiment binding required.')
    rebuilt = _experiment_binding(experiment.get('experiment_id'), experiment.get('run_id'), experiment.get('source_native'), experiment.get('arm_config_signatures'), configs)
    _require(_canonical(experiment) == _canonical(rebuilt), 'Experiment source/cfg/counter binding differs.')
    if expected_experiment_id is not None:
        _require(experiment['experiment_id'] == expected_experiment_id, 'Unexpected experiment identity.')
    if expected_run_id is not None:
        _require(experiment['run_id'] == expected_run_id, 'Unexpected run identity.')
    if expected_arm_config_signatures is not None:
        _require(_canonical(experiment['arm_config_signatures']) == _canonical(expected_arm_config_signatures), 'Unexpected authorized arm cfg signatures.')
    _require(type(report.get('training_seed_count')) is int and report['training_seed_count'] == 1 and report.get('training_seed_uncertainty_included') is False and report.get('lr_generalization_claim') is False and report.get('playing_strength_claim') is False and report.get('selection_authorized') is False, 'One-seed LR inference scope differs.')
    _require(report.get('format') == FORMAT and report.get('mode') == 'development' and report.get('status') == 'complete' and all(report.get(key) is True for key in ('complete', 'legal', 'finite')), 'Completed LR A/B development report required; partial evidence makes no inference.')
    _require(report.get('heldout_used') is False and report.get('training_authorized') is False and report.get('ack_authorized') is False, 'Evaluation cannot authorize training/ACK/heldout.')
    _require(report.get('wrapper_source_sha256') == _identity(__file__)['sha256'] and report.get('wrapper_source', {}).get('sha256') == report['wrapper_source_sha256'], 'Fresh wrapper source binding differs.')
    _require(report.get('evaluation_source_sha256') == FROZEN_SHA256 and report.get('frozen_opponent_source_sha256') == frozen.FROZEN_OPPONENT_SOURCE_SHA256 and report.get('opponent_manifest_sha256') == frozen.OPPONENT_MANIFEST_SHA256 and report.get('heldout_protocol_sha256') == frozen.HELDOUT_PROTOCOL_SHA256, 'Frozen evaluator/source/protocol binding differs.')
    plan = frozen.evaluation_plan('development', 'development')
    _require(_canonical(report.get('plan')) == _canonical(plan) and report.get('seed_name') == 'development', 'Exact frozen development plan required.')
    records = report.get('checkpoint_records')
    bindings = report.get('parameter_native_bindings')
    _require(isinstance(records, dict) and set(records) == {'baseline', 'candidate', 'source'} and isinstance(bindings, dict) and set(bindings) == set(records), 'Exact native arm bindings required.')
    native_specs = {}
    for arm, local in (('baseline', control_checkpoint), ('candidate', lr_checkpoint), ('source', source_checkpoint)):
        if enclosed_natives is not None:
            local = io.BytesIO(enclosed_natives[arm])
        else:
            _require(local is not None, 'Actual source/control/candidate native paths required for live-report audit.')
            local = local['path'] if isinstance(local, dict) else local
        manifest, params, binding = _native_info(local, records[arm], configs[arm], arm)
        _require(_canonical(binding) == _canonical(bindings[arm]), 'Actual native policy digest/revision/config binding differs: ' + arm)
        native_specs[arm] = manifest['leaves']
    _require(all(_canonical(specs) == _canonical(native_specs['baseline']) for specs in native_specs.values()), 'Actual native architecture/state specifications differ.')
    _require(bindings['baseline']['updates'] == bindings['candidate']['updates'], 'Incomplete A/B: unequal completed rollout counts.')
    files, transports = report.get('raw_files'), report.get('transport_files')
    _require(report.get('target_rollouts_per_trained_arm') == TARGET_ROLLOUTS and report.get('target_updates_per_trained_arm') == TARGET_UPDATES and report.get('games') == 1536 and report.get('unique_policy_games') == 1536, 'Exact128-rollout/1536-game campaign target differs.')
    count = 3 * len(plan['style_ids'])
    _require(isinstance(files, list) and isinstance(transports, list) and len(files) == len(transports) == count, 'Complete48 JSON+48 NPZ coverage required.')
    for values, field in ((files, 'raw_manifest_sha256'), (transports, 'transport_manifest_sha256')):
        _require(len({row['path'] for row in values}) == count and report.get(field) == _sha(_canonical(values).encode()), 'Exact unique artifact manifest differs.')
        for row in values:
            _require(set(row) == {'path', 'bytes', 'sha256'}, 'Exact artifact identity schema required.')
    raw = {'baseline': {}, 'candidate': {}, 'source': {}}
    roots_seen = set()
    for info in files:
        row = _json(read(info))
        arm, style = row.get('arm'), row.get('style_id')
        _require(arm in raw and type(style) is int and style in plan['style_ids'] and style not in raw[arm], 'Unknown/duplicate arm/style.')
        _require(row.get('comparison_name') == comparison and row.get('branch_label') == BRANCH_LABELS[arm] and row.get('experiment_id') == experiment['experiment_id'] and row.get('run_id') == experiment['run_id'] and row.get('native_config_signature') == signatures[arm] and _canonical(row.get('checkpoint')) == _canonical(records[arm]), 'Raw per-arm checkpoint/config binding differs.')
        _require(row.get('mode') == 'development' and row.get('seed_name') == 'development' and type(row.get('seed')) is int and row['seed'] == plan['seed'], 'Raw frozen seed/mode binding differs.')
        teams = row.get('learner_teams')
        _require(isinstance(teams, list) and teams == [0, 1] * plan['pairs_per_style'] and all(type(v) is int for v in teams), 'Exact duplicate-pair roles required.')
        transport = row.get('transport_snapshot')
        _require(transport in transports, 'Raw transport identity missing from report.')
        with np.load(io.BytesIO(read(transport)), allow_pickle=False) as saved:
            _require(sorted(saved.files) == [f'output_{i:02d}' for i in range(7)], 'Exactly seven saved kernel outputs required.')
            recomputed = frozen.validate_output(tuple(saved[f'output_{i:02d}'] for i in range(7)), np.asarray(teams, np.int32), max_hands=plan['max_hands'])
        _require(all(_canonical(row.get(key)) == _canonical(value) for key, value in recomputed.items()) and all(row.get(key) is True for key in ('complete', 'legal', 'finite')), 'Raw JSON differs from actual saved kernel outputs or reliability failed.')
        roots = row.get('pair_root_keys')
        _require(isinstance(roots, list) and len(roots) == plan['pairs_per_style'] and all(isinstance(key, list) and len(key) == 2 and all(type(v) is int and 0 <= v < 2**32 for v in key) for key in roots), 'Exact uint32 pair roots required.')
        raw[arm][style] = row
    for style in plan['style_ids']:
        _require(all(style in raw[arm] for arm in raw), 'Missing exact development style.')
        _require(all(raw['baseline'][style]['pair_root_keys'] == raw[arm][style]['pair_root_keys'] for arm in raw), 'All three policies must share exact roots.')
        roots = [tuple(key) for key in raw['baseline'][style]['pair_root_keys']]
        _require(len(set(roots)) == len(roots) and not (set(roots) & roots_seen), 'Duplicate roots within/across styles.')
        roots_seen.update(roots)
    analyses = _all_analyses(frozen, raw, plan)
    analysis = analyses['primary']
    _require(_canonical(report.get('comparisons')) == _canonical(analyses) and _canonical(report.get('analysis')) == _canonical(analysis) and _canonical(report.get('primary_analysis')) == _canonical(analysis) and report.get('aggregate_difference') == report.get('primary_aggregate_difference') == analysis['aggregate_difference'] and report.get('primary_complete') is True, 'Frozen primary/secondary paired statistics differ.')
    _require(_canonical(report.get('development_groups')) == _canonical(analyses['primary_development_groups']), 'Adaptation/retention summaries differ.')
    return dict(status='PASS', audit_scope='CONTENT_ONLY', comparison_name=comparison, experiment_binding=experiment, exact_evaluated_natives_embedded=enclosed_natives is not None,
                games=3 * len(plan['style_ids']) * plan['games_per_style'], raw_file_count=count,
                transport_file_count=count, native_configs_truthfully_bound=True,
                root_generation_independently_recomputed=False,
                root_audit_scope='Paired unique saved keys; host audit does not reimplement JAX Threefry.',
                training_authorized=False, ack_authorized=False, playing_strength_claim=False)


def package_lr_ab_evaluation_evidence(report_path, output_path, *, control_checkpoint, lr_checkpoint, source_checkpoint):
    """Package complete OR partial evidence with all three EXACT evaluated natives.

    Packaging is transport preservation, never an independent content-audit PASS.
    audit_lr_ab_evaluation rejects partial reports and checks actual native files.
    """
    report = _json(Path(report_path).read_bytes())
    _require(report.get('format') == FORMAT, 'LR A/B evaluation report format required.')
    output = Path(output_path)
    _require(not output.exists() and not output.is_symlink(), 'Never overwrite evaluation evidence.')
    members = {'evaluation_report.json': Path(report_path).read_bytes()}
    for arm, local in (('baseline', control_checkpoint), ('candidate', lr_checkpoint), ('source', source_checkpoint)):
        local = local['path'] if isinstance(local, dict) else local
        raw = Path(local).read_bytes()
        recorded = report['checkpoint_records'][arm]
        _require(len(raw) == recorded['bytes'] and _sha(raw) == recorded['sha256'], 'Refuse substituted final-save native; exact evaluated native required: ' + arm)
        members[NATIVE_MEMBERS[arm]] = raw
    for key, prefix, suffix in (('raw_files', 'raw', 'json'), ('transport_files', 'transport', 'npz')):
        for index, info in enumerate(report.get(key, [])):
            raw = Path(info['path']).read_bytes()
            _require(len(raw) == info['bytes'] and _sha(raw) == info['sha256'], 'Evaluation source artifact changed before packaging.')
            members[f'{prefix}_{index:03d}.{suffix}'] = raw
    manifest = dict(format=ARCHIVE_FORMAT, members={name: dict(bytes=len(raw), sha256=_sha(raw)) for name, raw in members.items()})
    members['evaluation_archive_manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(output) as archive:
        _require(archive.testzip() is None and len(archive.namelist()) == len(members) and set(archive.namelist()) == set(members), 'Evidence archive CRC/member mismatch.')
        for name, raw in members.items():
            _require(archive.read(name) == raw, 'Evidence archive content mismatch.')
    return _identity(output)
