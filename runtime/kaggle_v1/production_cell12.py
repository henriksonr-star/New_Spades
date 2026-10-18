"""Size real rollout/optimizer workloads for available GPU memory.

No dummy allocations or GPU-memory preallocation are used. Compilation is done
with abstract arrays, so an oversized candidate is rejected before its rollout
buffers are allocated. Exact checkpoint resumes retain their saved shapes.
"""
import csv
import gc
import math
import numbers
import os
import subprocess

GPU_TUNING = dict(enabled=True, target_gib=14.0, reserve_mib=1024,
                  safety_factor=1.45, extra_guard_mib=256, initial_envs=256, env_quantum=128,
                  maximum_envs=16384, maximum_profiles=3,
                  minibatches_per_epoch=8)
_GPU_TUNED_STEP = None
_GPU_TUNED_CONFIG = None
_GPU_PROFILE_REPORT = None
_GPU_LAST_ADMISSION = None
_GPU_TUNED_RUNTIME = None
RUNTIME_SAFETY_REVISION = "checkpoint-memory-safety-v2-kaggle-uuid-v1"


_GPU_COUNTER_FIELDS = (
    ('bytes_in_use', 'jax_live_bytes'),
    ('peak_bytes_in_use', 'jax_peak_live_bytes'),
    ('pool_bytes', 'jax_pool_bytes'),
    ('peak_pool_bytes', 'jax_peak_pool_bytes'),
    ('bytes_reserved', 'jax_reserved_bytes'),
    ('peak_bytes_reserved', 'jax_peak_reserved_bytes'),
    ('bytes_limit', 'jax_limit_bytes'),
    ('largest_free_block_bytes', 'jax_largest_free_block_bytes'),
)


def _gpu_counter_record(value):
    """Retain raw values in JSON-safe form without silently fixing counters."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    return repr(value)


def _gpu_snapshot_allocator_fields(stats):
    """Pure mapper for independently testable telemetry preservation."""
    if not isinstance(stats, dict):
        return dict(jax_memory_stats_raw={}, jax_memory_stats_error='not_a_dictionary')
    report = {'jax_memory_stats_raw': {
        str(key): _gpu_counter_record(value) for key, value in stats.items()}}
    for source, target in _GPU_COUNTER_FIELDS:
        if source in stats:
            # Retaining invalid input allows the budget helper to fail closed;
            # int(nan), int(inf), and string-to-int coercions are not attempted.
            report[target] = _gpu_counter_record(stats[source])
    return report


def _gpu_memory_metric_values(snapshot):
    """Display only valid byte counters; raw malformed telemetry stays in reports."""
    result = {}
    for name, value in snapshot.items():
        if (not name.endswith('_bytes') or isinstance(value, bool)
                or not isinstance(value, numbers.Integral) or value < 0):
            continue
        try:
            gib = value / 2**30
        except OverflowError:
            continue
        if math.isfinite(gib):
            result[name.removesuffix('_bytes') + '_gib'] = gib
    return result


def _gpu_selected_identity():
    """Selected physical board identity; logical device0 never means physical0."""
    platform = globals().get('_SPADES_KAGGLE_PLATFORM')
    if not isinstance(platform, dict):
        raise RuntimeError('Kaggle selected physical board metadata required.')
    selected = platform['selected_gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != selected:
        raise RuntimeError('Selected GPU visibility changed; no launch.')
    return (selected, platform['selected_gpu_pci_bus_id'],
            platform['selected_gpu_name'], platform['selected_gpu_total_bytes'],
            platform['driver_version'], os.environ.get('CUDA_VISIBLE_DEVICES'))


def _gpu_memory_snapshot(device=None):
    """UUID-bound one-board telemetry; all original allocator guards retained."""
    identity = _gpu_selected_identity()
    if device is None:
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != 'gpu' or 'T4' not in str(devices[0].device_kind):
            raise RuntimeError('Exactly one JAX-visible selected T4 required.')
        device = devices[0]
    report = dict(platform=device.platform, selected_gpu_uuid=identity[0])
    if device.platform != 'gpu':
        return report
    try:
        stats = device.memory_stats() or {}
    except (AttributeError, RuntimeError) as exc:
        stats = {}
        report['jax_memory_stats_error'] = type(exc).__name__
    report.update(_gpu_snapshot_allocator_fields(stats))
    try:
        query = subprocess.run(
            ['nvidia-smi', '--id=' + identity[0],
             '--query-gpu=uuid,index,pci.bus_id,name,memory.total,memory.used,memory.free,driver_version',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            check=False, timeout=5)
        rows = list(csv.reader(query.stdout.strip().splitlines()))
        if query.returncode != 0 or len(rows) != 1 or len(rows[0]) != 8:
            raise ValueError('Selected UUID board telemetry unavailable/ambiguous')
        uuid, index, pci, name, total, used, free, driver = [value.strip() for value in rows[0]]
        values = [float(value) for value in (total, used, free)]
        if (uuid != identity[0] or pci != identity[1] or name != identity[2] or driver != identity[4]
                or not index.isdecimal()
                or not all(math.isfinite(value) and value >= 0 and value.is_integer() for value in values)
                or int(values[0] * 2**20) != identity[3]
                or values[1] > values[0] or values[2] > values[0]
                or values[1] + values[2] > values[0]):
            raise ValueError('Selected UUID/PCI/name/driver/capacity identity changed or telemetry invalid')
        report.update(gpu_name=name, gpu_index=int(index), gpu_uuid=uuid, gpu_pci_bus_id=pci,
                      board_total_bytes=int(values[0] * 2**20),
                      board_used_bytes=int(values[1] * 2**20),
                      board_free_bytes=int(values[2] * 2**20))
    except (OSError, ValueError, OverflowError, subprocess.TimeoutExpired) as exc:
        report['selected_board_telemetry_error'] = type(exc).__name__ + ': ' + str(exc)
    return report


def _gpu_memory_budget_details(snapshot, options):
    """Additional allocation budget, its raw counter source, and any fallback.

    pool_bytes is the current BFC allocator pool total, including live bytes.
    bytes_reserved is accepted as a legacy total only when the caller explicitly
    declares that interpretation verified for its backend. The counters are
    never added together. Peak counters never provide current capacity.
    """
    target_value, reserve_value = options['target_gib'], options['reserve_mib']
    if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
           for value in (target_value, reserve_value)):
        raise ValueError('GPU target and reserve must be finite numbers.')
    target, reserve = float(target_value) * 2**30, float(reserve_value) * 2**20
    if not (math.isfinite(target) and target > 0 and math.isfinite(reserve) and reserve >= 0):
        raise ValueError('GPU target and reserve must be finite; target must be positive.')
    legacy_verified = options.get('legacy_reserved_is_total_pool', False)
    if not isinstance(legacy_verified, bool):
        raise ValueError('Legacy reserved semantics must be explicitly True or False.')

    counters = {name: _gpu_counter_record(snapshot[name]) for name in (
        'board_total_bytes', 'board_used_bytes', 'board_free_bytes',
        'jax_live_bytes', 'jax_pool_bytes', 'jax_peak_pool_bytes',
        'jax_reserved_bytes', 'jax_peak_reserved_bytes', 'jax_peak_live_bytes',
        'jax_limit_bytes', 'jax_largest_free_block_bytes') if name in snapshot}
    result = dict(status='starting', available_new_working_bytes=None,
                  requested_target_bytes=int(target), board_reserve_bytes=int(reserve),
                  raw_counters=counters, pool_counter_source=None,
                  pool_counter_snapshot_field=None, pool_counter_value_bytes=None,
                  pool_counter_status='unavailable', reusable_idle_pool_bytes=0,
                  legacy_reserved_is_total_pool=legacy_verified,
                  overlap_credit_bytes=0, reasons=[])

    def invalid(reason):
        result.update(status='invalid_telemetry', available_new_working_bytes=0)
        result['reasons'].append(reason)
        return result

    def valid_bytes(value):
        return (not isinstance(value, bool) and isinstance(value, numbers.Integral)
                and value >= 0)

    for name in ('board_total_bytes', 'board_free_bytes'):
        if name not in snapshot:
            result.update(status='telemetry_unavailable')
            result['reasons'].append('missing_' + name)
            return result
    core = ('board_total_bytes', 'board_free_bytes', 'board_used_bytes',
            'jax_live_bytes', 'jax_limit_bytes')
    for name in core:
        if name in snapshot and not valid_bytes(snapshot[name]):
            return invalid('invalid_' + name)
    total, free = int(snapshot['board_total_bytes']), int(snapshot['board_free_bytes'])
    if total <= 0 or free > total:
        return invalid('invalid_board_total_or_free_bounds')
    if 'board_used_bytes' in snapshot:
        used = int(snapshot['board_used_bytes'])
        if used > total or used + free > total:
            return invalid('invalid_board_used_bounds')
        # Retain reserved/rounding discrepancy. It is never reusable capacity.
        result['board_total_minus_used_minus_free_bytes'] = total - used - free

    live_known = 'jax_live_bytes' in snapshot
    live = int(snapshot['jax_live_bytes']) if live_known else total - free
    if live > total - free:
        return invalid('live_exceeds_occupied_board_capacity')
    if not live_known:
        result['reasons'].append('missing_live_uses_occupied_board_upper_bound')
    result.update(live_bytes_deducted=live, live_counter_known=live_known)
    limit = snapshot.get('jax_limit_bytes')
    if limit is not None:
        limit = int(limit)
        if limit > total:
            return invalid('allocator_limit_exceeds_board_total')
        result['allocator_remaining_bytes'] = max(0, limit - live)
    else:
        result['reasons'].append('allocator_limit_unavailable')

    pool_field = None
    if 'jax_pool_bytes' in snapshot:
        pool_field = 'jax_pool_bytes'
        raw_source = 'pool_bytes'
    elif 'jax_reserved_bytes' in snapshot and legacy_verified:
        pool_field = 'jax_reserved_bytes'
        raw_source = 'bytes_reserved'
    elif 'jax_reserved_bytes' in snapshot:
        result['pool_counter_status'] = 'legacy_semantics_unverified'
        result['reasons'].append('bytes_reserved_is_not_assumed_to_be_a_total_pool')

    idle = 0
    if pool_field is not None:
        value = snapshot[pool_field]
        result.update(pool_counter_source=raw_source,
                      pool_counter_snapshot_field=pool_field,
                      pool_counter_value_bytes=_gpu_counter_record(value))
        reason = None
        if not live_known:
            reason = 'pool_credit_requires_known_live_bytes'
        elif not valid_bytes(value):
            reason = 'pool_counter_is_not_nonnegative_integer_bytes'
        elif value < live:
            reason = 'pool_counter_is_below_live_bytes'
        elif value > total - free:
            reason = 'pool_counter_exceeds_occupied_board_capacity'
        elif limit is not None and value > limit:
            reason = 'pool_counter_exceeds_allocator_limit'
        if reason:
            result['pool_counter_status'] = 'unusable_zero_credit'
            result['reasons'].append(reason)
        else:
            idle = int(value) - live
            result['pool_counter_status'] = 'validated_current_total'
    else:
        result['reasons'].append('no_validated_current_pool_counter_zero_credit')

    physical = max(0, min(total - live, free + idle) - reserve)
    limits = [int(target), int(physical)]
    if limit is not None:
        limits.append(limit - live)
    result.update(status='ok', reusable_idle_pool_bytes=idle,
                  physical_remaining_after_reserve_bytes=int(physical),
                  available_new_working_bytes=max(0, int(min(limits))))
    if result['available_new_working_bytes'] == 0:
        result['status'] = 'no_capacity'
    return result


def _gpu_working_budget(snapshot, options):
    """Compatible scalar API; detailed decisions remain separately inspectable."""
    return _gpu_memory_budget_details(snapshot, options)['available_new_working_bytes']


def _gpu_candidate_config(base_cfg, n_env, options):
    cfg = dict(base_cfg)
    n_env = int(n_env)
    cfg['n_env'] = n_env
    total = n_env * int(cfg['rollout_len'])
    # Preserve PPO optimizer cadence while scaling data parallelism. A fixed
    # small minibatch on a huge rollout would perform hundreds of sequential
    # Adam steps, trigger KL early stopping, and waste most sampled decisions.
    minibatches = int(options.get('minibatches_per_epoch', 8))
    if minibatches < 1 or total % minibatches:
        raise ValueError('GPU candidate decisions must divide into whole minibatches.')
    cfg['minibatch_size'] = total // minibatches
    return learning_config(cfg)


def _gpu_compiled_bytes(analysis):
    """XLA executable buffer accounting; aliases occupy memory only once."""
    names = ('argument_size_in_bytes', 'output_size_in_bytes',
             'temp_size_in_bytes', 'alias_size_in_bytes')
    if analysis is None or any(getattr(analysis, name, None) is None for name in names):
        raise RuntimeError('This JAX backend does not expose executable memory accounting.')
    values = {name: int(getattr(analysis, name)) for name in names}
    if any(value < 0 for value in values.values()):
        raise RuntimeError('The compiler returned invalid memory accounting.')
    values['total_bytes'] = max(0, values['argument_size_in_bytes']
        + values['output_size_in_bytes'] + values['temp_size_in_bytes']
        - values['alias_size_in_bytes'])
    return values


def _gpu_tree_bytes(tree):
    return sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize
               for x in jax.tree_util.tree_leaves(tree))


def _gpu_profile_shape(cfg, options=None):
    """Compile the actual code, including snapshots, finite guards and shuffle."""
    opts = dict(GPU_TUNING) if options is None else dict(options)
    profile_admissions = []
    step = make_train_step(cfg)
    abstract_state = jax.eval_shape(lambda: initialize_training(cfg))
    state_bytes = _gpu_tree_bytes(abstract_state)
    profile_admissions.append(_gpu_memory_admission(state_bytes, opts, "before collect profile"))
    after_collect, abstract_batch, _ = jax.eval_shape(step.collect, abstract_state)
    collect_executable = step.collect.lower(abstract_state).compile()
    collect_bytes = _gpu_compiled_bytes(collect_executable.memory_analysis())
    profile_admissions.append(_gpu_memory_admission(collect_bytes["total_bytes"] + state_bytes, opts, "before optimize profile"))
    optimize_executable = step.optimize.lower(after_collect, abstract_batch).compile()
    optimize_bytes = _gpu_compiled_bytes(optimize_executable.memory_analysis())
    # Match the actual two-stage supervised runtime. Jitting the whole Python
    # wrapper could fuse away buffers that remain live between the real collect
    # and optimize calls, underestimating the teacher initialization peak.
    bootstrap = make_bootstrap_step(cfg)
    after_teacher, abstract_teacher_data = jax.eval_shape(bootstrap.collect, abstract_state)
    profile_admissions.append(_gpu_memory_admission(max(collect_bytes["total_bytes"], optimize_bytes["total_bytes"]) + state_bytes, opts, "before teacher collect profile"))
    teacher_collect_executable = bootstrap.collect.lower(abstract_state).compile()
    teacher_collect_bytes = _gpu_compiled_bytes(teacher_collect_executable.memory_analysis())
    profile_admissions.append(_gpu_memory_admission(max(collect_bytes["total_bytes"], optimize_bytes["total_bytes"], teacher_collect_bytes["total_bytes"]) + state_bytes, opts, "before teacher optimize profile"))
    teacher_optimize_executable = bootstrap.optimize.lower(after_teacher, abstract_teacher_data).compile()
    teacher_optimize_bytes = _gpu_compiled_bytes(teacher_optimize_executable.memory_analysis())
    state_bytes = _gpu_tree_bytes(abstract_state)
    # The caller retains the last completed state for rollback throughout the
    # entire update. Optimize also receives the post-rollout state and batch:
    # those buffers, shuffled copies, gradients, Adam and finite checks are
    # already in optimize's executable accounting. This extra full state is
    # conservative because some unchanged leaves alias between both states.
    peak = max(collect_bytes['total_bytes'], optimize_bytes['total_bytes'],
               teacher_collect_bytes['total_bytes'], teacher_optimize_bytes['total_bytes']) + state_bytes
    report = dict(n_env=cfg['n_env'], rollout_len=cfg['rollout_len'],
                  minibatch_size=cfg['minibatch_size'], width=cfg['width'],
                  state_bytes=state_bytes, profile_admissions=profile_admissions,
                  rollout_batch_bytes=_gpu_tree_bytes(abstract_batch),
                  collect=collect_bytes, optimize=optimize_bytes,
                  bootstrap_collect=teacher_collect_bytes, bootstrap_optimize=teacher_optimize_bytes,
                  estimated_peak_bytes=int(peak))
    # Keep the selected jitted functions alive to reuse their compilation. The
    # standalone bootstrap executables are used only for memory estimation.
    step.bootstrap = bootstrap
    del collect_executable, optimize_executable, teacher_collect_executable, teacher_optimize_executable, bootstrap
    gc.collect()
    return report, step


def _gpu_guarded_bytes(estimated_bytes, options):
    """The measured calibration is a floor at every allocation admission."""
    if (isinstance(estimated_bytes, bool) or not isinstance(estimated_bytes, numbers.Integral)
            or estimated_bytes <= 0):
        raise ValueError('The compiler/state estimate must be positive integer bytes.')
    safety, guard = options['safety_factor'], options.get('extra_guard_mib', 256)
    if any(isinstance(x, bool) or not isinstance(x, numbers.Real) for x in (safety, guard)):
        raise ValueError('GPU safety factor and guard must be finite numbers.')
    if not (math.isfinite(safety) and safety >= 1.45
            and math.isfinite(guard) and guard >= 256):
        raise ValueError('GPU admissions require at least 1.45 times the estimate plus 256 MiB.')
    return math.ceil(int(estimated_bytes) * float(safety)) + math.ceil(float(guard) * 2**20)


def _gpu_runtime_identity():
    """Live function objects, stack and revision must still match the profile."""
    functions = tuple(globals().get(name) for name in (
        'make_train_step', 'make_bootstrap_step', 'initialize_training', 'finish_bootstrap'))
    versions = tuple(getattr(globals().get(name), '__version__', None)
                     for name in ('jax', 'flax', 'optax'))
    return functions + versions + (globals().get('TRAINING_REVISION'), RUNTIME_SAFETY_REVISION) + _gpu_selected_identity()


def _gpu_runtime_summary():
    """Serializable diagnostics; live object equality is checked separately."""
    return dict(function_ids={name: id(globals().get(name)) for name in (
        'make_train_step', 'make_bootstrap_step', 'initialize_training', 'finish_bootstrap')},
        stack={name: getattr(globals().get(name), '__version__', None)
               for name in ('jax', 'flax', 'optax')},
        training_revision=globals().get('TRAINING_REVISION'),
        runtime_safety_revision=RUNTIME_SAFETY_REVISION)


def _gpu_current_budget_details(snapshot, options):
    """Production admissions require current live and allocator-ceiling evidence.

    The pure accounting helper retains conservative compatibility fallbacks for
    diagnostics. Production refuses to train or allocate without these counters.
    """
    details = _gpu_memory_budget_details(snapshot, options)
    missing = [name for name in ('board_total_bytes', 'board_free_bytes',
               'jax_live_bytes', 'jax_limit_bytes') if name not in snapshot]
    details['admission_missing_required_fields'] = missing
    if missing:
        details.update(status='telemetry_unavailable', available_new_working_bytes=0)
        details['reasons'].append('production_requires_current_board_live_and_allocator_limit')
    return details


def _gpu_memory_admission(estimated_bytes, options, stage, *, snapshot=None):
    """Refresh board/pool capacity; give no credit for live input overlap."""
    global _GPU_LAST_ADMISSION
    guarded = _gpu_guarded_bytes(estimated_bytes, options)
    snapshot = _gpu_memory_snapshot() if snapshot is None else snapshot
    report = dict(stage=str(stage), estimated_peak_bytes=int(estimated_bytes),
                  guarded_peak_bytes=guarded, snapshot=snapshot,
                  overlap_credit_bytes=0)
    if snapshot.get('platform') != 'gpu':
        report['status'] = 'no_gpu_unchanged'
        _GPU_LAST_ADMISSION = report
        return report
    details = _gpu_current_budget_details(snapshot, options)
    budget = details['available_new_working_bytes']
    report.update(budget_details=details, budget_bytes=budget)
    report['status'] = ('admitted' if details['status'] == 'ok'
                        and budget is not None and guarded <= budget else 'rejected')
    _GPU_LAST_ADMISSION = report
    if report['status'] != 'admitted':
        raise RuntimeError(f'GPU memory admission rejected before {stage}: '
                           f'{guarded} guarded bytes, {budget} available; '
                           f'{details["status"]}. Last accepted state is retained; '
                           'inspect _GPU_LAST_ADMISSION before any retry.')
    return report


def _gpu_choose_configuration(base_cfg, budget_bytes, options, profiler, budget_provider=None):
    """Bounded shape search. Compiler errors stop immediately, without retry.

    A capacity gate raised during partial compilation also stops the entire
    search, even if an earlier fully compiled candidate fit. This conservative
    fail-fast policy avoids continuing after any admission refusal.
    """
    quantum, max_env = int(options['env_quantum']), int(options['maximum_envs'])
    attempts = int(options['maximum_profiles'])
    _gpu_guarded_bytes(1, options)
    if quantum < 1 or max_env < quantum or attempts < 1:
        raise ValueError('Invalid GPU profiling limits.')
    minimum = min(quantum, max_env)
    n_env = max(minimum, min(max_env, int(options['initial_envs']) // quantum * quantum))
    candidates, tried, best = [], set(), None
    upper_failure = max_env + quantum
    for _ in range(attempts):
        if n_env in tried:
            break
        tried.add(n_env)
        cfg = _gpu_candidate_config(base_cfg, n_env, options)
        # Do not interpret compiler RuntimeError/MemoryError as a size signal.
        # No next shape or live state operation runs after either failure.
        profile, handle = profiler(cfg)
        estimate = profile['estimated_peak_bytes']
        guarded = _gpu_guarded_bytes(estimate, options)
        if budget_provider is not None:
            budget_bytes = budget_provider()
        if (isinstance(budget_bytes, bool) or not isinstance(budget_bytes, numbers.Integral)
                or budget_bytes <= 0):
            raise RuntimeError('Fresh GPU memory telemetry has no usable capacity.')
        profile = dict(profile, guarded_peak_bytes=guarded,
                       admission_budget_bytes=int(budget_bytes),
                       fits_budget=guarded <= budget_bytes)
        candidates.append(profile)
        if guarded <= budget_bytes:
            if best is None or n_env > best[0]['n_env']:
                best = (cfg, profile, handle)
            next_env = int(n_env * budget_bytes / guarded) // quantum * quantum
            next_env = min(max_env, upper_failure - quantum, max(n_env + quantum, next_env))
            if guarded >= .90 * budget_bytes or next_env <= n_env:
                break
        else:
            upper_failure = min(upper_failure, n_env)
            next_env = int(n_env * budget_bytes / guarded * .95) // quantum * quantum
            next_env = max(minimum, min(n_env - quantum, next_env))
        if next_env in tried or next_env < minimum:
            break
        n_env = next_env
    # An earlier accepted candidate may have lost capacity while later shapes
    # compiled. Its caller performs a separate fresh final admission.
    return best, candidates, []


def configure_gpu_workload(base_cfg, *, exact_resume=False, options=None):
    """Profile new shapes or characterize exactly the saved resume shape.

    Exact resume never changes model, optimizer, RNG, games, minibatches or
    counters. Its saved shape must pass characterization before an update.
    """
    global _GPU_TUNED_STEP, _GPU_TUNED_CONFIG, _GPU_PROFILE_REPORT, _GPU_TUNED_RUNTIME
    _GPU_TUNED_STEP = _GPU_TUNED_CONFIG = _GPU_PROFILE_REPORT = _GPU_TUNED_RUNTIME = None
    runtime_identity = _gpu_runtime_identity()
    cfg = learning_config(base_cfg)
    opts = dict(GPU_TUNING)
    if options:
        opts.update(options)
    _gpu_guarded_bytes(1, opts)
    snapshot = _gpu_memory_snapshot()
    report = dict(requested_gib=float(opts['target_gib']), snapshot=snapshot,
                  exact_resume=bool(exact_resume), selected_config=dict(cfg),
                  options=dict(opts), profiles=[], failures=[], runtime_signature=_gpu_runtime_summary())
    _GPU_PROFILE_REPORT = report
    if snapshot.get('platform') != 'gpu':
        report['status'] = 'exact_resume_unchanged' if exact_resume else 'no_gpu_unchanged'
        print('GPU workload sizing skipped: this runtime has no JAX GPU.')
        return cfg, report
    initial_details = _gpu_current_budget_details(snapshot, opts)
    report['initial_budget_details'] = initial_details
    if (initial_details['status'] != 'ok'
            or not initial_details['available_new_working_bytes']):
        report['status'] = 'rejected'
        raise RuntimeError('GPU workload profiling requires valid current memory telemetry and capacity.')
    budget_snapshots = []
    def refreshed_budget():
        fresh = _gpu_memory_snapshot()
        details = _gpu_current_budget_details(fresh, opts)
        budget_snapshots.append(dict(snapshot=fresh, budget_details=details))
        if details['status'] != 'ok':
            raise RuntimeError('GPU memory telemetry invalid or unavailable after profiling.')
        return details['available_new_working_bytes']
    report['profile_budget_snapshots'] = budget_snapshots
    def profile(candidate):
        print(f'Compiling memory profile: {candidate["n_env"]} games × '
              f'{candidate["rollout_len"]} steps; minibatch {candidate["minibatch_size"]}.', flush=True)
        # Check a fresh capacity snapshot before compilation as well. The
        # compiler itself is abstract; a compile failure still stops this call.
        if not refreshed_budget():
            raise RuntimeError('No GPU memory capacity remains before compilation.')
        return _gpu_profile_shape(candidate, options=opts)
    try:
        if exact_resume or not opts['enabled']:
            selected, step = profile(cfg)
            selected = dict(selected,
                guarded_peak_bytes=_gpu_guarded_bytes(selected['estimated_peak_bytes'], opts),
                admission_budget_bytes=refreshed_budget())
            selected['fits_budget'] = selected['guarded_peak_bytes'] <= selected['admission_budget_bytes']
            report['profiles'] = [selected]
        else:
            best, profiles, failures = _gpu_choose_configuration(cfg,
                initial_details['available_new_working_bytes'], opts, profile,
                budget_provider=refreshed_budget)
            report.update(profiles=profiles, failures=failures)
            if best is None:
                raise RuntimeError('No profiled training shape fits the fresh guarded GPU memory budget.')
            cfg, selected, step = best
        report.update(selected_config=dict(cfg), selected_profile=selected)
        final = _gpu_memory_admission(selected['estimated_peak_bytes'], opts,
            'selected exact resume' if exact_resume else 'selected workload')
        if runtime_identity != _gpu_runtime_identity():
            raise RuntimeError('Trainer definitions, runtime revision or library stack changed during profiling.')
        report.update(status='exact_resume_profiled_unchanged' if exact_resume else 'profiled',
                      final_admission=final)
    except (RuntimeError, MemoryError) as exc:
        report.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
        _GPU_TUNED_STEP = _GPU_TUNED_CONFIG = None
        raise
    _GPU_TUNED_CONFIG, _GPU_TUNED_STEP = dict(cfg), step
    _GPU_TUNED_RUNTIME = runtime_identity
    print(f'GPU workload admitted: {cfg["n_env"]} environments, '
          f'{cfg["n_env"] * cfg["rollout_len"]:,} decisions/update; '
          f'{selected["guarded_peak_bytes"] / 2**30:.2f} GiB with calibration and guard.', flush=True)
    return cfg, report


def gpu_admit_workload(cfg, stage):
    """Require the current exact configuration's profile at each outer boundary."""
    snapshot = _gpu_memory_snapshot()
    if snapshot.get('platform') != 'gpu':
        return dict(stage=str(stage), status='no_gpu_unchanged')
    cfg = learning_config(cfg)
    report = _GPU_PROFILE_REPORT or {}
    if (report.get('status') not in ('profiled', 'exact_resume_profiled_unchanged')
            or dict(cfg) != report.get('selected_config')
            or _GPU_TUNED_RUNTIME != _gpu_runtime_identity()
            or 'selected_profile' not in report):
        raise RuntimeError(f'No matching admitted GPU memory profile before {stage}; '
                           'characterize this exact configuration before training.')
    return _gpu_memory_admission(report['selected_profile']['estimated_peak_bytes'],
        report['options'], stage, snapshot=snapshot)


def gpu_admit_checkpoint_load(cfg, state_bytes):
    """Bound array transfer before any saved leaf is placed on the GPU.

    This is a state-transfer admission, not a characterization of training or
    eager initialization. The loader uses an abstract template, avoiding the
    former extra real template allocation. Training receives its separate full
    exact-shape profile and stage checks after loading.
    """
    return _gpu_memory_admission(state_bytes, dict(GPU_TUNING), 'checkpoint array transfer')


def gpu_train_step(cfg):
    """Reuse profiled functions only when the complete configuration matches."""
    return (_GPU_TUNED_STEP if _GPU_TUNED_STEP is not None
            and dict(cfg) == _GPU_TUNED_CONFIG
            and _GPU_TUNED_RUNTIME == _gpu_runtime_identity() else make_train_step(cfg))


def gpu_bootstrap_step(cfg):
    """Reuse the profiled teacher collect/optimize objects for this exact config."""
    return (_GPU_TUNED_STEP.bootstrap if _GPU_TUNED_STEP is not None
            and dict(cfg) == _GPU_TUNED_CONFIG
            and _GPU_TUNED_RUNTIME == _gpu_runtime_identity()
            and hasattr(_GPU_TUNED_STEP, 'bootstrap') else make_bootstrap_step(cfg))


def gpu_memory_metrics():
    """Only valid finite scalar byte counters enter dashboard arithmetic."""
    snapshot = _gpu_memory_snapshot()
    result = _gpu_memory_metric_values(snapshot)
    if snapshot.get('platform') == 'gpu':
        result['gpu_target_gib'] = float((_GPU_PROFILE_REPORT or {}).get(
            'requested_gib', GPU_TUNING['target_gib']))
    if _GPU_PROFILE_REPORT and 'selected_profile' in _GPU_PROFILE_REPORT:
        result['gpu_estimated_working_gib'] = (
            _GPU_PROFILE_REPORT['selected_profile']['estimated_peak_bytes'] / 2**30)
    return result
