"""Two-device GSPMD execution of one frozen Spades training state.

Definitions only: no framework import, provider action, checkpoint write, or
training is performed on import. The caller owns exact-native admission and
the externally reserved active clock. All kernels use global array semantics;
only environment and sample axes are partitioned. Adam, the PRNG key, counters,
and the historical bank remain replicated parts of ONE training state.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time

FORMAT = 'spades-dual-t4-gspmd-v1'
DEVICE_COUNT = 2
ENV_FIELDS = ('envs', 'league_kind', 'league_style', 'league_snapshot', 'learner_team')
REPLICATED_FIELDS = ('params', 'opt_state', 'rng', 'snapshot_params', 'snapshot_count',
    'snapshot_cursor', 'snapshot_retiring', 'last_snapshot_update', 'updates',
    'decisions_per_update', 'bootstrap_updates', 'bootstrap_complete', 'env_steps_low',
    'env_steps_high', 'base_updates')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def select_platform_nojax(legacy_runtime, expected_gpu_uuids=None):
    """Inventory both physical T4s before importing JAX; no launch or install."""
    inventory = legacy_runtime.inventory_kaggle_nojax()
    boards = sorted(inventory['physical_gpu_inventory'], key=lambda row: row['index'])
    if len(boards) != DEVICE_COUNT or any(re.search(r'(?:^|\s)T4(?:$|\s)', b['name']) is None for b in boards):
        raise RuntimeError('Exactly two physical T4 boards are required; no fallback.')
    uuids = [b['uuid'] for b in boards]
    if expected_gpu_uuids is not None:
        if len(expected_gpu_uuids) != 2 or set(expected_gpu_uuids) != set(uuids):
            raise RuntimeError('The two selected physical GPU UUIDs differ from inventory.')
        boards = [next(b for b in boards if b['uuid'] == u) for u in expected_gpu_uuids]
        uuids = list(expected_gpu_uuids)
    if any(int(b['driver_version'].split('.')[0]) < 525 for b in boards):
        raise RuntimeError('Both T4s require an admitted CUDA12 driver.')
    platform = dict(provider='kaggle', adapter=FORMAT, gpu_uuids=uuids,
        physical_gpu_inventory=boards, cuda_visible_devices=','.join(uuids),
        runtime_stack=inventory['runtime_stack'], python_version=inventory['python_version'],
        cuda_plugin='cuda12', device_count=2)
    return dict(inventory=inventory, platform=platform, status='dual_host_inventory_passed')


def compiled_memory(executable):
    analysis = executable.memory_analysis()
    names = ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')
    if analysis is None or any(getattr(analysis, n, None) is None for n in names):
        raise RuntimeError('Per-partition compiler memory accounting is required.')
    values = {name: int(getattr(analysis, name)) for name in names}
    if any(v < 0 for v in values.values()):
        raise RuntimeError('Negative compiler memory accounting.')
    values['total_bytes'] = max(0, values['argument_size_in_bytes'] + values['output_size_in_bytes']
        + values['temp_size_in_bytes'] - values['alias_size_in_bytes'])
    return values


def _hlo_matrix_operations(text):
    """Resolve matrix operands in their HLO computation's SSA symbol table.

    JAX 0.4 printed operand types inline; JAX 0.11 prints only SSA names in
    instructions. Looking up the scoped definitions proves the same actual
    local dimensions on both versions, without accepting a result-shape guess.
    """
    scopes, operations = {}, []
    scope = None
    shape_pattern = r'(?:pred|bf16|f(?:8\w*|16|32|64)|[su](?:8|16|32|64)|c(?:64|128))\[[0-9,]*\]'
    for line in text.splitlines():
        stripped = line.strip()
        if not line.startswith((' ', '\t')) and stripped.endswith('{'):
            scope = stripped.split('(', 1)[0].strip()
            scopes[scope] = {}
            continue
        if stripped == '}':
            scope = None
            continue
        if scope is None:
            continue
        definition = re.match(r'(?:ROOT\s+)?(%[^\s=]+)\s*=\s*(.*)', stripped)
        if definition is None:
            continue
        name, expression = definition.groups()
        before_operation = re.split(r'\b(?:[A-Za-z][A-Za-z0-9_-]*)\(', expression, maxsplit=1)[0]
        shapes = re.findall(shape_pattern, before_operation)
        scopes[scope][name] = shapes
        matrix = re.search(r'\b(dot|custom-call)\((.*?)\)', expression)
        if matrix is None or matrix.group(1) == 'custom-call' and not re.search(r'custom_call_target="[^"]*(?:gemm|matmul)', expression, re.I):
            continue
        names = re.findall(r'%[^\s,()]+', matrix.group(2))
        operations.append(dict(scope=scope, instruction=stripped, output_shapes=shapes, operand_names=names))
    for operation in operations:
        definitions = scopes[operation['scope']]
        operands = []
        for name in operation['operand_names']:
            if name not in definitions or not definitions[name]:
                raise RuntimeError('Compiler matrix operand shape cannot be resolved in its computation: '+name)
            operands.extend(definitions[name])
        if not operands:
            raise RuntimeError('Compiler matrix operation has no resolved operand shapes.')
        operation['operand_shapes'] = operands
    return operations


class UtilizationMonitor:
    """Bounded read-only nvidia-smi sampling during ACTUAL training calls."""
    def __init__(self, gpu_uuids, interval=0.25, maximum_samples=240):
        self.uuids = tuple(gpu_uuids)
        self.interval = max(float(interval), .1)
        self.maximum_samples = int(maximum_samples)
        self.samples = []
        self.errors = []
        self.unsupported = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        command = ['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']
        while not self.stop_event.is_set() and len(self.samples) < self.maximum_samples:
            try:
                query = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3)
                rows = list(csv.reader(query.stdout.strip().splitlines()))
                if query.returncode:
                    raise RuntimeError('nvidia-smi utilization query failed')
                selected = {}
                for row in rows:
                    if len(row) == 3 and row[0].strip() in self.uuids:
                        uuid, util, memory = [v.strip() for v in row]
                        if util.lower().strip('[]') in ('n/a', 'not supported', 'not_supported'):
                            record = dict(gpu_uuid=uuid, raw_utilization=util)
                            if record not in self.unsupported:
                                self.unsupported.append(record)
                            selected[uuid] = dict(utilization_gpu_percent=None, memory_used_mib=int(memory))
                            continue
                        selected[uuid] = dict(utilization_gpu_percent=int(util), memory_used_mib=int(memory))
                if set(selected) != set(self.uuids):
                    raise RuntimeError('Both selected UUID utilization rows required')
                if any(v['utilization_gpu_percent'] is not None and not 0 <= v['utilization_gpu_percent'] <= 100
                       or v['memory_used_mib'] < 0 for v in selected.values()):
                    raise RuntimeError('Invalid GPU utilization telemetry')
                self.samples.append(dict(monotonic_ns=time.monotonic_ns(), devices=selected))
                if all(v['utilization_gpu_percent'] is None for v in selected.values()):
                    break
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))
                break
            self.stop_event.wait(self.interval)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('A utilization monitor is single-use.')
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                raise RuntimeError('Utilization sampler did not terminate.')
        maximum = {u: max((s['devices'][u]['utilization_gpu_percent'] for s in self.samples
            if s['devices'][u]['utilization_gpu_percent'] is not None), default=0)
            for u in self.uuids}
        return dict(samples=self.samples, errors=self.errors, maximum_utilization_percent=maximum,
            available=bool(self.samples), positive_on_both=all(v > 0 for v in maximum.values()),
            status=('error' if self.errors else 'unsupported' if self.unsupported else 'observed' if self.samples else 'unobserved'),
            unsupported=self.unsupported,
            scope='Telemetry sampled only while the actual admitted collect/optimize calls executed.')


class DualDeviceRuntime:
    def __init__(self, jax, np, devices=None, *, production=True, platform=None):
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        self.jax, self.np = jax, np
        self.devices = tuple(jax.local_devices() if devices is None else devices)
        self.production = bool(production)
        self.platform = platform
        if len(self.devices) != 2 or len(set(self.devices)) != 2:
            raise RuntimeError('Exactly two distinct local devices required.')
        if len(jax.devices()) != 2 or jax.process_count() != 1:
            raise RuntimeError('Exactly one process and two global/local devices required.')
        if self.production:
            if platform is None or platform.get('device_count') != 2:
                raise RuntimeError('Physical dual-T4 platform identity is required.')
            if any(d.platform != 'gpu' or 'T4' not in d.device_kind for d in self.devices):
                raise RuntimeError('Both JAX devices must be physical T4 GPUs.')
            if os.environ.get('CUDA_VISIBLE_DEVICES') != platform['cuda_visible_devices']:
                raise RuntimeError('Dual physical UUID visibility differs.')
        elif any(d.platform != 'cpu' for d in self.devices):
            raise RuntimeError('Nonproduction testing is restricted to forced CPU devices.')
        self.mesh = Mesh(np.asarray(self.devices, dtype=object), ('data',))
        self.replicated = NamedSharding(self.mesh, PartitionSpec())
        self.partitioned = NamedSharding(self.mesh, PartitionSpec('data'))
        self.handles = None
        self.profile = None
        self.state_shardings = None
        self.batch_shardings = None
        self._platform_digest = _hash(platform)

    def identity_guard(self):
        if tuple(self.jax.local_devices()) != self.devices or tuple(self.jax.devices()) != self.devices:
            raise RuntimeError('Device assignment changed after compilation.')
        if self.jax.process_count() != 1 or _hash(self.platform) != self._platform_digest:
            raise RuntimeError('Process topology or physical platform identity changed.')
        if self.production and os.environ.get('CUDA_VISIBLE_DEVICES') != self.platform['cuda_visible_devices']:
            raise RuntimeError('CUDA UUID visibility changed.')

    def constrain_batch(self, batch):
        """Install as objective hook BEFORE make_train_step; partition actual loss inputs."""
        def constrain(a):
            if a.ndim < 1 or a.shape[0] % 2:
                raise ValueError('Every minibatch leaf needs an even sample axis.')
            return self.jax.lax.with_sharding_constraint(a, self.partitioned)
        return self.jax.tree_util.tree_map(constrain, batch)

    def _state_sharding(self, state, n_env):
        if set(state._fields) != set(ENV_FIELDS + REPLICATED_FIELDS):
            raise RuntimeError('Frozen complete TrainingState fields changed.')
        result = self.jax.tree_util.tree_map(lambda _: self.replicated, state)
        def env_sharding(a):
            if a.ndim < 1 or a.shape[0] != n_env or n_env % 2:
                raise RuntimeError('Frozen environment axis must split evenly without resizing.')
            return self.partitioned
        return result._replace(**{name: self.jax.tree_util.tree_map(env_sharding, getattr(state, name))
            for name in ENV_FIELDS})

    def tree_bytes(self, tree):
        return sum(math.prod(a.shape) * self.np.dtype(a.dtype).itemsize for a in self.jax.tree_util.tree_leaves(tree))

    def state_identity(self, state):
        """All native leaves, not only params: includes RNG, Adam and every match."""
        leaves, treedef = self.jax.tree_util.tree_flatten(state)
        hashes = []
        for leaf in leaves:
            a = self.np.asarray(self.jax.device_get(leaf))
            hashes.append(dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest()))
        return dict(leaf_count=len(leaves), tree=str(treedef), leaf_digest=_hash(hashes), leaves=hashes)

    def place_state(self, state):
        self.identity_guard()
        if self.state_shardings is None:
            raise RuntimeError('Compile/validate shapes before state placement.')
        before = self.state_identity(state)
        placed = self.jax.device_put(state, self.state_shardings)
        self.synchronize(placed)
        after = self.state_identity(placed)
        if before != after:
            raise RuntimeError('State changed during dual-device placement.')
        self.assert_state_layout(placed)
        return placed, dict(before=before, after=after, exact_equal=True,
            optimizer_reset=False, rng_reset=False, environment_reset=False, snapshots_reset=False)

    def synchronize(self, tree):
        """Block EACH addressable shard before one joint wall-clock charge ends."""
        for leaf in self.jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()
            for shard in getattr(leaf, 'addressable_shards', ()):
                shard.data.block_until_ready()

    def _layout_rows(self, array):
        shards = tuple(getattr(array, 'addressable_shards', ()))
        if len(shards) != 2 or {s.device for s in shards} != set(self.devices):
            raise RuntimeError('Every global state/batch leaf must address both assigned devices.')
        return [dict(device_id=int(s.device.id), platform=s.device.platform,
            index=str(s.index), local_shape=list(s.data.shape), replica_id=int(s.replica_id)) for s in shards]

    def assert_state_layout(self, state, *, check_replicas=False):
        rows = []
        for name in state._fields:
            for leaf in self.jax.tree_util.tree_leaves(getattr(state, name)):
                local = self._layout_rows(leaf)
                if name in ENV_FIELDS:
                    if any(r['local_shape'][0] * 2 != leaf.shape[0] for r in local):
                        raise RuntimeError('Environment samples are not disjoint half-batches.')
                    if local[0]['index'] == local[1]['index']:
                        raise RuntimeError('Environment samples were duplicated on the devices.')
                else:
                    if any(tuple(r['local_shape']) != leaf.shape for r in local):
                        raise RuntimeError('Global optimizer/key/bank/counter must be replicated.')
                    if check_replicas:
                        values = [self.np.asarray(self.jax.device_get(s.data)) for s in leaf.addressable_shards]
                        if not self.np.array_equal(values[0], values[1], equal_nan=False):
                            raise RuntimeError('Replicated global learner state diverged.')
                rows.append(dict(field=name, global_shape=list(leaf.shape), shards=local))
        return dict(leaf_count=len(rows), all_addressable_on_both=True,
            disjoint_environment_partitions=True, replicas_equal=bool(check_replicas), leaves=rows)

    def assert_batch_layout(self, batch):
        rows = []
        for leaf in self.jax.tree_util.tree_leaves(batch):
            local = self._layout_rows(leaf)
            if leaf.ndim < 1 or any(r['local_shape'][0] * 2 != leaf.shape[0] for r in local):
                raise RuntimeError('Training samples are not split equally across devices.')
            if local[0]['index'] == local[1]['index']:
                raise RuntimeError('Duplicated full training batches are forbidden.')
            rows.append(dict(global_shape=list(leaf.shape), shards=local))
        return dict(all_addressable_on_both=True, disjoint_sample_partitions=True, leaves=rows)

    def compile(self, trainer, restored_state, cfg, *, memory_guard=None, phase_guard=None):
        """AOT compile from restored SHAPES, never initialize/reseed a learner."""
        self.identity_guard()
        if self.handles is not None:
            raise RuntimeError('This dual-device runtime compiles exactly once.')
        guard = (lambda label: None) if phase_guard is None else phase_guard
        if self.production and memory_guard is None:
            raise RuntimeError('Production requires per-device memory admissions.')
        admit = (lambda estimate, label: None) if memory_guard is None else memory_guard
        if any(type(cfg[k]) is not int or cfg[k] <= 0 for k in ('n_env', 'rollout_len', 'minibatch_size')):
            raise RuntimeError('Positive frozen integer training shapes required.')
        if cfg['n_env'] % 2 or cfg['minibatch_size'] % 2:
            raise RuntimeError('Global environment/minibatch sizes must divide over two devices.')
        self.state_shardings = self._state_sharding(restored_state, cfg['n_env'])
        abstract = self.jax.tree_util.tree_map(lambda a: self.jax.ShapeDtypeStruct(a.shape, a.dtype), restored_state)
        guard('dual abstract collect')
        after_collect, batch, rollout_metrics = self.jax.eval_shape(trainer.collect, abstract)
        self.batch_shardings = self.jax.tree_util.tree_map(lambda _: self.partitioned, batch)
        metric_shardings = self.jax.tree_util.tree_map(lambda _: self.replicated, rollout_metrics)
        collect = self.jax.jit(lambda state: trainer.collect(state), in_shardings=(self.state_shardings,),
            out_shardings=(self.state_shardings, self.batch_shardings, metric_shardings))
        guard('dual collect lowering')
        admit(self.tree_bytes(abstract), 'dual collect compilation')
        collect_handle = collect.lower(abstract).compile()
        collect_memory = compiled_memory(collect_handle)
        guard('dual optimizer abstract')
        after_optimize, update_metrics = self.jax.eval_shape(trainer.optimize, after_collect, batch)
        opt_metric_shardings = self.jax.tree_util.tree_map(lambda _: self.replicated, update_metrics)
        optimize = self.jax.jit(lambda state, batch: trainer.optimize(state, batch),
            in_shardings=(self.state_shardings, self.batch_shardings),
            out_shardings=(self.state_shardings, opt_metric_shardings))
        admit(collect_memory['total_bytes'] + self.tree_bytes(abstract), 'dual optimizer compilation')
        guard('dual optimizer lowering')
        optimize_handle = optimize.lower(after_collect, batch).compile()
        optimize_memory = compiled_memory(optimize_handle)
        # Compiler analysis is per executable partition. Retain one FULL global
        # rollback state as a conservative per-board upper bound, no overlap credit.
        estimate = max(collect_memory['total_bytes'], optimize_memory['total_bytes']) + self.tree_bytes(abstract) + 65536
        admit(estimate, 'dual actual training kernels')
        self.handles = dict(collect=collect_handle, optimize=optimize_handle)
        self.profile = dict(format=FORMAT, global_device_count=2, local_device_count=2,
            process_count=1, mesh_axis='data', num_partitions=2,
            global_n_env=cfg['n_env'], per_device_n_env=cfg['n_env']//2,
            global_minibatch_size=cfg['minibatch_size'], per_device_minibatch_size=cfg['minibatch_size']//2,
            global_rollout_samples=cfg['n_env']*cfg['rollout_len'],
            configuration_sha256=_hash(cfg),
            state_bytes=self.tree_bytes(abstract), batch_bytes=self.tree_bytes(batch),
            collect=collect_memory, optimize=optimize_memory, estimated_peak_bytes_per_device=estimate,
            retained_rollback_states=1, donations=False, global_rng_stream=True,
            global_gradient_norm=True, global_optimizer_state=True,
            hlopartition=self.hlo_evidence(cfg))
        guard('dual kernels compiled')
        return dict(handles=self.handles, profile=self.profile,
            after_collect=after_collect, batch=batch, after_optimize=after_optimize)

    def hlo_evidence(self, cfg):
        results = {}
        for kind, executable in self.handles.items():
            text = executable.as_text()
            if not isinstance(text, str) or not re.search(r'num_partitions\s*=\s*2\b', text):
                raise RuntimeError('Compiler must prove two actual SPMD partitions: ' + kind)
            lines = text.splitlines()
            collective = [line.strip() for line in lines if re.search(r'\ball-reduce(?:-start|-done)?\(', line)]
            matrix_operations = _hlo_matrix_operations(text)
            dot = [row['instruction'] for row in matrix_operations]
            backward = [line for line in dot if re.search(r'transpose\(jvp|backward|grad', line, re.I)]
            gradient_collectives = [line for line in collective if 'transpose(jvp(ActorCritic))' in line]
            half_samples = cfg['n_env']//2 if kind == 'collect' else cfg['minibatch_size']//2
            local_inputs = []
            local_backwards = []
            for tower, features in (('actor_network', 753), ('critic_network', 1173)):
                shape = re.compile(r'\[(?:'+str(half_samples)+','+str(features)+'|'+str(features)+','+str(half_samples)+r')\]')
                forward = [row for row in matrix_operations if tower+'/input/' in row['instruction']
                    and 'transpose(jvp' not in row['instruction']
                    and any(shape.search(s) for s in row['operand_shapes'])]
                if not forward:
                    raise RuntimeError('Compiler did not partition '+kind+' '+tower+' input samples in half.')
                local_inputs.extend(forward)
                if kind == 'optimize':
                    back = [row for row in matrix_operations if tower+'/input/' in row['instruction']
                        and row['instruction'] in backward and any(shape.search(s) for s in row['operand_shapes'])]
                    if not back:
                        raise RuntimeError('Compiler did not partition '+tower+' backward input samples in half.')
                    local_backwards.extend(back)
            if not dot:
                raise RuntimeError('No actual model matrix operations in compiled ' + kind)
            if kind == 'optimize' and (not gradient_collectives or not backward):
                raise RuntimeError('Optimizer must contain backward matrix work and collective reductions.')
            assigned = tuple(executable.runtime_executable().local_devices())
            if set(assigned) != set(self.devices) or len(assigned) != 2:
                raise RuntimeError('Compiled executable is not assigned to both selected physical devices.')
            results[kind] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
                num_partitions=2, matrix_operation_count=len(dot), backward_matrix_operation_count=len(backward),
                all_reduce_count=len(collective), matrix_examples=dot[:8], backward_examples=backward[:8],
                collective_examples=collective[:8], gradient_all_reduce_count=len(gradient_collectives),
                gradient_collective_examples=gradient_collectives[:4],
                local_forward_samples=half_samples, partitioned_actor_and_critic_forward=True,
                partitioned_actor_and_critic_backward=(kind == 'optimize'),
                partitioned_input_examples=local_inputs, partitioned_backward_examples=local_backwards,
                executable_device_ids=[int(d.id) for d in assigned])
        return results

    def memory_snapshots(self, ns, legacy_runtime):
        self.identity_guard()
        if not self.production:
            raise RuntimeError('Physical memory evidence is unavailable on CPU fixtures.')
        command = ['nvidia-smi', '--query-gpu=uuid,index,pci.bus_id,name,memory.total,memory.used,memory.free,driver_version',
            '--format=csv,noheader,nounits']
        query = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
        if query.returncode:
            raise RuntimeError('Physical two-board memory query failed.')
        rows = legacy_runtime._aot_kaggle_board_rows(query.stdout)
        indexed = {b['uuid']: b for b in rows}
        output = []
        for device, expected in zip(self.devices, self.platform['physical_gpu_inventory']):
            current = indexed.get(expected['uuid'])
            if current is None or any(current[k] != expected[k] for k in ('uuid', 'pci_bus_id', 'name', 'total_bytes', 'driver_version')):
                raise RuntimeError('Selected physical board identity changed.')
            snap = dict(platform='gpu', device_id=int(device.id), selected_gpu_uuid=current['uuid'],
                gpu_uuid=current['uuid'], gpu_index=current['index'], gpu_pci_bus_id=current['pci_bus_id'],
                gpu_name=current['name'], board_total_bytes=current['total_bytes'],
                board_used_bytes=current['used_bytes'], board_free_bytes=current['free_bytes'])
            snap.update(ns['_gpu_snapshot_allocator_fields'](device.memory_stats()))
            output.append(snap)
        return output

    def memory_admission(self, ns, legacy_runtime, options, estimate, label):
        snapshots = self.memory_snapshots(ns, legacy_runtime)
        records = []
        for snap in snapshots:
            record = ns['_gpu_memory_admission'](estimate, options, label, snapshot=snap)
            details = record.get('budget_details', {})
            if (record.get('status') != 'admitted' or record.get('overlap_credit_bytes') != 0
                    or details.get('pool_counter_source') != 'pool_bytes'
                    or details.get('pool_counter_status') != 'validated_current_total'):
                raise RuntimeError('Each physical T4 needs current pool/board memory admission.')
            records.append(record)
        return dict(format=FORMAT, stage=label, both_devices_admitted=True,
            estimated_peak_bytes_per_device=int(estimate), devices=records)

    def install_memory_adapters(self, ns, legacy_runtime):
        """Adapt frozen namespace memory hooks before native restore.

        The source checkpoint loader retains its original byte estimates and
        calls this namespace's admission hook. Calls without a supplied snapshot
        are admitted independently on BOTH boards. Supplying a snapshot is only
        the internal pure per-board path, retaining all original pool guards.
        """
        original_admit = ns['_gpu_memory_admission']
        if getattr(original_admit, '_dual_adapter', False):
            raise RuntimeError('Memory adapter must be installed exactly once.')
        def selected_identity():
            self.identity_guard()
            return (self.platform['cuda_visible_devices'],
                tuple(b['pci_bus_id'] for b in self.platform['physical_gpu_inventory']),
                tuple(b['name'] for b in self.platform['physical_gpu_inventory']),
                tuple(b['total_bytes'] for b in self.platform['physical_gpu_inventory']),
                tuple(b['driver_version'] for b in self.platform['physical_gpu_inventory']),
                os.environ.get('CUDA_VISIBLE_DEVICES'))
        def snapshot(device=None):
            snapshots = self.memory_snapshots(ns, legacy_runtime)
            index = 0 if device is None else self.devices.index(device)
            result = dict(snapshots[index])
            result['dual_gpu_snapshots'] = snapshots
            return result
        def admit(estimate, options, stage, *, snapshot=None):
            if snapshot is not None:
                return original_admit(estimate, options, stage, snapshot=snapshot)
            combined = self.memory_admission(ns, legacy_runtime, options, estimate, stage)
            # Preserve frozen caller's first-board schema and expose both proofs.
            result = dict(combined['devices'][0])
            result['dual_gpu_admission'] = combined
            return result
        admit._dual_adapter = True
        ns['_gpu_selected_identity'] = selected_identity
        ns['_gpu_memory_snapshot'] = snapshot
        ns['_gpu_memory_admission'] = admit
        return dict(format=FORMAT, both_physical_devices_guarded=True,
            no_overlap_credit=True, checkpoint_loader_unchanged=True)

    def admission_evidence(self, before_state, collected, optimized, *, utilization=None, allow_utilization_pending=False):
        """After one charged real rollout/update, before continuing full training."""
        self.synchronize((collected, optimized))
        collected_state, batch, collect_metrics = collected
        candidate, opt_metrics = optimized
        self.assert_state_layout(before_state, check_replicas=True)
        self.assert_state_layout(collected_state, check_replicas=True)
        candidate_layout = self.assert_state_layout(candidate, check_replicas=True)
        batch_layout = self.assert_batch_layout(batch)
        if int(candidate.updates) != int(before_state.updates) + 1:
            raise RuntimeError('Admission must perform exactly one global update.')
        if int(collected_state.updates) != int(before_state.updates):
            raise RuntimeError('Collection must preserve the update counter.')
        if int(collect_metrics['illegal_actions']) != 0:
            raise RuntimeError('Illegal actual rollout action; dual admission failed.')
        for name in ('finite', 'diagnostics_finite', 'gradients_finite', 'grad_norm_finite', 'params_finite', 'optimizer_finite'):
            if not bool(opt_metrics[name]):
                raise RuntimeError('Nonfinite actual dual optimizer: ' + name)
        if int(opt_metrics['accepted_updates']) <= 0 or int(opt_metrics['rejected_updates']) != 0:
            raise RuntimeError('Actual dual admission needs accepted finite minibatch work.')
        utilization_ok = _utilization_satisfied(utilization)
        if self.production and not utilization_ok and not allow_utilization_pending:
            raise RuntimeError('Actual training needs both positive utilizations or explicit unsupported telemetry.')
        status = 'passed' if not self.production or utilization_ok else 'pending_utilization'
        return dict(format=FORMAT, status=status, scope='One real charged collect/optimizer update in the requested run',
            global_device_count=2, local_device_count=2, process_count=1,
            device_kinds=[d.device_kind for d in self.devices], platform='gpu' if self.production else 'cpu_fixture',
            gpu_uuids=[] if not self.production else list(self.platform['gpu_uuids']),
            configuration_sha256=self.profile['configuration_sha256'],
            profile=self.profile, batch_layout=batch_layout,
            candidate_state_layout=candidate_layout, global_update_before=int(before_state.updates),
            global_update_after=int(candidate.updates), accepted_minibatches=int(opt_metrics['accepted_updates']),
            rejected_minibatches=int(opt_metrics['rejected_updates']),
            utilization=utilization, optimizer_replicas_equal=True, rng_replicas_equal=True,
            independent_trainings=False, synchronized_device_ids=[int(d.id) for d in self.devices],
            admission_window_updates=1,
            active_clock_basis='one joint synchronized wall duration, never device sum')


def _utilization_satisfied(utilization):
    if not isinstance(utilization, dict) or utilization.get('errors'):
        return False
    maximum = utilization.get('maximum_utilization_percent', {})
    unsupported = {r['gpu_uuid'] for r in utilization.get('unsupported', [])}
    return (utilization.get('status') in ('observed', 'unsupported') and len(maximum) == 2
        and unsupported <= set(maximum)
        and all(u in unsupported or v > 0 for u, v in maximum.items()))


def merge_utilization_reports(reports):
    if not reports or len(reports) > 3:
        raise ValueError('Exactly one through three actual-update utilization reports required.')
    samples = [s for r in reports for s in r.get('samples', [])]
    errors = [e for r in reports for e in r.get('errors', [])]
    unsupported = [s for r in reports for s in r.get('unsupported', [])]
    uuids = set(reports[0].get('maximum_utilization_percent', {}))
    if len(uuids) != 2 or any(set(r.get('maximum_utilization_percent', {})) != uuids for r in reports):
        raise ValueError('All utilization windows must identify the same two UUIDs.')
    maximum = {u: max(r['maximum_utilization_percent'][u] for r in reports) for u in uuids}
    return dict(samples=samples, errors=errors, unsupported=unsupported,
        maximum_utilization_percent=maximum, available=bool(samples),
        positive_on_both=all(v > 0 for v in maximum.values()),
        status=('error' if errors else 'unsupported' if unsupported else 'observed' if samples else 'unobserved'),
        actual_update_windows=len(reports), scope='Only the first bounded real committed training updates.')


def complete_admission_evidence(evidence, utilization, completed_updates):
    if type(completed_updates) is not int or not 1 <= completed_updates <= 3:
        raise ValueError('At most three actual committed updates may establish admission.')
    result = dict(evidence, utilization=utilization, admission_window_updates=completed_updates)
    ok = _utilization_satisfied(utilization)
    result['status'] = 'passed' if ok else 'pending_utilization'
    if ok:
        validate_execution_evidence(result)
    return result


def validate_execution_evidence(evidence, *, require_training=True):
    """Pure fail-closed check for the actual-update receipt exported by operator."""
    if evidence is None and require_training is False:
        return dict(status='absent', training_admitted=False)
    permitted_statuses = ('passed',) if require_training else ('passed', 'pending_utilization')
    if (not isinstance(evidence, dict) or evidence.get('format') != FORMAT
            or evidence.get('status') not in permitted_statuses or evidence.get('platform') != 'gpu'
            or evidence.get('global_device_count') != 2 or evidence.get('local_device_count') != 2
            or evidence.get('process_count') != 1 or evidence.get('independent_trainings') is not False
            or evidence.get('optimizer_replicas_equal') is not True or evidence.get('rng_replicas_equal') is not True):
        raise ValueError('Actual coordinated two-T4 execution receipt required.')
    kinds, uuids = evidence.get('device_kinds', []), evidence.get('gpu_uuids', [])
    if len(kinds) != 2 or any('T4' not in k for k in kinds) or len(uuids) != 2 or len(set(uuids)) != 2:
        raise ValueError('Two distinct physical T4 identities required.')
    sync = evidence.get('synchronized_device_ids', [])
    if len(sync) != 2 or len(set(sync)) != 2:
        raise ValueError('Both assigned device IDs must have synchronized.')
    if (evidence.get('global_update_after') != evidence.get('global_update_before', -2) + 1
            or evidence.get('accepted_minibatches', 0) <= 0 or evidence.get('rejected_minibatches') != 0):
        raise ValueError('Exactly one accepted global training update required.')
    window = evidence.get('admission_window_updates')
    if type(window) is not int or not 1 <= window <= 3:
        raise ValueError('Admission is bounded to the first three committed updates.')
    batch = evidence.get('batch_layout', {})
    state = evidence.get('candidate_state_layout', {})
    if (batch.get('all_addressable_on_both') is not True or batch.get('disjoint_sample_partitions') is not True
            or state.get('all_addressable_on_both') is not True or state.get('replicas_equal') is not True):
        raise ValueError('Actual partition and synchronized replica proof required.')
    for row in batch.get('leaves', []):
        shards = row.get('shards', [])
        if (len(shards) != 2 or {s['device_id'] for s in shards} != set(sync)
                or shards[0]['index'] == shards[1]['index']
                or any(s['local_shape'][0]*2 != row['global_shape'][0] for s in shards)):
            raise ValueError('Each actual batch leaf must be partitioned on both assigned devices.')
    if not batch.get('leaves'):
        raise ValueError('Actual batch shard evidence is empty.')
    hlo = evidence.get('profile', {}).get('hlopartition', {})
    digest = evidence.get('configuration_sha256')
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None or digest != evidence.get('profile', {}).get('configuration_sha256'):
        raise ValueError('Actual profile/evidence must bind the same canonical training configuration.')
    if any(hlo.get(k, {}).get('num_partitions') != 2 or hlo.get(k, {}).get('matrix_operation_count', 0) <= 0
           for k in ('collect', 'optimize')):
        raise ValueError('Both compiled actual training kernels must partition model work.')
    if hlo['optimize'].get('all_reduce_count', 0) <= 0 or hlo['optimize'].get('backward_matrix_operation_count', 0) <= 0:
        raise ValueError('Backward work and global gradient collectives required.')
    for phase in ('collect', 'optimize'):
        h = hlo[phase]
        if h.get('partitioned_actor_and_critic_forward') is not True or set(h.get('executable_device_ids', [])) != set(sync):
            raise ValueError('Actual half-batch model forward work and executable device assignments required.')
    if hlo['optimize'].get('partitioned_actor_and_critic_backward') is not True or hlo['optimize'].get('gradient_all_reduce_count', 0) <= 0:
        raise ValueError('Actual half-batch model backward work and gradient collectives required.')
    if evidence.get('status') == 'pending_utilization' and require_training is False:
        return dict(status='pending_utilization', training_admitted=False, recovery_only=True)
    utilization = evidence.get('utilization', {})
    peaks = utilization.get('maximum_utilization_percent', {})
    if utilization.get('status') == 'unsupported':
        unsupported = utilization.get('unsupported', [])
        if (not unsupported or utilization.get('errors') or set(peaks) != set(uuids)
                or any(r.get('gpu_uuid') not in uuids or r.get('raw_utilization', '').lower().strip('[]')
                    not in ('n/a', 'not supported', 'not_supported') for r in unsupported)):
            raise ValueError('Unsupported telemetry must carry exact raw NVIDIA status and known UUIDs.')
        if not _utilization_satisfied(utilization):
            raise ValueError('Any device supporting utilization must still show positive actual training utilization.')
        return True
    if (utilization.get('status') != 'observed' or utilization.get('positive_on_both') is not True
            or utilization.get('available') is not True or set(peaks) != set(uuids)
            or any(peaks[u] <= 0 for u in uuids)):
        raise ValueError('Actual training utilization must be positive on each physical T4.')
    return True
