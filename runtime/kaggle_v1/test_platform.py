"""Independent host fixtures for the narrow Kaggle platform adapter.

No JAX import, GPU allocation, package installation or provider access. Device
and subprocess telemetry below are explicit synthetic data, never GPU evidence.
"""
import ast
import copy
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / 'runtime/kaggle_v1'
BASE = ROOT / 'notebooks/runtime_safety_v2.ipynb'
MIGRATED = ROOT / 'notebooks/runtime_kaggle_v1.ipynb'
UUID = 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
OTHER_UUID = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
PCI = '00000000:0B:00.0'
MIB = 2 ** 20


def load_operator():
    spec = importlib.util.spec_from_file_location('kaggle_peer_operator', OWNER / 'operator.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def function_map(source):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        self.source = (OWNER / 'production_cell12.py').read_text()
        self.ns = {'__name__': 'host_platform_fixture'}
        exec(compile(self.source, '<host_adapter>', 'exec'), self.ns)
        self.identity = dict(selected_gpu_uuid=UUID, selected_gpu_pci_bus_id=PCI,
                             selected_gpu_name='Tesla T4', selected_gpu_total_bytes=15360*MIB,
                             driver_version='570.195.03')
        self.ns['_SPADES_KAGGLE_PLATFORM'] = self.identity
        self.device = SimpleNamespace(platform='gpu', device_kind='Tesla T4', id=0,
            local_hardware_id=0, memory_stats=lambda: {
                'bytes_in_use': 512*MIB, 'pool_bytes': 1024*MIB,
                'bytes_limit': 15360*MIB})
        self.ns['jax'] = SimpleNamespace(devices=lambda: [self.device])
        self.env = self.enterContext(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': UUID}, clear=True))
        self.row = [UUID, '1', PCI, 'Tesla T4', '15360', '1024', '14336', '570.195.03']

    def query(self, row=None, *, code=0, rows=1):
        row = self.row if row is None else row
        output = ', '.join(row)
        return SimpleNamespace(returncode=code, stdout='\n'.join([output]*rows)+'\n', stderr='')

    def snapshot(self, row=None, **kwargs):
        with patch.object(self.ns['subprocess'], 'run', return_value=self.query(row, **kwargs)) as call:
            result = self.ns['_gpu_memory_snapshot']()
            return result, call.call_args.args[0]

    def assert_rejected(self, snapshot):
        # The unchanged production helper retains diagnostic fallback arithmetic.
        # The operator additionally requires a current raw pool counter before
        # any real workload, so exercise that complete admission boundary.
        op = load_operator()
        session = dict(ns=self.ns, options=self.ns['GPU_TUNING'],
                       profile=dict(collect=dict(total_bytes=1), state_bytes=0))
        with patch.object(op, '_aot_boundary'), patch.dict(self.ns, {'_gpu_memory_snapshot':lambda:snapshot}):
            with self.assertRaises(RuntimeError):
                op._aot_admit(session, 'collect', ())

    def test_physical_one_logical_zero_queries_full_uuid(self):
        result, argv = self.snapshot()
        self.assertEqual(result['gpu_index'], 1)
        self.assertEqual(result['gpu_uuid'], UUID)
        self.assertIn('--id='+UUID, argv)
        self.assertFalse(any(value in ('--id=0', '--id=1') for value in argv))
        self.assertEqual(result['board_total_bytes'], 15360*MIB)
        self.assertEqual(result['board_free_bytes'], 14336*MIB)

    def test_other_uuid_pci_name_driver_capacity_are_rejected(self):
        for index, value in ((0, OTHER_UUID), (2, '00000000:0C:00.0'),
                             (3, 'Tesla P100'), (4, '32768'), (7, '580.1')):
            with self.subTest(field=index):
                row = self.row.copy(); row[index] = value
                snap, _ = self.snapshot(row)
                self.assertIn('selected_board_telemetry_error', snap)
                self.assertNotIn('board_total_bytes', snap)
                self.assert_rejected(snap)

    def test_ambiguous_failed_and_empty_board_queries_reject(self):
        for kwargs in ({'rows':2}, {'rows':0}, {'code':1}):
            with self.subTest(kwargs=kwargs):
                snap, _ = self.snapshot(**kwargs)
                self.assert_rejected(snap)

    def test_nonfinite_fractional_negative_or_inconsistent_board_counters_reject(self):
        for index, value in ((4, 'nan'), (5, 'inf'), (6, '-1'), (4, '15360.25'),
                             (5, '15361'), (6, '15361'), (6, '15000'), (1, '-1')):
            with self.subTest(field=index, value=value):
                row = self.row.copy(); row[index] = value
                snap, _ = self.snapshot(row)
                self.assert_rejected(snap)

    def test_unavailable_and_timed_out_nvidia_query_reject(self):
        for error in (OSError('synthetic unavailable'), subprocess.TimeoutExpired('nvidia-smi', 5)):
            with self.subTest(error=type(error).__name__):
                with patch.object(self.ns['subprocess'], 'run', side_effect=error):
                    snap = self.ns['_gpu_memory_snapshot']()
                self.assert_rejected(snap)

    def test_visibility_mutation_stops_before_query(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = OTHER_UUID
        with patch.object(self.ns['subprocess'], 'run') as query:
            with self.assertRaises(RuntimeError): self.ns['_gpu_memory_snapshot']()
            query.assert_not_called()

    def test_two_jax_visible_boards_stop_before_query(self):
        self.ns['jax'].devices = lambda: [self.device, self.device]
        with patch.object(self.ns['subprocess'], 'run') as query:
            with self.assertRaises(RuntimeError): self.ns['_gpu_memory_snapshot']()
            query.assert_not_called()

    def test_missing_pool_live_or_limit_cannot_admit(self):
        for key in ('pool_bytes', 'bytes_in_use', 'bytes_limit'):
            with self.subTest(key=key):
                stats = self.device.memory_stats(); del stats[key]
                self.device.memory_stats = lambda stats=stats: stats
                snap, _ = self.snapshot()
                self.assert_rejected(snap)
                self.device.memory_stats = lambda: {'bytes_in_use':512*MIB, 'pool_bytes':1024*MIB, 'bytes_limit':15360*MIB}

    def test_malformed_allocator_stats_cannot_admit(self):
        for stats in ({'bytes_in_use':True, 'pool_bytes':1024*MIB, 'bytes_limit':15360*MIB},
                      {'bytes_in_use':512*MIB, 'pool_bytes':float('nan'), 'bytes_limit':15360*MIB},
                      {'bytes_in_use':512*MIB, 'pool_bytes':1024*MIB, 'bytes_limit':'bad'}, 'not_dict'):
            with self.subTest(stats=stats):
                self.device.memory_stats = lambda stats=stats: stats
                snap, _ = self.snapshot()
                self.assert_rejected(snap)

    def test_no_training_cell_or_unrelated_production_function_changed(self):
        old = json.loads(BASE.read_text()); new = json.loads(MIGRATED.read_text())
        self.assertEqual(len(new['cells']), 38)
        for i in range(38):
            if i != 12: self.assertEqual(new['cells'][i], old['cells'][i], 'cell'+str(i))
        old_functions = function_map(''.join(old['cells'][12]['source']))
        new_functions = function_map(''.join(new['cells'][12]['source']))
        self.assertEqual(set(new_functions)-set(old_functions), {'_gpu_selected_identity'})
        self.assertEqual(set(old_functions)-set(new_functions), set())
        changed = {k for k in old_functions if old_functions[k] != new_functions[k]}
        self.assertEqual(changed, {'_gpu_memory_snapshot', '_gpu_runtime_identity'})

    def test_all_retained_memory_admission_math_is_ast_identical(self):
        old = json.loads(BASE.read_text()); a = function_map(''.join(old['cells'][12]['source'])); b = function_map(self.source)
        for name in a:
            if name not in ('_gpu_memory_snapshot', '_gpu_runtime_identity'):
                self.assertEqual(a[name], b[name], name)

    def test_host_ram_guard_refuses_before_device_materialization(self):
        op = load_operator()
        calls = []
        ns = dict(np=SimpleNamespace(), jax=SimpleNamespace(device_get=lambda x:calls.append(x),
                  tree_util=SimpleNamespace(tree_leaves=lambda x:calls.append(x))))
        with patch.object(op, '_aot_host_available_bytes', return_value=0):
            with self.assertRaises(MemoryError):
                op._aot_host_finite({'ns':ns}, object(), 'synthetic collection', 273, 1000)
        self.assertEqual(calls, [])


class InventoryChecks(unittest.TestCase):
    def setUp(self):
        self.op = load_operator()
        self.versions = {'jax':'0.11.1', 'jaxlib':'0.11.1', 'flax':'0.12.10', 'optax':'0.2.8',
                         'jax-cuda12-plugin':'0.11.1', 'jax-cuda12-pjrt':'0.11.1'}
        self.rows = [
            [OTHER_UUID, '0', '00000000:0A:00.0', 'Tesla T4', '15360', '1024', '14336', '570.195.03'],
            [UUID, '1', PCI, 'Tesla T4', '15360', '1024', '14336', '570.195.03']]
        def version(name):
            if name not in self.versions: raise importlib.metadata.PackageNotFoundError(name)
            return self.versions[name]
        self.enterContext(patch('importlib.metadata.version', version))
        self.enterContext(patch('importlib.metadata.distributions', return_value=[]))
        self.env = self.enterContext(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0,1'}, clear=True))
        self.query = self.enterContext(patch('subprocess.run', side_effect=lambda *args,**kw:
            SimpleNamespace(returncode=0, stdout='\n'.join(', '.join(row) for row in self.rows)+'\n', stderr='')))

    def test_two_physical_boards_inventory_selects_second_by_uuid_without_mutation(self):
        before = set(sys.modules)
        result = self.op.preflight_kaggle_nojax(UUID)
        self.assertEqual(result['platform']['selected_gpu_uuid'], UUID)
        self.assertEqual(result['platform']['selected_gpu_pci_bus_id'], PCI)
        self.assertEqual(len(result['physical_gpu_inventory']), 2)
        self.assertEqual(result['environment']['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertFalse(result['jax_imported'])
        self.assertFalse(result['compute_allocated'])
        self.assertFalse(any(k == 'jax' or k.startswith('jax.') for k in set(sys.modules)-before))
        self.assertNotIn('--id=0', self.query.call_args.args[0])

    def test_all_six_runtime_version_mismatches_reject_before_board_query(self):
        for key in self.versions:
            with self.subTest(package=key):
                self.query.reset_mock()
                original = self.versions[key]; self.versions[key] = '0.0.0'
                with self.assertRaises(RuntimeError): self.op.inventory_kaggle_nojax()
                self.query.assert_not_called(); self.versions[key] = original

    def test_any_additional_cuda13_plugin_or_pjrt_rejects(self):
        for key in ('jax-cuda13-plugin', 'jax-cuda13-pjrt'):
            with self.subTest(package=key):
                self.query.reset_mock(); self.versions[key] = '0.11.1'
                with self.assertRaises(RuntimeError): self.op.inventory_kaggle_nojax()
                self.query.assert_not_called(); del self.versions[key]

    def test_already_imported_jax_jaxlib_or_plugin_rejects_before_query(self):
        for key in ('jax', 'jaxlib', 'jax.foo', 'jaxlib.foo', 'jax_plugins.xla_cuda12'):
            with self.subTest(module=key):
                self.query.reset_mock()
                with patch.dict(sys.modules, {key:SimpleNamespace()}):
                    with self.assertRaises(RuntimeError): self.op.inventory_kaggle_nojax()
                self.query.assert_not_called()

    def test_constraint_skip_and_other_jax_visibility_reject(self):
        for key, value in (('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','0'), ('JAX_CUDA_VISIBLE_DEVICES','0')):
            with self.subTest(key=key):
                self.query.reset_mock()
                with patch.dict(os.environ, {key:value}):
                    with self.assertRaises(RuntimeError): self.op.inventory_kaggle_nojax()
                self.query.assert_not_called()

    def test_non_uuid_and_absent_uuid_have_no_fallback(self):
        for value in ('0', 'GPU-bbbb', 'all', '', None):
            with self.subTest(value=value):
                self.query.reset_mock()
                with self.assertRaises(ValueError): self.op.preflight_kaggle_nojax(value)
                self.query.assert_not_called()
        with self.assertRaises(RuntimeError): self.op.preflight_kaggle_nojax('GPU-cccccccc-cccc-cccc-cccc-cccccccccccc')

    def test_selected_p100_bad_driver_and_lookalike_t4_reject(self):
        for index, value in ((3,'Tesla P100'), (3,'Tesla T4x'), (7,'524.99')):
            with self.subTest(field=index, value=value):
                original = self.rows[1][index]; self.rows[1][index] = value
                with self.assertRaises(RuntimeError): self.op.preflight_kaggle_nojax(UUID)
                self.rows[1][index] = original

    def test_duplicate_uuid_or_pci_inventory_rejects(self):
        for index in (0,2):
            with self.subTest(field=index):
                original = self.rows[1][index]; self.rows[1][index] = self.rows[0][index]
                with self.assertRaises(ValueError): self.op.preflight_kaggle_nojax(UUID)
                self.rows[1][index] = original

    def test_definitions_only_and_two_explicit_compiles(self):
        tree = ast.parse((OWNER/'operator.py').read_text())
        for n in tree.body:
            self.assertTrue(isinstance(n,(ast.FunctionDef,ast.ClassDef)) or
                            isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant))
        prepare = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_aot_long_run')
        compiles = [n for n in ast.walk(prepare) if isinstance(n,ast.Call) and
                    isinstance(n.func,ast.Attribute) and n.func.attr=='compile']
        self.assertEqual(len(compiles),2)
        imports = [n for n in ast.walk(prepare) if isinstance(n,ast.Import) and any(a.name=='jax' for a in n.names)]
        env = [n for n in ast.walk(prepare) if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and t.value.attr=='environ'
                   and isinstance(t.slice,ast.Constant) and t.slice.value=='CUDA_VISIBLE_DEVICES' for t in n.targets)]
        self.assertEqual(len(imports),1); self.assertEqual(len(env),1)
        self.assertLess(env[0].lineno,imports[0].lineno)

    def test_learning_work_ack_and_segment_controls_ast_identical(self):
        old = function_map((ROOT/'runtime/colab_c3/operator.py').read_text())
        new = function_map((OWNER/'operator.py').read_text())
        names = ['_aot_validate_candidate_package','_aot_same_json','_aot_resident_credit',
                 '_aot_host_finite','_aot_launch','_aot_step','run_until_barrier',
                 'ack_export','_aot_verify_resume_grant','next_segment','_aot_limit_reason']
        for name in names: self.assertEqual(old[name],new[name],name)

    def coordination(self):
        result=json.loads((OWNER/'coordination_TEMPLATE.json').read_text())
        result.update(run_authorized=True,parent_review_complete=True,runtime_exclusive=True,
                      candidate_native_export_verified=True,selected_gpu_uuid=UUID,
                      authorization_valid_until_utc='2026-10-21T00:00:00Z')
        return result

    def test_changed_compiler_flags_or_allocator_settings_reject_before_file_access(self):
        for key,value in (('XLA_FLAGS',''),('CUBLAS_WORKSPACE_CONFIG',':4096:8'),
                          ('TF_DETERMINISTIC_OPS','1'),('JAX_DEFAULT_MATMUL_PRECISION','highest'),
                          ('XLA_PYTHON_CLIENT_PREALLOCATE','true'),('XLA_PYTHON_CLIENT_MEM_FRACTION','0.50')):
            with self.subTest(key=key):
                with patch.dict(os.environ,{key:value}),patch.object(self.op,'_aot_file_identity') as files:
                    with self.assertRaises(RuntimeError):
                        self.op.prepare_aot_long_run(original_path='/must-not-read',production_path='/must-not-read',
                            candidate_package_path='/must-not-read',coordination=self.coordination(),
                            output_dir='/must-not-create',recovery_sink={})
                    files.assert_not_called()

    def test_wrong_source_pin_or_false_authorization_rejects_before_inventory(self):
        for key,value in (('run_authorized',False),('max_segments',9),('max_segments',1.0),
                          ('supplemental_export_interval_seconds',900.0),('export_wait_timeout_seconds',3601)):
            with self.subTest(key=key,value=value):
                co=self.coordination();co[key]=value
                self.query.reset_mock()
                with self.assertRaises((RuntimeError,ValueError)):
                    self.op.prepare_aot_long_run(original_path='/must-not-read',production_path='/must-not-read',
                        candidate_package_path='/must-not-read',coordination=co,output_dir='/must-not-create',recovery_sink={})
                self.query.assert_not_called()
        co=self.coordination();co['source_pins']['production_sha256']='0'*64
        self.query.reset_mock()
        with self.assertRaises(RuntimeError):
            self.op.prepare_aot_long_run(original_path='/must-not-read',production_path='/must-not-read',
                candidate_package_path='/must-not-read',coordination=co,output_dir='/must-not-create',recovery_sink={})
        self.query.assert_not_called()


class LaunchIdentityChecks(unittest.TestCase):
    """Reuse the established synthetic state generator; assertions are new."""
    def build(self):
        spec = importlib.util.spec_from_file_location('c3_fixture_for_kaggle_peer', ROOT/'runtime/colab_c3/test_operator.py')
        fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        self.op = load_operator()
        self.enterContext(patch.object(fixture, 'op', self.op))
        platform = dict(provider='kaggle', adapter='spades-kaggle-aot-v1', selected_gpu_uuid=UUID,
            selected_gpu_pci_bus_id=PCI, selected_gpu_name='Tesla T4', selected_gpu_total_bytes=15360*MIB,
            driver_version='570.195.03', cuda_plugin='cuda12', python_version='3.12.0',
            runtime_stack={'jax':'0.11.1','jaxlib':'0.11.1','flax':'0.12.10','optax':'0.2.8',
                           'jax-cuda12-plugin':'0.11.1','jax-cuda12-pjrt':'0.11.1'})
        self.enterContext(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':UUID}, clear=True))
        actual_new = self.op._aot_new_session
        device = SimpleNamespace(platform='gpu',device_kind='Tesla T4')
        def adapt_fixture(**kwargs):
            kwargs['ns']['_SPADES_KAGGLE_PLATFORM'] = copy.deepcopy(platform)
            kwargs['ns']['jax'].devices = lambda:[device]
            kwargs['profile']['device'] = device
            return actual_new(**kwargs)
        self.enterContext(patch.object(self.op, '_aot_new_session', adapt_fixture))
        return fixture.OwnerChecks.build(self)

    def ack(self, session):
        receipt = {k:session['pending_export'][k] for k in
                   ('nonce','package_path','package_sha256','package_bytes','native_path','native_sha256','native_bytes')}
        receipt.update(verified=True,library_receipt='TEST_EXTERNAL_RECEIPT',valid_until_utc='2026-10-18T23:00:00Z')
        return self.op.ack_export(session,receipt)

    def test_new_kaggle_session_initial_ack_one_update_still_barriers(self):
        s, clock, counts, recovery, order = self.build()
        self.assertEqual(counts['collect'],0)
        self.assertEqual(self.op.run_until_barrier(s)['status'],'export_required')
        self.ack(s)
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertEqual(result['reason'],'validation'); self.assertFalse(result['terminal'])
        self.assertEqual(s['accepted_update'],675); self.assertEqual(order,['collect','optimize'])
        self.assertEqual(result['platform']['selected_gpu_uuid'],UUID)

    def test_selected_environment_mutation_after_ack_blocks_every_workload(self):
        s, clock, counts, recovery, order = self.build(); self.ack(s)
        os.environ['CUDA_VISIBLE_DEVICES']=OTHER_UUID
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertTrue(result['terminal']); self.assertEqual(result['status'],'resume_verification_failed')
        self.assertEqual(order,[]); self.assertEqual(s['accepted_update'],674)

    def test_platform_metadata_alias_mutation_after_ack_blocks_every_workload(self):
        s, clock, counts, recovery, order = self.build(); self.ack(s)
        s['platform']['driver_version']='580.1'
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertTrue(result['terminal']); self.assertEqual(order,[])
        self.assertEqual(s['accepted_update'],674)

    def test_namespace_platform_mutation_after_ack_blocks_every_workload(self):
        s, clock, counts, recovery, order = self.build(); self.ack(s)
        s['ns']['_SPADES_KAGGLE_PLATFORM']['runtime_stack']['optax']='0.0.0'
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertTrue(result['terminal']); self.assertEqual(order,[])
        self.assertEqual(s['accepted_update'],674)

    def test_second_jax_device_appearing_after_ack_blocks_every_workload(self):
        s, clock, counts, recovery, order = self.build(); self.ack(s)
        device=s['profile']['device']; s['ns']['jax'].devices=lambda:[device,device]
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertTrue(result['terminal']); self.assertEqual(order,[])
        self.assertEqual(s['accepted_update'],674)

    def test_visibility_change_during_collect_prevents_optimizer(self):
        s, clock, counts, recovery, order = self.build(); self.ack(s)
        actual=s['handles']['collect']
        def changed(state):
            result=actual(state); os.environ['CUDA_VISIBLE_DEVICES']=OTHER_UUID; return result
        s['handles']['collect']=changed; s['saved_handles']['collect']=changed
        result=self.op.run_until_barrier(s,max_updates=1)
        self.assertTrue(result['terminal']); self.assertEqual(order,['collect'])
        self.assertEqual(counts['optimize'],0); self.assertEqual(s['accepted_update'],674)


if __name__ == '__main__':
    unittest.main(verbosity=2)
