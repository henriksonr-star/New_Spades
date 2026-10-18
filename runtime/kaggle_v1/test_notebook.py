"""Host-only execution tests for fixed Kaggle migration data/control cells; never import JAX."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('kaggle_notebook_builder', ROOT / 'operator_notebook_builder.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FixedCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_jax = 'jax' in sys.modules
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'fixture.ipynb'
        builder.build_operator_notebook(payload_path='/kaggle/working/test-kaggle.zip',
            payload_sha256='a' * 64, payload_bytes=1234,
            operator_sha256='b' * 64, output_path=cls.path)
        cls.notebook = json.loads(cls.path.read_text())
        cls.cells = {c['id']: ''.join(c['source']) for c in cls.notebook['cells']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def mapped_path(self, raw):
        test_root = Path(self.temp.name)
        class Mapped:
            def __init__(self, value): self.value = PurePosixPath(value)
            def expanduser(self): return self
            def resolve(self): return self
            @property
            def parents(self): return tuple(Mapped(str(v)) for v in self.value.parents)
            def __eq__(self, other): return getattr(other, 'value', None) == self.value
            def __truediv__(self, other): return Mapped(str(self.value / other))
            def open(self, mode):
                relative = self.value.relative_to('/kaggle/working')
                actual = test_root / relative
                actual.parent.mkdir(parents=True, exist_ok=True)
                return actual.open(mode)
        return Mapped(raw)

    def namespace(self, data='', **extra):
        namespace = {'input': lambda _: data, 'print': lambda *args: None,
                     'parse_json_object': builder.parse_json_object, 'Path': self.mapped_path,
                     'KAGGLE_PREFLIGHT': {'platform': {'selected_gpu_uuid': 'GPU-test-uuid'}},
                     'json': json}
        namespace.update(extra)
        return namespace

    def execute(self, cell, namespace):
        exec(compile(self.cells[cell], cell, 'exec'), namespace)

    def test_pins_and_review_notice(self):
        meta = self.notebook['metadata']['spades_frozen_operator']
        self.assertEqual(meta['candidate_update'], 674)
        self.assertEqual(meta['candidate_env_steps'], 309198848)
        self.assertEqual(meta['candidate_package_sha256'], builder.CANDIDATE_PACKAGE_SHA256)
        text = self.path.read_text()
        self.assertIn('2026-10-19T15:49:14Z', text)
        self.assertNotIn('05:14:14', text)
        self.assertIn('not durably exported', text)

    def test_migration_metadata_does_not_claim_physical_test(self):
        meta = self.notebook['metadata']['spades_frozen_operator']
        self.assertEqual(meta['schema'], 'spades-aot-kaggle-payload-v1')
        self.assertEqual(meta['migration_authorized_utc'], '2026-10-18T22:33:33Z')
        self.assertEqual(meta['preparation_cap_utc'], '2026-10-19T15:49:14Z')
        self.assertIn('untested', meta['physical_qa'])
        text = self.path.read_text()
        self.assertIn('Do not use Run All', text)
        self.assertIn('spades-aot-external-export-kaggle-v1', text)
        self.assertNotIn('external manifest remains formatV2', text)

    def test_all_cells_parse_and_are_unexecuted(self):
        self.assertTrue(builder.validate_notebook_structure(self.notebook))
        self.assertEqual(len(self.notebook['cells']), 17)
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_builder_refuses_existing_output(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            builder.build_operator_notebook(payload_path='/kaggle/working/test-kaggle.zip',
                payload_sha256='a' * 64, payload_bytes=1234,
                operator_sha256='b' * 64, output_path=self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_definitions_reject_rebinding_permitted_builtin_class_bases(self):
        fixture = '''def prepare_aot_long_run(): pass
def run_until_barrier(): pass
def ack_export(): pass
def next_segment(): pass
'''
        for name in ('RuntimeError', 'Exception'):
            for definition in ('class ' + name + '(Exception):\n    pass\n',
                               'def ' + name + '(): pass\n'):
                with self.subTest(name=name, definition=definition):
                    with self.assertRaisesRegex(ValueError, 'built-in class bases'):
                        builder.operator_definitions_tree(fixture + definition)

    def test_no_network_installer_browser_or_drive_calls(self):
        calls = []
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node, ast.Call):
                        calls.append(ast.unparse(node.func))
        forbidden = ('mount', 'requests', 'urlopen', 'subprocess', 'system', 'sleep')
        self.assertFalse([call for call in calls if any(word in call for word in forbidden)])

    def test_prepare_blank_retains_recovery(self):
        recovery = {'retained': object()}
        ns = self.namespace(RECOVERY=recovery)
        self.execute('aot-prepare-data', ns)
        self.assertIs(ns['RECOVERY'], recovery)
        self.assertNotIn('AOT_SESSION', ns)

    def test_prepare_uses_package_api_once(self):
        calls = []
        returned = {'root': '/kaggle/working/test-output'}
        request = {'candidate_package_path': '/kaggle/input/private/P90.zip',
                   'output_dir': '/kaggle/working/test-output', 'coordination': {'reviewed': True, 'selected_gpu_uuid': 'GPU-test-uuid'}}
        ns = self.namespace(json.dumps(request), PAYLOAD={'paths': {
            'original.ipynb': Path('/original'), 'production.ipynb': Path('/production')}},
            prepare_aot_long_run=lambda **kwargs: calls.append(kwargs) or returned)
        self.execute('aot-prepare-data', ns)
        self.assertIs(ns['AOT_SESSION'], returned)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['candidate_package_path'], '/kaggle/input/private/P90.zip')
        self.assertNotIn('checkpoint_path', calls[0])
        self.assertIs(calls[0]['recovery_sink'], ns['RECOVERY'])

    def test_prepare_rejects_loose_native_or_extra_key(self):
        cases = [dict(checkpoint_path='/loose.zip', output_dir='/kaggle/working/test-output', coordination={'selected_gpu_uuid':'GPU-test-uuid'}),
                 dict(candidate_package_path='/P90.zip', output_dir='/kaggle/working/test-output', coordination={'selected_gpu_uuid':'GPU-test-uuid'}, retry=True)]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.execute('aot-prepare-data', self.namespace(json.dumps(case)))

    def test_prepare_rejects_existing_session_or_recovery(self):
        data = json.dumps(dict(candidate_package_path='/P90.zip', output_dir='/kaggle/working/test-output', coordination={'selected_gpu_uuid':'GPU-test-uuid'}))
        for extra in ({'AOT_SESSION': {}}, {'RECOVERY': {'accepted': True}}):
            with self.subTest(extra=extra):
                with self.assertRaises(RuntimeError):
                    self.execute('aot-prepare-data', self.namespace(data, **extra))

    def test_prepare_error_retains_recovery(self):
        def rejected(**kwargs):
            kwargs['recovery_sink']['accepted_state'] = 'retained'
            raise RuntimeError('fail fast')
        data = json.dumps(dict(candidate_package_path='/P90.zip', output_dir='/kaggle/working/test-output', coordination={'selected_gpu_uuid':'GPU-test-uuid'}))
        ns = self.namespace(data, PAYLOAD={'paths': {
            'original.ipynb': '/original', 'production.ipynb': '/production'}},
            prepare_aot_long_run=rejected)
        with self.assertRaisesRegex(RuntimeError, 'fail fast'):
            self.execute('aot-prepare-data', ns)
        self.assertEqual(ns['RECOVERY'], {'accepted_state': 'retained'})
        self.assertNotIn('AOT_SESSION', ns)

    def test_ack_only_calls_ack(self):
        calls = []
        session = object()
        receipt = {'nonce': 'actual', 'verified': True}
        ns = self.namespace(json.dumps(receipt), AOT_SESSION=session,
            ack_export=lambda *args: calls.append(args) or {'status': 'ready'})
        self.execute('aot-ack-data', ns)
        self.assertEqual(calls, [(session, receipt)])
        self.assertNotIn('run_until_barrier', ns)

    def test_ack_blank_retains_live_session(self):
        session = object()
        ns = self.namespace(AOT_SESSION=session)
        self.execute('aot-ack-data', ns)
        self.assertIs(ns['AOT_SESSION'], session)
        self.assertNotIn('ACK_STATUS', ns)

    def test_ack_rejects_absent_session_and_bad_json(self):
        with self.assertRaises(RuntimeError):
            self.execute('aot-ack-data', self.namespace('{"nonce":"actual"}'))
        with self.assertRaises(ValueError):
            self.execute('aot-ack-data', self.namespace('{"verified":true,"verified":false}', AOT_SESSION={}))

    def test_next_segment_confirmation_exact_and_no_training(self):
        calls = []
        session = object()
        ns = self.namespace('{"action":"next_segment"}', AOT_SESSION=session,
            next_segment=lambda *args: calls.append(args) or {'status': 'ready'})
        self.execute('aot-next-segment-data', ns)
        self.assertEqual(calls, [(session,)])
        self.assertNotIn('run_until_barrier', ns)

    def test_next_segment_rejects_false_or_extra_confirmation(self):
        for data in ('{}', '{"action":"run"}', '{"action":"next_segment","auto":true}',
                     '{"action":"next_segment","action":"next_segment"}'):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    self.execute('aot-next-segment-data', self.namespace(data, AOT_SESSION={}))

    def test_next_segment_blank_and_missing_session(self):
        ns = self.namespace(RECOVERY={'retained': True})
        self.execute('aot-next-segment-data', ns)
        self.assertNotIn('NEXT_SEGMENT_STATUS', ns)
        self.assertEqual(ns['RECOVERY'], {'retained': True})
        with self.assertRaises(RuntimeError):
            self.execute('aot-next-segment-data', self.namespace('{"action":"next_segment"}'))

    def test_control_cells_require_session(self):
        for cell in ('aot-smoke', 'aot-run'):
            with self.subTest(cell=cell):
                with self.assertRaises(RuntimeError):
                    self.execute(cell, self.namespace())

    def test_run_controls_pass_exact_bounds(self):
        calls = []
        ns = self.namespace(AOT_SESSION='same-live-session',
            run_until_barrier=lambda *args, **kwargs: calls.append((args, kwargs)) or {'status': 'ready'})
        self.execute('aot-smoke', ns)
        self.execute('aot-run', ns)
        self.assertEqual(calls, [(('same-live-session',), {'max_updates': 1}),
                                 (('same-live-session',), {'max_updates': None})])

    def test_preflight_inventory_printed_before_uuid_selection(self):
        events=[]
        ns=self.namespace('{"selected_gpu_uuid":"GPU-test-uuid"}',
            inventory_kaggle_nojax=lambda: events.append('inventory') or {'physical_gpu_inventory':[]},
            preflight_kaggle_nojax=lambda **kw: events.append(('select',kw)) or {'platform': kw})
        self.execute('aot-preflight-data',ns)
        self.assertEqual(events,['inventory',('select',{'selected_gpu_uuid':'GPU-test-uuid'})])
        self.assertEqual(ns['KAGGLE_PREFLIGHT'],{'platform':{'selected_gpu_uuid':'GPU-test-uuid'}})
        self.assertNotIn('AOT_SESSION',ns)

    def test_preflight_blank_does_not_select(self):
        calls=[]
        ns=self.namespace(inventory_kaggle_nojax=lambda: calls.append('inventory') or {})
        self.execute('aot-preflight-data',ns)
        self.assertEqual(calls,['inventory'])

    def test_preflight_rejects_bad_json_and_extra_keys(self):
        for data in ('{}','{"selected_gpu_uuid":""}',
                     '{"selected_gpu_uuid":"GPU-test-uuid","auto":true}'):
            with self.subTest(data=data),self.assertRaises(ValueError):
                self.execute('aot-preflight-data',self.namespace(data,
                    inventory_kaggle_nojax=lambda:{}))

    def test_preflight_rejects_live_session_before_inventory(self):
        with self.assertRaises(RuntimeError):
            self.execute('aot-preflight-data',self.namespace(AOT_SESSION={},
                inventory_kaggle_nojax=lambda:self.fail('must not inventory live run')))

    def test_prepare_requires_preflight_uuid_and_writable_platform_path(self):
        request={'candidate_package_path':'/P90.zip','output_dir':'/kaggle/working/test-output',
                 'coordination':{'selected_gpu_uuid':'GPU-test-uuid'}}
        for override in ({'KAGGLE_PREFLIGHT':{}},):
            with self.assertRaises(ValueError):
                self.execute('aot-prepare-data',self.namespace(json.dumps(request),**override))
        ns=self.namespace(json.dumps(request));ns.pop('KAGGLE_PREFLIGHT')
        with self.assertRaises(RuntimeError):self.execute('aot-prepare-data',ns)
        for path in ('/content/new','/kaggle/working','/kaggle/input/new','/tmp/new'):
            bad=dict(request,output_dir=path)
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.execute('aot-prepare-data',self.namespace(json.dumps(bad)))

    def test_prepare_writes_exact_inner_coordination_record(self):
        request={'candidate_package_path':'/P90.zip','output_dir':'/kaggle/working/record-output',
                 'coordination':{'selected_gpu_uuid':'GPU-test-uuid','authorized':True}}
        ns=self.namespace(json.dumps(request),PAYLOAD={'paths':{
            'original.ipynb':'/original','production.ipynb':'/production'}},
            prepare_aot_long_run=lambda **kw:{'root':'/kaggle/working/record-output'})
        self.execute('aot-prepare-data',ns)
        record=Path(self.temp.name)/'record-output'/'fixed_coordination.json'
        self.assertEqual(json.loads(record.read_text()),request['coordination'])
        self.assertNotIn('candidate_package_path',json.loads(record.read_text()))

    def test_idle_export_links_include_all_auditor_inputs(self):
        for name in ('aot-smoke','aot-run'):
            self.assertIn('fixed_coordination.json',self.cells[name])
            self.assertIn('status.json',self.cells[name])
            self.assertIn('FileLink',self.cells[name])
            self.assertLess(self.cells[name].index('run_until_barrier('),self.cells[name].index('FileLink'))

    def test_no_jax_import(self):
        self.assertEqual('jax' in sys.modules, self.before_jax)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixedCells)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='PASS_HOST_ONLY' if result.wasSuccessful() else 'FAIL',
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  GPU_executed=False, JAX_imported='jax' in sys.modules)
    (ROOT / 'builder_behavior_tests.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
