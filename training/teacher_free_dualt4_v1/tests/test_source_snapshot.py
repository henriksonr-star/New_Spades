"""Publication host checks. No JAX import, checkpoint restore or training."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT/relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def opponent_namespace():
    namespace = {'jnp': np}
    paths = ['derived/base_opponents_cell8.py'] + [
        'source/frontend/prior9971/'+name for name in (
            'training_opponents_v2.py', 'formula_opponents_v3.py', 'tactical_opponents_v4.py')]
    with contextlib.redirect_stdout(io.StringIO()):
        for relative in paths:
            exec(compile((ROOT/relative).read_bytes(), relative, 'exec'), namespace)
    return namespace


class SourceSnapshotTests(unittest.TestCase):
    def test_allowlisted_hashes_and_ast(self):
        manifest = json.loads((ROOT/'source_manifest.json').read_text())
        self.assertEqual(len(manifest['files']), 23)
        for item in manifest.get('publication_support_files', []):
            raw = (ROOT/item['public_path']).read_bytes()
            self.assertEqual(len(raw), item['public']['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item['public']['sha256'])
        allowed = {entry['public_path'] for entry in manifest['files']}
        allowed.update(item['public_path'] for item in manifest.get('publication_support_files', []))
        allowed.add('source_manifest.json')
        actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts}
        self.assertEqual(actual, allowed)
        for entry in manifest['files']:
            raw = (ROOT/entry['public_path']).read_bytes()
            self.assertEqual(len(raw), entry['public']['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['public']['sha256'])
            if entry['mode'] == 'unchanged_complete_file':
                self.assertEqual(entry['source_original'], entry['public'])
            if entry['public_path'].endswith('.py'):
                tree = ast.parse(raw)
                digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest, entry['public_ast_sha256'])
            if entry['mode'] == 'exact_definition_excerpt':
                nodes = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
                for name, expected in entry['definition_ast_sha256'].items():
                    self.assertEqual(hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest(), expected)

    def test_all_python_parses_and_no_launch_notebooks(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_bytes(), filename=str(path.relative_to(ROOT)))
        self.assertEqual(list(ROOT.rglob('*.ipynb')), [])
        self.assertEqual(list(ROOT.rglob('*.zip')), [])
        excerpt = (ROOT/'derived/checkpoint_guard_excerpt.py').read_bytes()
        self.assertEqual(excerpt, excerpt.rstrip(b'\n')+b'\n')

    def test_private_literal_scan(self):
        forbidden = [r'libfile'+r'_[A-Za-z0-9_]+', '/work'+'space/',
                     r'01a[0-9a-f]{5}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                     r'https?://(?:drive|docs)\.google\.com/', r'gh[pousr]_[A-Za-z0-9]{20,}',
                     r'AKIA[0-9A-Z]{16}']
        for path in ROOT.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.md', '.json'):
                for pattern in forbidden:
                    self.assertIsNone(re.search(pattern, path.read_text()), str(path.relative_to(ROOT)))

    def test_config_hash_schema_and_rejection(self):
        cfg = json.loads((ROOT/'training_configuration.json').read_text())
        self.assertEqual(hashlib.sha256(canonical(cfg).encode()).hexdigest(),
                         'ca6e57b013d44bca93f87e03927e151b13535d61709cbc3ff836635785240824')
        objective = load('source/frontend/prior9971/teacher_free_objective.py')
        profile = load('source/runtime/novel845_profile.py')
        recovery = load('source/runtime/recovery12120_profile.py')
        legacy = profile.source_config(cfg)
        self.assertTrue(objective.validate_config(legacy))
        self.assertEqual(profile.build_training_config(legacy, objective), cfg)
        self.assertEqual(recovery.build_training_config(cfg, objective), cfg)
        self.assertFalse(set(cfg) & set(objective.REMOVED_SETTINGS))
        self.assertEqual((cfg['lr'], cfg['hand_score_reward_coef']), (.0002, 0))
        self.assertEqual((cfg['n_env']//2, cfg['minibatch_size']//2), (1792, 28672))
        for changed in [dict(legacy, lr=.0001), dict(legacy, play_anchor_coef=0), dict(legacy, n_env=True)]:
            with self.assertRaises(ValueError):
                objective.validate_config(changed)

    def test_final_definition_is_exact_passive_instrumentation(self):
        base = (ROOT/'derived/teacher_free_make_train_step.py').read_text()
        final = (ROOT/'derived/instrumented_make_train_step.py').read_text()
        diagnostics = load('source/diagnostics/blind_diagnostics.py')
        self.assertEqual(diagnostics.instrumented_training_definition(base), final)
        forbidden = {'teacher_action', 'teacher_actions', 'teacher_mask', 'bidding_anchor_loss',
                     'play_anchor_loss', 'make_bootstrap_step', '_plasticity_anchor_gate'}
        for text in (base, final):
            names = {n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)}
            self.assertFalse(names & forbidden)
            self.assertIn('loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy', text)
        def nested(text, name):
            return ast.dump(next(n for n in ast.walk(ast.parse(text))
                                 if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
        for name in ('loss_fn', 'optimize'):
            self.assertEqual(nested(base, name), nested(final, name))
        with self.assertRaises(ValueError):
            diagnostics.instrumented_training_definition(base+'\n')

    def test_sequential_patch_and_dimensions_are_preserved(self):
        patch = load('source/runtime/sequential_bn_visibility.py')
        combined = (ROOT/'source/runtime/sequential_bn_rules.py').read_text()
        self.assertTrue(combined.endswith('\n\n'+patch.PATCH_SOURCE))
        original = combined[:-len('\n\n'+patch.PATCH_SOURCE)]
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), patch.RULES_SHA256)
        self.assertEqual(patch.patched_rules_source(original), combined)
        with self.assertRaises(ValueError):
            patch.patched_rules_source(combined)
        nodes = [n.name for n in ast.parse(combined).body if isinstance(n, ast.FunctionDef)]
        for name in ('_blind_turn', '_observe_for', 'observe', 'critic_observe'):
            self.assertEqual(nodes.count(name), 2)

    def test_opponent_installation_rosters_and_exclusion(self):
        namespace = opponent_namespace()
        installer = load('source/opponents/install_expanded_opponents_v5.py')
        cfg = json.loads((ROOT/'training_configuration.json').read_text())
        report = installer.install(namespace, cfg)
        self.assertEqual((report['train_style_count'], report['heldout_style_count']), (845, 256))
        train, holdout = map(set, (namespace['NOVEL_TRAIN_STYLE_IDS'], namespace['NOVEL_HOLDOUT_STYLE_IDS']))
        self.assertEqual((len(train), len(holdout), len(train & holdout)), (845, 256, 0))
        self.assertEqual(len(train | holdout), 1101)
        for symbol, filename in [('DIVERSITY_OPPONENT_MANIFEST', 'opponent_manifest.json'),
                                 ('EXPANSION_OPPONENT_MANIFEST', 'training_opponent_manifest_v2.json'),
                                 ('FORMULA_OPPONENT_MANIFEST', 'formula_manifest_v3.json'),
                                 ('TACTICAL_OPPONENT_MANIFEST', 'tactical_opponent_manifest_v4.json')]:
            self.assertEqual(namespace[symbol], json.loads((ROOT/'source/frontend/prior9971'/filename).read_text()))
        # NumPy host execution of pure ID predicates; this does not run JAX policies.
        all_ids = np.array(sorted(train | holdout), dtype=np.int32)
        train_mask = namespace['is_train_opponent_style_id'](all_ids)
        holdout_mask = namespace['is_holdout_opponent_style_id'](all_ids)
        self.assertEqual(set(all_ids[train_mask]), train)
        self.assertEqual(set(all_ids[holdout_mask]), holdout)
        self.assertFalse(np.any(train_mask & holdout_mask))
        with self.assertRaises(ValueError):
            installer.install(namespace, cfg)
        with self.assertRaises(ValueError):
            installer.sampled_id_diagnostics(namespace, np.array([min(holdout)], dtype=np.int32))

    def test_invalid_opponent_configuration_rejected_before_mutation(self):
        namespace = opponent_namespace()
        installer = load('source/opponents/install_expanded_opponents_v5.py')
        cfg = json.loads((ROOT/'training_configuration.json').read_text())
        with self.assertRaises(ValueError):
            installer.install(namespace, dict(cfg, snapshot_prob=.1))
        self.assertNotIn('NOVEL_OPPONENT_REVISION', namespace)

    def test_diagnostic_zero_denominators_and_inconsistent_flow(self):
        diagnostic = load('source/diagnostics/blind_diagnostics.py')
        metrics = dict.fromkeys(diagnostic.ALL_METRIC_NAMES, 0)
        result = diagnostic.summarize(metrics)
        self.assertTrue(all(row['value'] is None for row in result['rates'].values()))
        with self.assertRaises(ValueError):
            diagnostic.summarize(dict(metrics, blind_diag_declarations=1))
        with self.assertRaises(ValueError):
            diagnostic.summarize_exposure(dict(metrics, opponent_diag_invalid_learner_decisions=1))

    def test_synthetic_diagnostic_exposure_uses_accepted_assignments(self):
        diagnostic = load('source/diagnostics/blind_diagnostics.py')
        kinds = np.array([2, 2, 0])
        styles = np.array([20000, 20064, 20128])
        valid = np.ones(3, dtype=bool)
        counts = diagnostic.opponent_exposure_counts(kinds, styles, valid,
            np.array([True, False, True]), kinds, styles, valid,
            np.array([False, True, True]), xp=np)
        n = len(diagnostic.OPPONENT_FAMILIES)
        self.assertEqual(float(sum(counts[:n])), 1)
        self.assertEqual(float(sum(counts[n:2*n])), 1)
        self.assertEqual(float(sum(counts[2*n:])), 1)
        self.assertEqual(float(counts[2*n+2]), 1)

    def test_compiler_memory_requires_complete_nonnegative_evidence(self):
        dual = load('source/frontend/prior9971/dual_gpu_runtime.py')
        values = dict(argument_size_in_bytes=10, output_size_in_bytes=20,
                      temp_size_in_bytes=30, alias_size_in_bytes=5)
        fake = types.SimpleNamespace(memory_analysis=lambda: types.SimpleNamespace(**values))
        self.assertEqual(dual.compiled_memory(fake)['total_bytes'], 55)
        values['temp_size_in_bytes'] = -1
        with self.assertRaises(RuntimeError):
            dual.compiled_memory(fake)
        with self.assertRaises(RuntimeError):
            dual.compiled_memory(types.SimpleNamespace(memory_analysis=lambda: None))

    def test_hlo_parser_requires_operand_shapes(self):
        dual = load('source/frontend/prior9971/dual_gpu_runtime.py')
        fixture = '''ENTRY %main {
  %lhs = f32[1792,753] parameter(0)
  %rhs = f32[753,256] parameter(1)
  ROOT %dot = f32[1792,256] dot(%lhs, %rhs)
}
'''
        result = dual._hlo_matrix_operations(fixture)
        self.assertEqual(result[0]['operand_shapes'], ['f32[1792,753]', 'f32[753,256]'])
        with self.assertRaises(RuntimeError):
            dual._hlo_matrix_operations(fixture.replace('dot(%lhs, %rhs)', 'dot(%unknown, %rhs)'))

    def test_two_device_admission_rejects_absent_execution_evidence(self):
        dual = load('source/frontend/prior9971/dual_gpu_runtime.py')
        with self.assertRaises(ValueError):
            dual.validate_execution_evidence(None)
        self.assertFalse(dual.validate_execution_evidence(None, require_training=False)['training_admitted'])
        with self.assertRaises(ValueError):
            dual.validate_execution_evidence({'format': dual.FORMAT, 'status': 'passed'})

    def test_scoped_readback_cleanup_and_nesting(self):
        guard = load('derived/checkpoint_guard_excerpt.py')
        ns, data = {}, {}
        transfer = dict(namespace=ns, data=data, accepted_readback_session=None)
        session = dict(ns=ns, data=data, native_transfer_guard=transfer)
        with self.assertRaisesRegex(RuntimeError, 'synthetic readback error'):
            with guard._accepted_native_readback(session):
                self.assertIs(transfer['accepted_readback_session'], session)
                with self.assertRaisesRegex(RuntimeError, 'nonnested'):
                    with guard._accepted_native_readback(session):
                        pass
                raise RuntimeError('synthetic readback error')
        self.assertIsNone(transfer['accepted_readback_session'])

    def test_readback_keeps_transfer_admission_and_config_binding(self):
        guard = load('derived/checkpoint_guard_excerpt.py')
        calls = []
        guard._preparation_guard = lambda data, label: calls.append(('setup', label))
        guard._native_readback_guard = lambda session, label: calls.append(('readback', label))
        def original(saved, size):
            calls.append(('memory', size))
            return 'admitted'
        ns, data = {'gpu_admit_checkpoint_load': original}, {}
        transfer = guard._install_checkpoint_transfer_admission(ns, data)
        session = dict(ns=ns, data=data, native_transfer_guard=transfer, saved_cfg={'lr': .0002})
        self.assertEqual(ns['gpu_admit_checkpoint_load'](session['saved_cfg'], 100), 'admitted')
        self.assertEqual([row[0] for row in calls], ['setup', 'memory', 'setup'])
        calls.clear()
        with guard._accepted_native_readback(session):
            self.assertEqual(ns['gpu_admit_checkpoint_load'](session['saved_cfg'], 100), 'admitted')
            self.assertEqual([row[0] for row in calls], ['readback', 'memory', 'readback'])
            calls.clear()
            with self.assertRaises(ValueError):
                ns['gpu_admit_checkpoint_load']({'lr': .0001}, 100)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
