"""Source, pin and fail-closed distribution checks; never import JAX."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASELINE='2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistributionChecks(unittest.TestCase):
    def notebooks(self):
        return [json.loads((ROOT/'notebooks'/name).read_text()) for name in
            ('source_recovered.ipynb','runtime_safety_v2.ipynb','runtime_kaggle_v1.ipynb')]

    def test_all_notebooks_have38cells_and_no_execution_metadata(self):
        for notebook in self.notebooks():
            self.assertEqual(len(notebook['cells']),38)
            self.assertEqual(set(notebook['metadata']),{'kernelspec','language_info'})
            for cell in notebook['cells']:
                self.assertEqual(cell['metadata'],{})
                if cell['cell_type']=='code':
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])
                    ast.parse(''.join(cell['source']))

    def test_rules_opponents_learning_source_match(self):
        original,colab,kaggle=self.notebooks()
        for index in (6,8,10):
            self.assertEqual(original['cells'][index]['source'],colab['cells'][index]['source'])
            self.assertEqual(original['cells'][index]['source'],kaggle['cells'][index]['source'])

    def test_kaggle_only_changes_telemetry_cell12(self):
        _,colab,kaggle=self.notebooks()
        for index in range(38):
            if index!=12:
                self.assertEqual(colab['cells'][index]['source'],kaggle['cells'][index]['source'])
        self.assertNotEqual(colab['cells'][12]['source'],kaggle['cells'][12]['source'])

    def test13_opponents_preserved(self):
        original,_,_=self.notebooks()
        tree=ast.parse(''.join(original['cells'][8]['source']))
        names=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='HEURISTIC_NAMES' for t in node.targets))
        self.assertEqual(len(names),13)
        self.assertEqual(names[7],'adaptive')

    def test_templates_keep_false_gates_and_fixed_baseline(self):
        for provider in ('colab_c3','kaggle_v1'):
            template=json.loads((ROOT/'runtime'/provider/'coordination_TEMPLATE.json').read_text())
            for gate in ('run_authorized','parent_review_complete','runtime_exclusive','candidate_native_export_verified'):
                self.assertIs(template[gate],False)
            self.assertEqual(template['candidate_external_receipt'],dict(
                library_receipt='LOCAL_VERIFIED_UPDATE674',archive_sha256=BASELINE))
            self.assertEqual(template['source_pins']['candidate_package_sha256'],BASELINE)
            self.assertEqual(template['supplemental_export_interval_seconds'],900)

    def test_rebound_source_pins_and_exact_byte_limits(self):
        for provider,production in (('colab_c3','runtime_safety_v2.ipynb'),('kaggle_v1','runtime_kaggle_v1.ipynb')):
            directory=ROOT/'runtime'/provider
            builder=load('distribution_pins_'+provider,directory/'operator_notebook_builder.py')
            template=json.loads((directory/'coordination_TEMPLATE.json').read_text())
            for member,path,pin in (
                ('original.ipynb',ROOT/'notebooks/source_recovered.ipynb','original_sha256'),
                ('production.ipynb',ROOT/'notebooks'/production,'production_sha256')):
                raw=path.read_bytes()
                self.assertEqual(template['source_pins'][pin],hashlib.sha256(raw).hexdigest())
                self.assertEqual(builder.PAYLOAD_LIMITS[member],len(raw))
            self.assertEqual(builder.ORIGINAL_SHA256,template['source_pins']['original_sha256'])
            self.assertEqual(builder.PRODUCTION_SHA256,template['source_pins']['production_sha256'])
            auditor=load('distribution_auditor_'+provider,directory/'audit_export_package.py')
            self.assertEqual(auditor.PINS,template['source_pins'])

    def test_operator_inert_definitions_and_unauthorized_gate(self):
        for provider in ('colab_c3','kaggle_v1'):
            directory=ROOT/'runtime'/provider
            builder=load('distribution_definitions_'+provider,directory/'operator_notebook_builder.py')
            builder.operator_definitions_tree((directory/'operator.py').read_text())
            op=load('distribution_operator_'+provider,directory/'operator.py')
            template=json.loads((directory/'coordination_TEMPLATE.json').read_text())
            with self.assertRaisesRegex(RuntimeError,'review'):
                op.prepare_aot_long_run(original_path='missing',production_path='missing',
                    candidate_package_path='missing',coordination=template,output_dir='missing',recovery_sink={})

    def test_foreign_initial_receipt_cannot_pass(self):
        for provider in ('colab_c3','kaggle_v1'):
            directory=ROOT/'runtime'/provider
            op=load('distribution_bad_receipt_'+provider,directory/'operator.py')
            template=json.loads((directory/'coordination_TEMPLATE.json').read_text())
            for gate in ('run_authorized','parent_review_complete','runtime_exclusive','candidate_native_export_verified'):
                template[gate]=True
            template['candidate_external_receipt']['library_receipt']='UNVERIFIED'
            with self.assertRaisesRegex(RuntimeError,'receipt'):
                op.prepare_aot_long_run(original_path='missing',production_path='missing',
                    candidate_package_path='missing',coordination=template,output_dir='missing',recovery_sink={})

    def test_payload_builder_verifies_clean_both_provider_bindings(self):
        builder=load('distribution_payload_builder',ROOT/'tools/build_payload.py')
        for provider in ('colab','kaggle'):
            with tempfile.TemporaryDirectory() as directory:
                receipt=builder.build(provider,Path(directory)/'new')
                self.assertFalse(receipt['gpu_started'])
                self.assertFalse(receipt['candidate_checkpoint_included'])
                self.assertFalse(receipt['external_upload_verified'])

    def test_provenance_pins_match_distributed_source(self):
        manifest=json.loads((ROOT/'docs/source_provenance.json').read_text())
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT/entry['distribution_path']).read_bytes()).hexdigest(),
                entry['distribution_sha256'])

    def test_no_private_external_ids_or_private_machine_paths(self):
        forbidden=(r'libfile'+r'_[A-Za-z0-9_]+',r'/work'+'space/spades',
            r'01a[0-9a-f]{5}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
        # Do not scan these literal checks as if they were leaked source data.
        for path in ROOT.rglob('*'):
            if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts or path==Path(__file__):
                continue
            if path.suffix in ('.py','.md','.json','.ipynb','.txt'):
                text=path.read_text()
                for pattern in forbidden:
                    self.assertIsNone(re.search(pattern,text),str(path.relative_to(ROOT)))


if __name__=='__main__':
    unittest.main()
