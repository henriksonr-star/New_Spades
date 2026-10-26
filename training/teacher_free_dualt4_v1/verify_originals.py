"""Verify source curation against an independently supplied original package."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_definitions(text, names):
    lines = text.splitlines(keepends=True)
    chunks = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            chunks.append(''.join(lines[start-1:node.end_lineno]))
    return '\n\n'.join(chunks)


def verify(source_root):
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    record = json.loads((source_root/'FROZEN_SOURCE_RECORD.json').read_text())
    closure = manifest['original_frozen_closure']
    assert len(record['sources']) == closure['member_count']
    assert hashlib.sha256(canonical(record['sources']).encode()).hexdigest() == closure['canonical_member_map_sha256']
    for rel, expected in record['sources'].items():
        path = source_root/rel
        assert not path.is_symlink(), rel
        assert identity(path.read_bytes()) == expected, rel
    objective = load(source_root/'frontend/prior9971/teacher_free_objective.py')
    diagnostics = load(source_root/'diagnostics/blind_diagnostics.py')
    profile = load(source_root/'runtime/novel845_profile.py')
    for entry in manifest['files']:
        original = (source_root/entry['source_path']).read_bytes()
        public = (ROOT/entry['public_path']).read_bytes()
        assert identity(original) == entry['source_original'], entry['source_path']
        assert identity(public) == entry['public'], entry['public_path']
        mode = entry['mode']
        if mode == 'unchanged_complete_file':
            assert public == original
        elif mode == 'exact_notebook_cell':
            assert public == ''.join(json.loads(original)['cells'][entry['cell_index']]['source']).encode()
        elif mode == 'exact_executed_source_generation':
            generated = objective.patched_training_definition(source_root/entry['source_path'])
            if entry['stage'] == 'final_training_definition_with_passive_diagnostics':
                generated = diagnostics.instrumented_training_definition(generated)
            assert public == generated.encode()
        elif mode == 'exact_definition_excerpt':
            selected = selected_definitions(original.decode(), entry['definitions'])
            assert identity(selected.encode()) == entry['selected_original']
            selected_ast = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(selected).body}
            public_ast = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(public).body
                          if isinstance(n, ast.FunctionDef)}
            assert selected_ast == public_ast
            assert selected in public.decode()
            assert public == public.rstrip(b'\n') + b'\n'
        elif mode == 'exact_configuration_derivation':
            config = profile.build_training_config(json.loads(original), objective)
            assert json.loads(public) == config
            assert hashlib.sha256(canonical(config).encode()).hexdigest() == entry['canonical_sha256']
        else:
            raise AssertionError('Unknown provenance mode: '+mode)
    return dict(status='passed', original_closure_members_verified=len(record['sources']),
                selected_public_entries_verified=len(manifest['files']),
                extraction_and_generation_parity=True, training_started=False,
                checkpoint_loaded=False, gpu_test_performed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root), indent=2, sort_keys=True))
