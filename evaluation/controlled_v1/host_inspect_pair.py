"""Check the two actual native inputs on CPU before any cloud allocation.

No JAX, optimizer, provider, network, training or campaign-journal action.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-native', required=True)
    parser.add_argument('--candidate-native', required=True)
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'runtime/controlled_eval_runtime.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'be469f5238303e6e55d466f9311053ee2e8aa1aec4b497001ae215016e469699':
        raise ValueError('Reviewed evaluation runtime bytes changed.')
    spec = importlib.util.spec_from_file_location('controlled_eval_readonly_host_check', path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    records = {arm: runtime.inspect_native(value, arm)[0] for arm, value in (
        ('baseline', args.baseline_native), ('candidate', args.candidate_native))}
    runtime.validate_native_pair(records)
    if any(name == 'jax' or name.startswith(('jax.', 'jaxlib', 'jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Unexpected framework import in CPU-only native inspection.')
    print(json.dumps(dict(status='PASS_HOST_CURRENT_NATIVE_CONTENT', checkpoints=records,
        framework_imported=False, gpu_allocated=False, training=False,
        campaign_journal_mutated=False, playing_strength_evaluated=False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
