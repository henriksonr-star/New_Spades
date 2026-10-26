"""Public training-result and source-binding checks; no model or provider work."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class TrainingUpdate17574Checks(unittest.TestCase):
    def test_counters_match_the_resumed_configuration(self):
        result = read('results/update17574/summary.json')
        config = read('training/teacher_free_dualt4_v1/training_configuration.json')
        self.assertEqual(result['source_update'], 15247)
        self.assertEqual(result['final_update'], 17574)
        self.assertEqual(result['new_updates'], 2327)
        self.assertEqual(result['final_update'] - result['source_update'], result['new_updates'])
        self.assertEqual(result['environment_steps'],
                         6994591744 + result['new_updates'] * config['n_env'] * config['rollout_len'])
        self.assertEqual(config['n_env'], 3584)
        self.assertEqual(result['hardware']['environments_per_device'] * 2, config['n_env'])

    def test_partial_duration_is_never_reported_as_three_completed_hours(self):
        result = read('results/update17574/summary.json')
        self.assertEqual(Decimal(str(result['measured_active_seconds'])) +
                         Decimal(str(result['active_shortfall_seconds'])), Decimal(10800))
        self.assertFalse(result['requested_target_reached'])
        self.assertGreater(result['provider_elapsed_seconds'], result['measured_active_seconds'])

    def test_native_availability_is_explicit_without_a_binary_in_git(self):
        checkpoint = read('results/update17574/summary.json')['checkpoint']
        self.assertEqual(checkpoint['native_bytes'], 114647443)
        self.assertEqual(checkpoint['native_sha256'],
                         'aa64a6277ec5b31e6563749215682be82aaa42f7b0e2cf7ca60269575fdc8569')
        self.assertFalse(checkpoint['included_in_git'])
        self.assertIsNone(checkpoint['public_download_url'])
        for path in (ROOT / 'training/teacher_free_dualt4_v1').rglob('*'):
            if path.is_file():
                self.assertLess(path.stat().st_size, 100 * 1024 * 1024)
                self.assertNotIn(path.suffix, ('.npz', '.zip', '.ckpt'))

    def test_saved_configuration_hash_is_exact(self):
        config = read('training/teacher_free_dualt4_v1/training_configuration.json')
        self.assertEqual(digest(config),
                         'ca6e57b013d44bca93f87e03927e151b13535d61709cbc3ff836635785240824')
        self.assertEqual(config['lr'], 0.0002)
        self.assertFalse(read('results/update17574/summary.json')['learning']['teacher_action_supervision'])

    def test_no_unaudited_strength_result_is_promoted(self):
        result = read('results/update17574/summary.json')
        self.assertFalse(result['playing_strength_claim'])
        self.assertFalse(result['audited_evaluation_results_for_this_checkpoint_included'])
        self.assertEqual(result['verification']['finite_native_leaves'], 214)
        self.assertTrue(result['verification']['healthy_settlement_candidate'])

    def test_historical_protocol_cleanup_keeps_scientific_fields(self):
        record = read('docs/publication_metadata_sanitization.json')
        path = ROOT / record['file']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['public_sha256'])
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol.pop('source_root_at_freeze'), 'normal_lr_schedule_amendment_v3/sources')
        protocol['seed_history'].pop('audit_scope')
        self.assertEqual(digest(protocol), record['unchanged_protocol_semantic_sha256'])


if __name__ == '__main__':
    unittest.main()
