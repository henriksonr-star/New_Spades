"""Independent long-run AOT host fixtures; no JAX/GPU/network.
Dependencies: stdlib, NumPy, same-directory operator.py. Reference journal/health
helpers below are exact definitions copied from frozen production cell18
SHA0404311dc73890285e6fc6223a861cbbdfb2be7d04e20c897ef0afbf9f21aef8.
They exercise operator integration without inventing historical work rows.
"""
import sys
# The implementation file is named operator.py; preload the standard module
# before direct-script sys.path can shadow it during stdlib imports.
if sys.path and sys.path[0] == __file__.rsplit("/", 1)[0]:
    _test_entry_path = sys.path.pop(0)
    import operator as _standard_operator
    sys.path.insert(0, _test_entry_path)
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
import zipfile
from collections import namedtuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parent
DECISIONS = 458752
REAL_SHA256 = hashlib.sha256
REAL_WRITE_TEXT = Path.write_text
jax = SimpleNamespace(device_get=lambda tree: tree)

class State(namedtuple("StateBase", "arrays updates decisions_per_update bootstrap_updates bootstrap_complete env_steps_low env_steps_high base_updates")):
    __slots__ = ()
    @property
    def env_steps(self):
        prior = (int(self.env_steps_high) << 32) | int(self.env_steps_low)
        return prior + (int(self.updates) - int(self.base_updates)) * int(self.decisions_per_update)

def array(value, tag="", dtype=np.float32):
    return PublicArray(value, tag=tag, dtype=dtype)

def state(update, tag=""):
    return State([array([i / 256.], tag) for i in range(207)],
                 array(update, tag, np.int32), array(DECISIONS, tag, np.int32),
                 array(0, tag, np.int32), array(True, tag, np.bool_),
                 array(0, tag, np.uint32), array(0, tag, np.uint32), array(0, tag, np.int32))


MockDevice = namedtuple("MockDevice", "platform device_kind")
MOCK_DEVICE = MockDevice("gpu", "Tesla T4 host-only fixture")
MockLayout = namedtuple("MockLayout", "major_to_minor tiling sub_byte_element_size_in_bits")
class MockFormat:
    def __init__(self, layout): self.layout = layout
    def __eq__(self, other): return isinstance(other, MockFormat) and self.layout == other.layout
    def __repr__(self): return "MockFormat(" + repr(self.layout) + ")"
class MockSingleDeviceSharding:
    memory_kind = "device"

class MockHandle:
    def __init__(self, function, input_formats=None):
        self.function, self.input_formats = function, input_formats
    def __call__(self, *args): return self.function(*args)

class PublicArray(np.ndarray):
    def __new__(cls, values, *, tag="", weak_type=False, dtype=np.float32):
        obj = np.asarray(values, dtype=dtype).view(cls)
        obj.tag, obj.weak_type, obj.peer_fences = tag, weak_type, 0
        return obj

    def __array_finalize__(self, source):
        self.tag = getattr(source, "tag", "")
        self.weak_type = getattr(source, "weak_type", False)
        self.peer_fences = getattr(source, "peer_fences", 0)

    committed = False
    sharding = MockSingleDeviceSharding()
    @property
    def format(self):
        return MockFormat(MockLayout(tuple(range(self.ndim)), (), 0))
    @property
    def layout(self): return self.format.layout
    def on_device_size_in_bytes(self): return self.nbytes
    is_fully_addressable = True
    is_fully_replicated = True

    def devices(self):
        return {MOCK_DEVICE}

    def block_until_ready(self):
        self.peer_fences += 1
        return self


class TreeUtil:
    @staticmethod
    def tree_flatten_with_path(tree):
        leaves = []

        def visit(value, path):
            if isinstance(value, dict):
                return ("dict", tuple((k, visit(value[k], path + (repr(k),))) for k in sorted(value)))
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                return (type(value).__name__, tuple((k, visit(v, path + (k,)))
                         for k, v in zip(value._fields, value)))
            if isinstance(value, (list, tuple)):
                return (type(value).__name__, tuple(visit(v, path + (str(i),))
                         for i, v in enumerate(value)))
            leaves.append((path, value))
            return "leaf"

        return leaves, visit(tree, ())

    @classmethod
    def tree_leaves(cls, tree):
        return [value for _, value in cls.tree_flatten_with_path(tree)[0]]

    @staticmethod
    def tree_map(function, tree):
        if isinstance(tree, dict): return {k: TreeUtil.tree_map(function, v) for k, v in tree.items()}
        if isinstance(tree, tuple) and hasattr(tree, "_fields"):
            return type(tree)(*(TreeUtil.tree_map(function, v) for v in tree))
        if isinstance(tree, (list, tuple)): return type(tree)(TreeUtil.tree_map(function, v) for v in tree)
        return function(tree)

    @staticmethod
    def keystr(path):
        return ".".join(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class OptimizerHealthStop(RuntimeError):
    """Accepted state is retained; inspect the durable stop checkpoint before retry."""


def _optimizer_exact_integer(value, label, *, minimum=0, allow_bool=False):
    """Accept scalar numeric counts, never fractional/nonfinite/coerced strings."""
    array = np.asarray(jax.device_get(value))
    if (array.shape != () or array.dtype.kind not in ('i', 'u', 'f', 'b')
            or array.dtype.kind == 'b' and not allow_bool):
        raise ValueError(f'{label} must be an exact scalar numeric integer.')
    number = array.item()
    if not np.isfinite(number) or number != int(number) or number < minimum:
        raise ValueError(f'{label} must be a finite integer >= {minimum}.')
    return int(number)


def _optimizer_cadence(cfg):
    values = {name: _optimizer_exact_integer(cfg[name], name, minimum=1)
              for name in ('n_env', 'rollout_len', 'minibatch_size', 'ppo_epochs')}
    decisions = values['n_env'] * values['rollout_len']
    if decisions % values['minibatch_size']:
        raise ValueError('Saved PPO cadence must divide into whole minibatches.')
    return values


def _optimizer_config_signature(cfg):
    """Fingerprint every saved learning setting, not only minibatch cadence."""
    return hashlib.sha256(json.dumps(_json_safe(dict(cfg)),sort_keys=True,
        separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _optimizer_work_counts(cfg, metrics):
    """Validate actual raw optimizer counters before accepting a candidate."""
    cadence = _optimizer_cadence(cfg)
    planned = cadence['ppo_epochs'] * (cadence['n_env'] * cadence['rollout_len'] // cadence['minibatch_size'])
    try:
        counts = {target: _optimizer_exact_integer(metrics[source], source)
            for source, target in (('attempted_updates', 'attempted'),
                ('accepted_updates', 'accepted'), ('rejected_updates', 'rejected'))}
        kl = _optimizer_exact_integer(metrics['kl_early_stop'], 'kl_early_stop', allow_bool=True)
    except KeyError as exc:
        raise ValueError('Missing optimizer work metric: ' + str(exc)) from exc
    if not 1 <= counts['accepted'] == counts['attempted'] <= planned or counts['rejected'] != 0:
        raise ValueError('Optimizer counts require1<=accepted==attempted<=planned and rejected=0.')
    if kl not in (0, 1) or counts['attempted'] < planned and not kl:
        raise ValueError('Skipped optimizer slots require an exact KL-stop flag.')
    return dict(planned=planned, **counts, skipped=planned-counts['attempted'],
        attempted_fraction=counts['attempted']/planned, accepted_fraction=counts['accepted']/planned,
        skipped_fraction=(planned-counts['attempted'])/planned, kl_early_stop=bool(kl))


def _optimizer_journal_rows(history, *, ignored=None, current_update=None):
    """Retain actual journaled work; a trainer migration starts a new lineage."""
    start = max((i+1 for i, row in enumerate(history)
                 if row.get('kind') == 'trainer_upgrade'), default=0)
    rows = []
    ignored = [] if ignored is None else ignored
    for index, row in enumerate(history[start:],start):
        if row.get('kind') in ('optimizer_health_stop','optimizer_health_acknowledgement'):
            # A recorded stop/explicit recovery boundary never stitches an old
            # partial health window to later optimizer work. Totals stay intact.
            if rows:
                rows[-1]['continuity_break_after']=True
            if row.get('kind')=='optimizer_health_acknowledgement' and not (
                    type(row.get('optimizer_health_version')) is int and row['optimizer_health_version']==1
                    and type(row.get('stop_update')) is int and row['stop_update']>=1
                    and isinstance(row.get('stop_id'),str) and len(row['stop_id'])==64
                    and all(c in '0123456789abcdef' for c in row['stop_id'])
                    and isinstance(row.get('config_signature'),str) and len(row['config_signature'])==64
                    and all(c in '0123456789abcdef' for c in row['config_signature'])
                    and isinstance(row.get('reason'),str) and len(row['reason'].strip())>=8):
                ignored.append(dict(history_index=index,kind='optimizer_health_acknowledgement',
                    reason='Ignored malformed or unrecognized optional health acknowledgement.'))
            continue
        if row.get('kind') != 'optimizer_work':
            continue
        try:
            if (type(row.get('optimizer_work_version')) is not int or row['optimizer_work_version'] != 1
                    or not isinstance(row.get('cadence'), dict)):
                raise ValueError('Unsupported or malformed optimizer-work journal.')
            signature=row.get('config_signature')
            if (not isinstance(signature,str) or len(signature)!=64
                    or any(c not in '0123456789abcdef' for c in signature)):
                raise ValueError('Missing or malformed full saved-config signature.')
            update = _optimizer_exact_integer(row.get('update'), 'journal update', minimum=1)
            if current_update is not None and update > current_update:
                raise ValueError('Optional optimizer journal runs ahead of the saved training state.')
            if rows and update <= rows[-1]['update']:
                raise ValueError('Optional optimizer journal update ID is duplicate or out of order.')
            restored = _optimizer_work_counts(row['cadence'], dict(
                attempted_updates=row.get('attempted'), accepted_updates=row.get('accepted'),
                rejected_updates=row.get('rejected'), kl_early_stop=row.get('kl_early_stop')))
            for name in ('planned', 'attempted', 'accepted', 'skipped', 'rejected'):
                actual = _optimizer_exact_integer(row.get(name), 'journal '+name)
                if actual != restored[name]:
                    raise ValueError('Optional journal counts do not match its cadence.')
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            ignored.append(dict(history_index=index,kind='optimizer_work',reason=str(exc)))
            if rows:
                rows[-1]['continuity_break_after']=True
            continue
        rows.append(dict(restored, kind='optimizer_work', optimizer_work_version=1,
                         update=update, cadence=_optimizer_cadence(row['cadence']),config_signature=signature))
    return rows


def _optimizer_health(rows):
    """Three consecutive same-cadence KL stops below25% trigger a safe stop."""
    window = rows[-3:]
    planned, accepted = sum(row['planned'] for row in window), sum(row['accepted'] for row in window)
    consecutive = (len(window) == 3 and all(window[i]['update']+1 == window[i+1]['update']
                    for i in range(2)) and all(row['cadence'] == window[0]['cadence']
                    and row['config_signature']==window[0]['config_signature'] for row in window)
                    and not any(row.get('continuity_break_after',False) for row in window[:-1]))
    all_kl = bool(consecutive and all(row['kl_early_stop'] for row in window))
    fraction = accepted/planned if planned else 0.0
    return dict(window_rollouts=len(window), planned=planned, accepted=accepted,
        aggregate_accepted_fraction=fraction, consecutive_same_cadence=bool(consecutive),
        all_three_kl_stopped=all_kl, should_stop=bool(all_kl and 4*accepted < planned))


def _optimizer_totals(rows):
    """Actual journal coverage only; never infer missing historic minibatches."""
    totals = {name: sum(row[name] for row in rows)
              for name in ('planned', 'attempted', 'accepted', 'skipped', 'rejected')}
    planned = totals['planned']
    return dict(totals, journaled_ppo_rollouts=len(rows),
        kl_stopped_rollouts=sum(bool(row['kl_early_stop']) for row in rows),
        attempted_fraction=totals['attempted']/planned if planned else 0.0,
        accepted_fraction=totals['accepted']/planned if planned else 0.0,
        skipped_fraction=totals['skipped']/planned if planned else 0.0,
        first_journaled_update=rows[0]['update'] if rows else None,
        last_journaled_update=rows[-1]['update'] if rows else None,
        interpretation='Observed logged optimizer minibatches; prior or missing updates are not inferred as lifetime work.')


def _optimizer_resume_journal(history, cfg, current_update, acknowledgement=None):
    """Use checkpoint history; require an explicit matching recovery after a stop."""
    current_update = _optimizer_exact_integer(current_update, 'current update')
    ignored=[]
    rows = _optimizer_journal_rows(history,ignored=ignored,current_update=current_update)
    start = max((i+1 for i, row in enumerate(history)
                 if row.get('kind') == 'trainer_upgrade'), default=0)
    stopped=None
    stopped_index=None
    signature=_optimizer_config_signature(cfg)
    for index,row in enumerate(history[start:],start):
        if row.get('kind')!='optimizer_health_stop':
            continue
        if (type(row.get('optimizer_health_version')) is int and row['optimizer_health_version']==1
                and row.get('config_signature')==signature and isinstance(row.get('reason'),str)
                and len(row['reason'].strip())>=8
                and isinstance(row.get('stop_id'),str) and len(row['stop_id'])==64
                and all(c in '0123456789abcdef' for c in row['stop_id'])
                and type(row.get('update')) is int and 1<=row['update']<=current_update):
            stopped=row
            stopped_index=index
        else:
            ignored.append(dict(history_index=index,kind='optimizer_health_stop',
                reason='Ignored unrecognized, malformed or different-config optional health-stop record.'))
    if ignored:
        history.append(dict(kind='optimizer_journal_notice',update=current_update,
            reason='Optional older health records ignored; rolling window reset across uncertain coverage.',
            ignored_records=ignored))
    if stopped is None:
        if acknowledgement is not None:
            raise ValueError('Optimizer health acknowledgement is stale: no matching current stop.')
        return rows
    acknowledged = any(row.get('kind')=='optimizer_health_acknowledgement'
        and type(row.get('optimizer_health_version')) is int and row['optimizer_health_version']==1
        and row.get('stop_id')==stopped['stop_id'] and type(row.get('stop_update')) is int
        and row['stop_update']==stopped['update'] and row.get('config_signature')==signature
        and isinstance(row.get('reason'),str) and len(row['reason'].strip())>=8
        for row in history[stopped_index+1:])
    if acknowledgement is not None:
        if (not isinstance(acknowledgement,dict)
                or type(acknowledgement.get('stop_update')) is not int
                or acknowledgement['stop_update']!=stopped['update']
                or acknowledgement.get('stop_id')!=stopped['stop_id']
                or acknowledgement.get('config_signature')!=signature
                or not isinstance(acknowledgement.get('reason'),str)
                or len(acknowledgement['reason'].strip())<8):
            raise ValueError('Optimizer health acknowledgement must match the latest stop ID/update/config and document its reviewed fix.')
        if not acknowledged:
            if current_update!=stopped['update']:
                raise ValueError('A new acknowledgement must recover exactly the stopped accepted update.')
            history.append(dict(kind='optimizer_health_acknowledgement',optimizer_health_version=1,
                stop_id=stopped['stop_id'],stop_update=stopped['update'],config_signature=signature,
                update=current_update,reason=acknowledgement['reason'].strip(),
                scope='Reset rolling health window only; model/optimizer/RNG/counters/config unchanged.'))
            if rows:
                rows[-1]['continuity_break_after']=True
            acknowledged=True
    if not acknowledged:
        raise OptimizerHealthStop('This native checkpoint records an optimizer health stop. '
            'Inspect its reason and recovery checkpoint; do not rerun unchanged. '
            'After a reviewed fix, supply a matching explicit RUN_CFG optimizer_health_acknowledgement. '
            + str(stopped.get('checkpoint_path',stopped.get('checkpoint_tag','optimizer_health_stop'))))
    return rows


def _optimizer_work_record(previous, candidate, cfg, metrics):
    work = _optimizer_work_counts(cfg, metrics)
    old_update=_optimizer_exact_integer(previous.updates,'previous PPO update')
    new_update=_optimizer_exact_integer(candidate.updates,'candidate PPO update')
    old_steps=_optimizer_exact_integer(previous.env_steps,'previous lifetime decisions')
    new_steps=_optimizer_exact_integer(candidate.env_steps,'candidate lifetime decisions')
    if new_update != old_update+1:
        raise ValueError('Accepted PPO candidate must advance exactly one update.')
    if new_steps != old_steps + int(cfg['n_env'])*int(cfg['rollout_len']):
        raise ValueError('Accepted PPO candidate lifetime decisions are inconsistent.')
    return dict(work, kind='optimizer_work', optimizer_work_version=1,
        update=new_update, env_steps=new_steps, cadence=_optimizer_cadence(cfg),
        config_signature=_optimizer_config_signature(cfg))


CFG = {'seed': 42, 'n_env': 3584, 'rollout_len': 128, 'ppo_epochs': 3, 'minibatch_size': 57344, 'width': 256, 'lr': 0.0002, 'gae_lambda': 0.995, 'gae_per_hand': False, 'clip_eps': 0.2, 'value_clip': 0.25, 'max_grad_norm': 0.5, 'entropy_coef': 0.015, 'value_coef': 0.5, 'target_kl': 0.04, 'snapshot_size': 4, 'snapshot_interval_updates': 100, 'selfplay_prob': 0.5, 'snapshot_prob': 0.25, 'bootstrap_epochs': 2, 'bootstrap_minibatch_size': 1024, 'teacher_warmup_decisions': 6291456, 'hand_score_reward_coef': 0, 'bidding_anchor_coef': 0.1, 'play_anchor_coef': 0.3, 'teacher_style': 7, 'teacher_decline_blind': True, 'max_training_match_hands': 128}


