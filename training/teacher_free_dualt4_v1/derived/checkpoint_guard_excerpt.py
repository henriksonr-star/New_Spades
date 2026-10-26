"""Exact checkpoint guard excerpts for inspection, not a standalone runtime.

Original caller, authorization, clocks, file ownership and native validation
dependencies are intentionally omitted. See README and source_manifest.json.
"""
from contextlib import contextmanager
from pathlib import Path
import copy
import os
import time
import uuid

def _native_readback_guard(s,label):
    """Recovery uses the export reserve, never a new setup or workload window."""
    data=s['data'];_progress(data,label)
    now,elapsed=_wall_origin(data,label)
    if (now>=_utc(data['claim']['platform_hard_end_utc'])
            or elapsed>=data['feasibility']['maximum_segment_wall_seconds']):
        raise BoundaryStop('Nonrenewable checkpoint recovery hard deadline:'+label)
    s['dual'].identity_guard();s['v2']._memory_options_guard(s['options'])
    return elapsed


@contextmanager
def _accepted_native_readback(s):
    """Scope only the accepted-state loader; exceptions cannot leak permission."""
    transfer=s['native_transfer_guard']
    if (transfer['namespace'] is not s['ns'] or transfer['data'] is not s['data']
            or transfer['accepted_readback_session'] is not None):
        raise RuntimeError('Exact nonnested accepted-native readback scope required.')
    transfer['accepted_readback_session']=s
    try:
        yield
    finally:
        transfer['accepted_readback_session']=None


def _install_checkpoint_transfer_admission(ns,data):
    """Ordinary restores retain setup admission; scoped readback retains memory admission."""
    original=ns['gpu_admit_checkpoint_load']
    transfer=dict(namespace=ns,data=data,accepted_readback_session=None)
    def check(label,saved):
        session=transfer['accepted_readback_session']
        if session is None:
            return _preparation_guard(data,label)
        if saved!=session['saved_cfg']:
            raise ValueError('Accepted-native readback configuration differs.')
        return _native_readback_guard(session,label)
    def load_admit(saved,size):
        check('native before both-device transfer admission',saved)
        result=original(saved,size)
        check('native after both-device transfer admission',saved)
        return result
    ns['gpu_admit_checkpoint_load']=load_admit
    return transfer


def _remove_owned_pending(s,info,protected):
    """Rotate only this new run's superseded local pending files."""
    path=Path(info['path']);root=Path(s['root']).resolve()
    if str(path) in protected:return
    if root not in path.resolve().parents or path.is_symlink() or _identity(path)!=info:
        raise RuntimeError('Refuse rotation outside exact owned pending file.')
    path.unlink()


def _save_native(s,reason,terminal=False):
    _progress(s['data'],'native checkpoint '+reason,accepted_update=s['accepted_update'])
    clock=s['active_clock'].summary();nonce=uuid.uuid4().hex
    s['history'].append(dict(kind='normal_lr_checkpoint',claim_id=s['data']['claim']['claim_id'],
        update=s['accepted_update'],env_steps=s['accepted_env_steps'],kernel_event_head_sha256=clock['ledger_sha256'],
        active_clock=clock,terminal=bool(terminal),reason=reason,nonce=nonce))
    s['recovery_sink'].update(state=s['state'],cfg=copy.deepcopy(s['saved_cfg']),history=s['history'])
    elapsed=(time.monotonic_ns()-s['data']['config']['frontend_preparation']['started_monotonic_ns'])/NS_PER_SECOND
    native=s['ns']['save_checkpoint'](s['state'],s['saved_cfg'],s['history'],Path(s['root'])/'checkpoints',
        cumulative_seconds=s['data']['source']['native_manifest']['cumulative_seconds']+elapsed,
        tag='normal_lr_'+reason+'_'+nonce,keep=None)
    s['helpers']._native_local_audit(s,native)
    # Native writer already atomically renamed; make the completed bytes durable.
    with Path(native).open('rb') as stream:os.fsync(stream.fileno())
    directory=os.open(str(Path(native).parent),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    validate_training_native(s,native)
    info=_identity(Path(native).absolute());s['last_native']=info;s['durable_updates']=s['accepted_update']
    s['local_natives'].append(info);s['assignment_diagnostic']=_assignment_diagnostic(s)
    # Supersede the older cheap pending checkpoint only after complete readback.
    protected={str(s['data']['source_native']['path']),str(s['data']['config']['original_source_native_path']),info['path']}
    if s.get('last_hourly_native'):protected.add(s['last_hourly_native']['path'])
    retained=[]
    for old in s['local_natives']:
        if old['path'] in protected:retained.append(old)
        else:_remove_owned_pending(s,old,protected)
    s['local_natives']=retained;_write_status(s);return info
