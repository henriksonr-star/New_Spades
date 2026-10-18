"""Experimental Kaggle Spades AOT native continuation; definitions only, no autorun.

Exact durable update674 package; immutable learning/rules. Two explicit AOT profiles.
Kernel returns for external Library export; no Drive, network or busy waiting.
Original combined-JIT bitwise continuity failure remains unresolved.
"""

def _aot_kaggle_uuid(value):
    import re
    if not isinstance(value,str) or re.fullmatch(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',value) is None:
        raise ValueError('Explicit canonical physical NVIDIA GPU UUID required; no ordinal/autoselection.')
    return value


def _aot_kaggle_board_rows(stdout):
    import csv,math,re
    rows=list(csv.reader(stdout.strip().splitlines()))
    if not rows:raise RuntimeError('No physical GPU board telemetry.')
    result=[]
    for row in rows:
        if len(row)!=8:raise ValueError('Exact eight-field physical GPU telemetry required.')
        uuid,index,pci,name,total,used,free,driver=[item.strip() for item in row]
        _aot_kaggle_uuid(uuid)
        if (not index.isdecimal() or re.fullmatch(r'[0-9a-fA-F]{4}(?:[0-9a-fA-F]{4})?:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',pci) is None
                or re.fullmatch(r'\d+(?:\.\d+)+',driver) is None):
            raise ValueError('Invalid board ordinal/PCI/driver telemetry.')
        values=[float(item) for item in (total,used,free)]
        if (not all(math.isfinite(item) and item>=0 and item.is_integer() for item in values)
                or values[0]<=0 or values[1]>values[0] or values[2]>values[0]
                or values[1]+values[2]>values[0]):
            raise ValueError('Invalid/noncurrent physical board capacity telemetry.')
        result.append(dict(uuid=uuid,index=int(index),pci_bus_id=pci,name=name,
            total_bytes=int(values[0]*2**20),used_bytes=int(values[1]*2**20),
            free_bytes=int(values[2]*2**20),driver_version=driver))
    if len({row['uuid'] for row in result})!=len(result) or len({row['pci_bus_id'] for row in result})!=len(result):
        raise ValueError('Duplicate physical GPU UUID/PCI telemetry.')
    return result


def inventory_kaggle_nojax():
    """Read-only host inventory. No JAX import, package install or GPU workload."""
    import importlib.metadata,os,re,subprocess,sys
    if any(name=='jax' or name=='jaxlib' or name.startswith(('jax.','jaxlib.','jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Fresh kernel required before JAX/JAXlib/plugin import; no reset/reconnect.')
    if tuple(sys.version_info[:2])<(3,12):raise RuntimeError('Python>=3.12 required and recorded.')
    expected={'jax':'0.11.1','jaxlib':'0.11.1','flax':'0.12.10','optax':'0.2.8',
              'jax-cuda12-plugin':'0.11.1','jax-cuda12-pjrt':'0.11.1'}
    installed={key:importlib.metadata.version(key) for key in expected}
    if installed!=expected:raise RuntimeError('Exact Kaggle CUDA12 runtime packages required; no installer:'+str(installed))
    for name in ('jax-cuda13-plugin','jax-cuda13-pjrt'):
        try:importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:continue
        raise RuntimeError('Ambiguous additional CUDA13 JAX plugin installed; no automatic package changes.')
    if os.environ.get('JAX_SKIP_CUDA_CONSTRAINTS_CHECK') is not None:
        raise RuntimeError('CUDA constraint skip environment forbidden; no bypass.')
    if os.environ.get('JAX_CUDA_VISIBLE_DEVICES') not in (None,'all'):
        raise RuntimeError('JAX_CUDA_VISIBLE_DEVICES must be unset/all before UUID selection.')
    command=['nvidia-smi','--query-gpu=uuid,index,pci.bus_id,name,memory.total,memory.used,memory.free,driver_version',
             '--format=csv,noheader,nounits']
    query=subprocess.run(command,capture_output=True,text=True,check=False,timeout=5)
    if query.returncode!=0:raise RuntimeError('nvidia-smi physical board inventory failed; no CPU fallback.')
    boards=_aot_kaggle_board_rows(query.stdout)
    libraries={}
    for distribution in importlib.metadata.distributions():
        name=distribution.metadata.get('Name','')
        if name.lower().startswith('nvidia-') and name.lower().endswith('-cu12'):
            libraries[name]=distribution.version
    return dict(status='host_inventory_recorded',physical_gpu_inventory=boards,runtime_stack=installed,
        python_version='.'.join(str(value) for value in sys.version_info[:3]),cuda12_libraries=libraries,
        environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES',
            'JAX_CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH')},jax_imported=False,compute_allocated=False)


def preflight_kaggle_nojax(selected_gpu_uuid):
    """Read-only selected-board validation; prepare alone sets UUID visibility."""
    import re
    selected_gpu_uuid=_aot_kaggle_uuid(selected_gpu_uuid)
    inventory=inventory_kaggle_nojax();boards=inventory['physical_gpu_inventory']
    selected=[row for row in boards if row['uuid']==selected_gpu_uuid]
    if len(selected)!=1:raise RuntimeError('Selected physical GPU UUID absent/ambiguous; no fallback.')
    board=selected[0]
    if re.search(r'(?:^|\s)T4(?:$|\s)',board['name']) is None:raise RuntimeError('Selected board must be an actual T4.')
    if int(board['driver_version'].split('.')[0])<525:raise RuntimeError('CUDA12 requires NVIDIA driver>=525.')
    platform=dict(provider='kaggle',adapter='spades-kaggle-aot-v1',
        selected_gpu_uuid=selected_gpu_uuid,selected_gpu_pci_bus_id=board['pci_bus_id'],
        selected_gpu_name=board['name'],selected_gpu_total_bytes=board['total_bytes'],
        driver_version=board['driver_version'],cuda_plugin='cuda12',runtime_stack=inventory['runtime_stack'],
        python_version=inventory['python_version'])
    return dict(inventory,status='host_preflight_passed',platform=platform,numerical_equivalence=False,
        scope='Host inventory only; actual JAX device, memory admissions, update and external ACK gates remain required.')


def _aot_kaggle_identity_guard(s):
    import os
    platform=s['platform']
    if (not _aot_same_json(platform,s['saved_platform'])
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=platform['selected_gpu_uuid']
            or not _aot_same_json(s['ns'].get('_SPADES_KAGGLE_PLATFORM'),platform)):
        raise RuntimeError('Selected physical GPU environment/platform identity changed; no launch.')
    devices=s['ns']['jax'].devices()
    if (len(devices)!=1 or devices[0]!=s['profile']['device'] or devices[0].platform!='gpu'
            or 'T4' not in str(devices[0].device_kind)):
        raise RuntimeError('Exactly one original selected JAX T4 required at every launch.')



class AOTLongRunBoundaryStop(RuntimeError):
    pass


def _aot_utc(value):
    from datetime import datetime
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None or result.utcoffset().total_seconds()!=0:
        raise ValueError('Explicit UTC timestamp required.')
    return result


def _aot_file_identity(path):
    from pathlib import Path
    import hashlib
    target=Path(path);digest=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(target),bytes=target.stat().st_size,sha256=digest.hexdigest())



def _aot_validate_candidate_package(path):
    """Host-only validation, rooted in the exact externally verified674 package."""
    import hashlib,io,json,zipfile
    import numpy as np
    info=_aot_file_identity(path)
    if info['sha256']!='2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13':
        raise ValueError('Exact verified durable update674 package bytes required; no fallback.')
    with zipfile.ZipFile(path) as package:
        if sorted(package.namelist())!=['export_manifest.json','native_checkpoint.zip'] or package.testzip() is not None:
            raise ValueError('Durable package must contain exactly the two reviewed members with valid CRC.')
        external=json.loads(package.read('export_manifest.json'));native=package.read('native_checkpoint.zip')
    if (external.get('format')!='spades-aot-external-export-v1'
            or type(external.get('updates')) is not int or external['updates']!=674
            or type(external.get('env_steps')) is not int or external['env_steps']!=309198848
            or external.get('RUN_NAME')!='modified_spades_run_001'
            or external.get('reason')!='periodic90'):
        raise ValueError('Exact durable P90 update674 package provenance/counters required.')
    record=external.get('native',{})
    if (type(record.get('bytes')) is not int or record['bytes']!=len(native)
            or record.get('sha256')!=hashlib.sha256(native).hexdigest()):
        raise ValueError('Enclosed native does not match trusted external package record.')
    with zipfile.ZipFile(io.BytesIO(native)) as archive:
        if sorted(archive.namelist())!=['history.json','manifest.json','state.npz'] or archive.testzip() is not None:
            raise ValueError('Native must contain exactly three reviewed members with valid CRC.')
        manifest=json.loads(archive.read('manifest.json'));history=json.loads(archive.read('history.json'));payload=archive.read('state.npz')
    cfg=manifest['cfg']
    if (hashlib.sha256(payload).hexdigest()!=manifest['sha256']
            or manifest.get('training_revision')!='strong-teacher-hand-gae-v3' or manifest.get('training_state_schema')!=2
            or manifest.get('versions')!={'jax':'0.11.1','flax':'0.12.10','optax':'0.2.8'}
            or (cfg.get('seed'),cfg.get('n_env'),cfg.get('rollout_len'),cfg.get('width'),cfg.get('minibatch_size'),cfg.get('ppo_epochs'))!=(42,3584,128,256,57344,3)
            or type(manifest.get('updates')) is not int or manifest['updates']!=674
            or type(manifest.get('env_steps')) is not int or manifest['env_steps']!=309198848
            or not isinstance(history,list)):
        raise ValueError('Exact native674 payload/revision/schema/full saved cfg/counters/stack required.')
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        names=[f'leaf_{i:05d}' for i in range(214)]
        if sorted(arrays.files)!=names or len(manifest['leaves'])!=214:
            raise ValueError('Native214 schema required.')
        for name,spec in zip(names,manifest['leaves']):
            array=arrays[name]
            if (list(array.shape)!=spec['shape'] or str(array.dtype)!=spec['dtype'] or array.dtype.hasobject
                    or np.issubdtype(array.dtype,np.inexact) and not np.isfinite(array).all()):
                raise ValueError('Invalid/nonfinite native host leaf:'+name)
        indices=(207,208,209,210,211,212,213)
        expected=('int32','int32','int32','bool','uint32','uint32','int32')
        counters={i:arrays[f'leaf_{i:05d}'] for i in indices}
        if any(value.shape!=() or str(value.dtype)!=dtype for value,dtype in zip(counters.values(),expected)):
            raise ValueError('Native counter scalar shapes/dtypes changed.')
        values={i:value.item() for i,value in counters.items()}
        lifetime=((int(values[212])<<32)|int(values[211]))+(int(values[207])-int(values[213]))*int(values[208])
        if (values[207]!=674 or values[208]!=458752 or values[209]!=0 or values[210] is not True
                or not 0<=values[213]<=values[207] or lifetime!=309198848):
            raise ValueError('Native exact bootstrap/lifetime counters invalid.')
    return native,manifest,history


def _aot_same_json(left,right):
    import json
    return json.dumps(left,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(right,sort_keys=True,separators=(',',':'),allow_nan=False)

def _aot_operator_identity():
    return tuple((name,id(value)) for name,value in sorted(globals().items())
                 if (name.startswith('_aot_') or name in ('prepare_aot_long_run','run_until_barrier','ack_export','next_segment','inventory_kaggle_nojax','preflight_kaggle_nojax')) and callable(value))


def _aot_elapsed(s):
    import time
    return 0.0 if s['segment_started'] is None else time.monotonic()-s['segment_started']


def _aot_status(s):
    result=dict(status=s['status'],root=s['root'],run_id=s['run_id'],RUN_NAME='modified_spades_run_001',
        updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
        elapsed_segment_seconds=_aot_elapsed(s),remaining_segment_seconds=max(0.,10800.-_aot_elapsed(s)),
        segment_number=s['segment_number'],max_segments=s['coordination']['max_segments'],
        supplemental_export_interval_seconds=900,next_supplemental_elapsed=s['next_export_elapsed'],
        next_periodic90_elapsed=s['next_checkpoint_elapsed'],
        restoration=s['restoration'],
        terminal=s['terminal'],stop_reason=s.get('stop_reason'),
        original_combined_JIT_continuity='failed; unresolved',route='experimental_sequential_AOT',
        platform=dict(s['platform'],runtime_stack=dict(s['platform']['runtime_stack'])),
        promotion=False,evaluation_calls=0,last_native=s.get('last_native'),
        observed_optimizer_totals=s['ns']['_optimizer_totals'](s['journal_rows']))
    if s.get('pending_export') is not None:
        result.update(s['pending_export']);result['status']='export_required';result['terminal']=s['terminal']
    if s.get('error') is not None:result['error']=s['error']
    return result


def _aot_write(s):
    import json
    from pathlib import Path
    status=_aot_status(s)
    status['recent_calls']=s['recent_calls'][-8:]
    status['last_metrics']=s.get('last_metrics')
    target=Path(s['root'])/'status.json';temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');temporary.replace(target)
    return status


def _aot_event(s,kind,**values):
    from datetime import datetime,timezone
    from pathlib import Path
    import json
    row=dict(kind=kind,utc=datetime.now(timezone.utc).isoformat(),elapsed_segment_seconds=_aot_elapsed(s),**values)
    with (Path(s['root'])/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def _aot_limit_reason(s):
    from datetime import datetime,timezone
    if datetime.now(timezone.utc)>=_aot_utc(s['coordination']['authorization_valid_until_utc']):
        return 'authorization_expired'
    if s['segment_started'] is not None and _aot_elapsed(s)>=10800.:
        return 'segment_complete'
    return None


def _aot_identity_guard(s):
    _aot_kaggle_identity_guard(s)
    ns=s['ns']
    if (s['operator_identity']!=_aot_operator_identity() or s['runtime_identity']!=ns['_gpu_runtime_identity']()
            or not _aot_same_json(s['cfg'],s['saved_cfg']) or s['handles']!=s['saved_handles']
            or not _aot_same_json(s['coordination'],s['saved_coordination'])):
        raise RuntimeError('Operator/source/runtime/full saved cfg or AOT handles changed; no retry.')
    if any(s['options'].get(k)!=v for k,v in dict(target_gib=14.,reserve_mib=1024,safety_factor=1.45,extra_guard_mib=256).items()):
        raise RuntimeError('Reviewed exact memory guard settings changed.')


def _aot_boundary(s,stage):
    _aot_identity_guard(s)
    reason=_aot_limit_reason(s)
    if reason:raise AOTLongRunBoundaryStop(reason+':'+stage)
    if s['terminal'] or s.get('pending_export') is not None:
        raise AOTLongRunBoundaryStop('Export barrier/terminal state forbids new workload:'+stage)


def _aot_input_specs(jax,np,tree):
    import math
    pairs,_=jax.tree_util.tree_flatten_with_path(tree)
    return [dict(path=jax.tree_util.keystr(path),shape=list(value.shape),dtype=str(np.dtype(value.dtype)),
                 bytes=math.prod(value.shape)*np.dtype(value.dtype).itemsize) for path,value in pairs]


def _aot_resident_credit(s,arguments,snapshot):
    """Credit ONE proven dense resident argument; no cross-leaf alias claims."""
    import hashlib,json,math,numbers
    jax,np=s['ns']['jax'],s['ns']['np'];wrapped=(arguments,{})
    pairs,_=jax.tree_util.tree_flatten_with_path(wrapped)
    specs=_aot_input_specs(jax,np,wrapped)
    expected=s['profile']['optimize_input_specs']
    argument_bytes=int(s['profile']['optimize']['argument_size_in_bytes'])
    if len(pairs)!=225 or specs!=expected or sum(row['bytes'] for row in specs)!=argument_bytes:
        raise RuntimeError('Actual225 optimizer input specs/payload do not match compiled argument accounting.')
    formats,_=jax.tree_util.tree_flatten_with_path(s['handles']['optimize'].input_formats)
    if len(formats)!=225 or [jax.tree_util.keystr(path) for path,_ in formats]!=[row['path'] for row in specs]:
        raise RuntimeError('Compiled expected optimizer input formats unavailable/mismatched.')
    index=max(range(len(specs)),key=lambda i:specs[i]['bytes'])
    path,value=pairs[index];spec=specs[index];expected_format=formats[index][1]
    if not isinstance(value,jax.Array):raise RuntimeError('Resident credit requires an actual JAX device array, not host/abstract data.')
    device=s['profile']['device']
    devices=value.devices()
    if (devices!={device} or device.platform!='gpu' or 'T4' not in str(device.device_kind)
            or value.is_fully_addressable is not True or value.is_fully_replicated is not True
            or not isinstance(value.sharding,jax.sharding.SingleDeviceSharding)
            or value.sharding.memory_kind!='device'):
        raise RuntimeError('Largest argument is not a single fully resident T4 device-memory shard.')
    layout=value.format.layout
    if (value.format!=expected_format or tuple(layout.tiling or ())!=() or layout.sub_byte_element_size_in_bits!=0
            or sorted(layout.major_to_minor)!=list(range(len(value.shape)))
            or int(value.nbytes)!=spec['bytes'] or spec['bytes']<=0):
        raise RuntimeError('Largest argument layout/density does not exactly match compiled expected format.')
    query=getattr(value,'on_device_size_in_bytes',None)
    if not callable(query):raise RuntimeError('Actual resident argument byte query unavailable.')
    resident=query();live=snapshot.get('jax_live_bytes')
    if (not isinstance(resident,numbers.Integral) or isinstance(resident,bool) or resident<spec['bytes']
            or not isinstance(live,numbers.Integral) or isinstance(live,bool) or live<spec['bytes']):
        raise RuntimeError('Actual resident argument/live-byte evidence insufficient.')
    return dict(credit_bytes=spec['bytes'],path=spec['path'],shape=spec['shape'],dtype=spec['dtype'],
        on_device_size_in_bytes=int(resident),current_live_bytes=int(live),device=str(device),format=str(value.format),
        argument_payload_bytes=argument_bytes,input_specs_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
        scope='ONE actual dense resident argument; all other arguments remain charged; no union/alias/workspace claim.')


def _aot_admit(s,stage,arguments):
    _aot_boundary(s,stage+' before admission')
    snapshot=s['ns']['_gpu_memory_snapshot']()
    reference=int(s['profile'][stage]['total_bytes'])+s['profile']['state_bytes']+65536
    credit=_aot_resident_credit(s,arguments,snapshot) if stage=='optimize' else dict(credit_bytes=0,scope='No argument credit for collect.')
    estimate=reference-credit['credit_bytes']
    outcome=s['ns']['_gpu_memory_admission'](estimate,s['options'],stage,snapshot=snapshot)
    details=outcome.get('budget_details',{});checked=outcome.get('snapshot',{})
    if (outcome.get('status')!='admitted' or outcome.get('overlap_credit_bytes')!=0
            or checked.get('platform')!='gpu' or 'pool_bytes' not in checked.get('jax_memory_stats_raw',{})
            or details.get('pool_counter_source')!='pool_bytes' or details.get('pool_counter_status')!='validated_current_total'):
        raise RuntimeError('Current raw pool/board capacity is required.')
    outcome=dict(outcome,uncredited_whole_reference_bytes=reference,resident_input_credit=credit,
        wrapper_accounting_policy='phase-specific whole compiler profile plus rollback minus ONE proven resident dense optimizer argument',
        legacy_helper_overlap_credit_bytes=0)
    _aot_boundary(s,stage+' after admission')
    return outcome


def _aot_host_available_bytes():
    from pathlib import Path
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Available host RAM telemetry unavailable.')


def _aot_host_finite(s,tree,stage,expected_leaves,payload_guard_bytes):
    np,jax=s['ns']['np'],s['ns']['jax']
    required=2*int(payload_guard_bytes)+512*1024**2
    available=_aot_host_available_bytes()
    if available<required:
        raise MemoryError('Host finite validation requires '+str(required)+' bytes; available '+str(available)+'. Accepted state retained.')
    leaves=jax.tree_util.tree_leaves(tree)
    if len(leaves)!=expected_leaves:raise ValueError('Unexpected source leaf count at '+stage)
    # No full-tree deepcopy or retained host batch. JAX may cache host copies;
    # whole-payload RAM admission above accounts for that until batch release.
    for index,leaf in enumerate(leaves):
        array=np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject or (np.issubdtype(array.dtype,np.inexact) and not np.isfinite(array).all()):
            raise FloatingPointError('Invalid/nonfinite '+stage+' leaf '+str(index))
        del array


def _aot_sync(s,tree):
    for leaf in s['ns']['jax'].tree_util.tree_leaves(tree):
        if hasattr(leaf,'block_until_ready'):leaf.block_until_ready()


def _aot_launch(s,kind,function,arguments):
    import time
    admission=_aot_admit(s,kind,arguments)
    row=dict(kind=kind,admission=admission,before=s['ns']['_gpu_memory_snapshot'](),started=False,synchronized=False)
    s['recent_calls'].append(row);s['recent_calls']=s['recent_calls'][-8:]
    _aot_event(s,'operation_prelaunch',operation=row);_aot_write(s)
    _aot_heartbeat(s,'prelaunch',phase=kind)
    _aot_boundary(s,kind+' final launch after telemetry/report IO')
    begin=time.monotonic();row['started']=True
    result=function()
    s['active_device_result']=result
    s['recovery_sink']['active_device_result']=result
    _aot_sync(s,result)
    row.update(synchronized=True,seconds=time.monotonic()-begin,completed_after_boundary=_aot_limit_reason(s))
    # No postoptimizer IO here: the validated candidate must commit first.
    return result,row


def _aot_new_session(*,ns,state,cfg,history,profile,coordination,output_dir,recovery_sink,root_precreated=False):
    from pathlib import Path
    import copy,uuid
    root=Path(output_dir)
    if root_precreated:
        if not root.is_dir() or (root/'checkpoints').exists():raise RuntimeError('Unique precreated preparation root required.')
    else:root.mkdir(parents=True,exist_ok=False)
    (root/'checkpoints').mkdir()
    journal=ns['_optimizer_resume_journal'](history,cfg,state.updates)
    s=dict(platform=copy.deepcopy(ns['_SPADES_KAGGLE_PLATFORM']),saved_platform=copy.deepcopy(ns['_SPADES_KAGGLE_PLATFORM']),ns=ns,state=state,cfg=dict(cfg),saved_cfg=copy.deepcopy(cfg),history=history,
        profile=profile,options=dict(profile['options']),handles=dict(profile['execution_handles']),
        saved_handles=dict(profile['execution_handles']),runtime_identity=ns['_gpu_runtime_identity'](),
        coordination=copy.deepcopy(coordination),saved_coordination=copy.deepcopy(coordination),root=str(root),run_id=uuid.uuid4().hex,
        accepted_update=int(state.updates),accepted_env_steps=state.env_steps,segment_started=None,next_checkpoint_elapsed=5400.,
        next_export_elapsed=900.,segment_number=1,completed_segment_seconds=0.,
        restoration=dict(source_update=674,source_env_steps=309198848,previous_last_observed_update=1075,
            previous_last_observed_env_steps=493158400,lost_unexported_updates=401,lost_unexported_decisions=183959552,
            new_authorized_session=True,old_segment_clock_recovered=False,playing_strength_evaluated=False),
        pending_export=None,resume_grant=None,
        terminal=False,status='prepared',recent_calls=[],journal_rows=journal,exports=[],recovery_sink=recovery_sink)
    s['operator_identity']=_aot_operator_identity()
    recovery_sink.update(state=state,cfg=copy.deepcopy(cfg),history=history,session=s)
    return s


def _aot_heartbeat(s,event,**values):
    import json
    row=dict(event=event,update=s['accepted_update'],env_steps=s['accepted_env_steps'],
             elapsed_segment_seconds=_aot_elapsed(s),**values)
    try:print(json.dumps(row,allow_nan=False),flush=True)
    except (OSError,ValueError) as exc:s['heartbeat_error']=type(exc).__name__+': '+str(exc)


def _aot_audit_native(s,path):
    import hashlib,io,json,zipfile
    np=s['ns']['np']
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise IOError('Native checkpoint CRC failure.')
        manifest=json.loads(archive.read('manifest.json'));history=json.loads(archive.read('history.json'));payload=archive.read('state.npz')
    if (hashlib.sha256(payload).hexdigest()!=manifest['sha256'] or manifest['cfg']!=s['saved_cfg']
            or manifest['updates']!=s['accepted_update'] or manifest['env_steps']!=s['accepted_env_steps']
            or manifest.get('training_revision')!='strong-teacher-hand-gae-v3' or manifest.get('training_state_schema')!=2
            or manifest.get('versions')!={'jax':'0.11.1','flax':'0.12.10','optax':'0.2.8'}):
        raise IOError('Native readback payload/config/counters/version mismatch.')
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        names=[f'leaf_{i:05d}' for i in range(214)]
        if sorted(arrays.files)!=names or len(manifest['leaves'])!=214:raise IOError('Native214 readback schema mismatch.')
        for name,spec in zip(names,manifest['leaves']):
            array=arrays[name]
            if (list(array.shape)!=spec['shape'] or str(array.dtype)!=spec['dtype'] or array.dtype.hasobject
                    or np.issubdtype(array.dtype,np.inexact) and not np.isfinite(array).all()):
                raise IOError('Native readback invalid leaf:'+name)
        counters={i:arrays[f'leaf_{i:05d}'] for i in (207,208,209,210,211,212,213)}
        expected_dtypes=('int32','int32','int32','bool','uint32','uint32','int32')
        if any(value.shape!=() or str(value.dtype)!=dtype for value,dtype in zip(counters.values(),expected_dtypes)):
            raise IOError('Native readback scalar counter schema mismatch.')
        values={i:value.item() for i,value in counters.items()}
        decisions=int(s['saved_cfg']['n_env'])*int(s['saved_cfg']['rollout_len'])
        lifetime=((int(values[212])<<32)|int(values[211]))+(int(values[207])-int(values[213]))*int(values[208])
        if (values[207]!=s['accepted_update'] or values[208]!=decisions or values[209]!=0 or values[210] is not True
                or not 0<=values[213]<=values[207] or lifetime!=s['accepted_env_steps']):
            raise IOError('Native readback actual scalar counters disagree with accepted state.')
    if history!=s['history']:raise IOError('Native history changed on readback.')
    return manifest


def _aot_checkpoint_barrier(s,reason,terminal=False):
    from pathlib import Path
    from datetime import datetime,timezone
    import hashlib,json,time,uuid,zipfile
    # Accepted state is retained before any checkpoint/package/report IO.
    s['terminal']=bool(terminal);s['status']='saving_export_barrier';s['stop_reason']=reason if terminal else None
    s['recovery_sink'].update(state=s['state'],cfg=dict(s['cfg']),history=s['history'],session=s)
    s['history'].append(dict(kind='aot_export_barrier',reason=reason,update=s['accepted_update'],
        elapsed_segment_seconds=_aot_elapsed(s),terminal=bool(terminal),no_promotion=True))
    native=s['ns']['save_checkpoint'](s['state'],s['cfg'],s['history'],Path(s['root'])/'checkpoints',
        cumulative_seconds=s['completed_segment_seconds']+_aot_elapsed(s),tag='aot_kaggle_v1_s'+str(s['segment_number'])+'_'+reason,keep=None)
    _aot_audit_native(s,native)
    native_info=_aot_file_identity(native);s['last_native']=native_info
    s['recovery_sink']['native_checkpoint']=native_info
    nonce=uuid.uuid4().hex
    manifest=dict(format='spades-aot-external-export-kaggle-v1',nonce=nonce,run_id=s['run_id'],reason=reason,
        platform=dict(s['platform'],runtime_stack=dict(s['platform']['runtime_stack'])),
        native=native_info,updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
        RUN_NAME='modified_spades_run_001',source_pins=s['coordination']['source_pins'],
        candidate_label='durable update674 experimental AOT continuation; unpromoted',
        segment_number=s['segment_number'],max_segments=s['coordination']['max_segments'],restoration=s['restoration'],
        original_JIT_continuity='failed; unresolved')
    package=Path(s['root'])/'export_package.zip';temporary=Path(s['root'])/'export_package.zip.tmp'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.write(native,'native_checkpoint.zip')
        archive.writestr('export_manifest.json',json.dumps(manifest,indent=2,allow_nan=False))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise IOError('Export package CRC failure.')
        digest=hashlib.sha256();size=0
        with archive.open('native_checkpoint.zip') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);size+=len(block)
        if digest.hexdigest()!=native_info['sha256'] or size!=native_info['bytes']:
            raise IOError('Export package native payload mismatch.')
    temporary.replace(package);package_info=_aot_file_identity(package)
    pending=dict(nonce=nonce,reason=reason,terminal=bool(terminal),package_path=package_info['path'],
        package_sha256=package_info['sha256'],package_bytes=package_info['bytes'],
        native_path=native_info['path'],native_sha256=native_info['sha256'],native_bytes=native_info['bytes'],
        created_utc=datetime.now(timezone.utc).isoformat(),created_monotonic=time.monotonic(),
        updates=int(s['state'].updates),env_steps=s['state'].env_steps)
    s['pending_export']=pending;s['resume_grant']=None;s['status']='export_required'
    if reason=='periodic90':s['next_checkpoint_elapsed']=(int(_aot_elapsed(s)//5400.)+1)*5400.
    s['next_export_elapsed']=_aot_elapsed(s)+900.
    _aot_event(s,'external_export_required',**pending)
    status=_aot_write(s)
    _aot_heartbeat(s,'export_required',nonce=nonce,reason=reason,package_bytes=pending['package_bytes'],package_sha256=pending['package_sha256'],package_path=pending['package_path'])
    return status


def ack_export(s,receipt):
    """Explicit host-only verified Library receipt; returns with kernel idle."""
    from datetime import datetime,timezone
    import json,time
    from pathlib import Path
    _aot_identity_guard(s)
    pending=s.get('pending_export')
    if pending is None:raise ValueError('No current export nonce; stale/replayed acknowledgement.')
    if time.monotonic()-pending['created_monotonic']>=s['coordination']['export_wait_timeout_seconds']:
        s.update(terminal=True,status='export_wait_expired',stop_reason='export_wait_expired');_aot_write(s)
        raise RuntimeError('Export wait expired; accepted checkpoint retained; no automatic resume.')
    if not isinstance(receipt,dict) or receipt.get('verified') is not True:
        raise ValueError('Parent verified external Library receipt required.')
    for key in ('nonce','package_path','package_sha256','package_bytes','native_path','native_sha256','native_bytes'):
        if type(receipt.get(key)) is not type(pending[key]) or receipt[key]!=pending[key]:
            raise ValueError('Stale/wrong export receipt field:'+key)
    if not isinstance(receipt.get('library_receipt'),str) or not receipt['library_receipt'].strip():
        raise ValueError('Nonempty verified Library receipt required.')
    expiry=_aot_utc(receipt['valid_until_utc']);now=datetime.now(timezone.utc)
    if not now<expiry<=_aot_utc(s['coordination']['authorization_valid_until_utc']):
        raise ValueError('Receipt expired or exceeds run authorization.')
    for prefix in ('package','native'):
        actual=_aot_file_identity(pending[prefix+'_path'])
        if actual['bytes']!=pending[prefix+'_bytes'] or actual['sha256']!=pending[prefix+'_sha256']:
            raise ValueError('Local exported '+prefix+' bytes changed.')
    _aot_identity_guard(s)
    now=datetime.now(timezone.utc)
    if (not now<expiry<=_aot_utc(s['coordination']['authorization_valid_until_utc'])
            or time.monotonic()-pending['created_monotonic']>=s['coordination']['export_wait_timeout_seconds']):
        raise ValueError('Receipt/authorization/export wait expired during acknowledgement hash IO.')
    if (s['accepted_update']!=pending['updates'] or s['accepted_env_steps']!=pending['env_steps']
            or int(s['state'].updates)!=pending['updates'] or s['state'].env_steps!=pending['env_steps']):
        raise ValueError('Pending export does not bind the current accepted state.')
    committed=dict(receipt,acknowledged_utc=now.isoformat(),reason=pending['reason'])
    target=Path(s['root'])/('export_ack_'+pending['nonce']+'.json')
    temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(committed,indent=2));temporary.replace(target)
    s['exports'].append(committed);s['history'].append(dict(kind='external_export_ack',**committed))
    s['pending_export']=None;s['resume_grant']=dict(committed,updates=pending['updates'],env_steps=pending['env_steps'])
    s['status']='terminal_exported' if s['terminal'] else 'ready_to_resume'
    # The latest accepted state is now externally durable: avoid catch-up
    # snapshots after a long ACK wait. Absolute90/180min milestones do not move.
    s['next_export_elapsed']=_aot_elapsed(s)+900.
    try:return _aot_write(s)
    except BaseException:
        s.update(terminal=True,status='ack_report_failed',stop_reason='ack_report_failed');raise



def _aot_verify_resume_grant(s):
    from datetime import datetime,timezone
    grant=s.get('resume_grant')
    if grant is None:raise RuntimeError('Exact current external export acknowledgement required before resume.')
    _aot_identity_guard(s)
    if datetime.now(timezone.utc)>=_aot_utc(grant['valid_until_utc']):
        raise ValueError('Acknowledgement expired before resume.')
    for prefix in ('package','native'):
        actual=_aot_file_identity(grant[prefix+'_path'])
        if actual['bytes']!=grant[prefix+'_bytes'] or actual['sha256']!=grant[prefix+'_sha256']:
            raise ValueError('Acknowledged local '+prefix+' changed before resume.')
    if (grant['updates']!=s['accepted_update'] or grant['env_steps']!=s['accepted_env_steps']
            or int(s['state'].updates)!=s['accepted_update'] or s['state'].env_steps!=s['accepted_env_steps']
            or not s['exports'] or s['exports'][-1]['nonce']!=grant['nonce']):
        raise ValueError('Acknowledgement does not bind the current accepted state.')
    now=datetime.now(timezone.utc)
    if now>=_aot_utc(grant['valid_until_utc']) or now>=_aot_utc(s['coordination']['authorization_valid_until_utc']):
        raise ValueError('Acknowledgement/authorization expired during resume hash IO.')
    return grant


def next_segment(s):
    """Explicit host-only arming after a durable final; never automatically runs."""
    import time
    if (not s['terminal'] or s.get('pending_export') is not None
            or s['status']!='terminal_exported' or s.get('stop_reason')!='segment_complete'
            or s.get('error') is not None or s.get('last_health',{}).get('should_stop',False)):
        raise RuntimeError('Only a healthy externally acknowledged segment_complete final can arm another segment.')
    if s['segment_number']>=s['coordination']['max_segments']:
        raise RuntimeError('Fixed authorized segment limit reached; no automatic extension.')
    grant=_aot_verify_resume_grant(s)
    if grant.get('reason')!='segment_complete':
        raise RuntimeError('Current receipt does not acknowledge the exact three-hour final.')
    if s['segment_started'] is None or _aot_elapsed(s)<10800.:
        raise RuntimeError('Previous segment has not completed its original three-hour budget.')
    previous=s['segment_number'];s['completed_segment_seconds']+=10800.
    s.update(segment_number=previous+1,segment_started=None,next_checkpoint_elapsed=5400.,next_export_elapsed=900.,
             terminal=False,status='ready_to_resume',stop_reason=None)
    s['history'].append(dict(kind='aot_explicit_next_segment',previous_segment=previous,
        segment_number=s['segment_number'],updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
        final_receipt_nonce=grant['nonce'],max_segments=s['coordination']['max_segments'],
        automatic=False,no_promotion=True))
    try:
        _aot_event(s,'next_segment_armed',segment_number=s['segment_number'],final_receipt_nonce=grant['nonce'])
        return _aot_write(s)
    except BaseException:
        s.update(terminal=True,status='segment_arm_report_failed',stop_reason='segment_arm_report_failed')
        raise

def _aot_step(s):
    import copy,math
    ns=s['ns'];previous=s['state'];cfg=s['cfg']
    collected,collect_row=_aot_launch(s,'collect',lambda:s['handles']['collect'](previous),(previous,))
    collected_state,batch,rollout_metrics=collected
    _aot_host_finite(s,collected,'collection',273,s['profile']['state_bytes']+s['profile']['batch_bytes'])
    scalar_collect=ns['_scalar_metrics'](rollout_metrics)
    if scalar_collect.get('illegal_actions')!=0 or not all(math.isfinite(v) for v in scalar_collect.values()):
        raise RuntimeError('Illegal/invalid collection; optimizer not launched.')
    if int(collected_state.updates)!=int(previous.updates) or collected_state.env_steps!=previous.env_steps:
        raise RuntimeError('Collection partial-state counters changed.')
    _aot_boundary(s,'optimizer after full host collection validation')
    result,optimize_row=_aot_launch(s,'optimize',lambda:s['handles']['optimize'](collected_state,batch),(collected_state,batch))
    candidate,raw_metrics=result
    _aot_host_finite(s,result,'optimizer',238,s['profile']['state_bytes'])
    work=ns['_optimizer_work_record'](previous,candidate,cfg,raw_metrics)
    scalar_opt=ns['_scalar_metrics'](raw_metrics)
    required=('finite','diagnostics_finite','gradients_finite','grad_norm_finite','params_finite','optimizer_finite')
    if (any(scalar_opt.get(k)!=1. for k in required) or work['rejected']!=0
            or not 1<=work['accepted']==work['attempted'] or not all(math.isfinite(v) for v in scalar_opt.values())):
        raise RuntimeError('Rejected/nonfinite optimizer candidate; accepted rollback retained.')
    # Validated state commits BEFORE report/checkpoint IO. History is host only.
    s['state']=candidate;s['accepted_update']=int(candidate.updates);s['accepted_env_steps']=candidate.env_steps;s['history'].append(copy.deepcopy(work));s['journal_rows'].append(copy.deepcopy(work))
    s['history'].append(dict(kind='aot_raw_metrics',update=int(candidate.updates),collection=scalar_collect,optimizer=scalar_opt))
    s['last_metrics']=dict(update=int(candidate.updates),collection=scalar_collect,optimizer=scalar_opt,work=work)
    s['recovery_sink'].update(state=candidate,cfg=dict(cfg),history=s['history'],session=s)
    s.pop('active_device_result',None);s['recovery_sink'].pop('active_device_result',None)
    health=ns['_optimizer_health'](s['journal_rows']);s['last_health']=health
    if health['should_stop']:
        signature=ns['_optimizer_config_signature'](cfg)
        import hashlib,json
        stop_id=hashlib.sha256(json.dumps(dict(update=int(candidate.updates),signature=signature,health=health),sort_keys=True).encode()).hexdigest()
        s['history'].append(dict(kind='optimizer_health_stop',optimizer_health_version=1,update=int(candidate.updates),
            config_signature=signature,stop_id=stop_id,reason='Three consecutive KL-stopped rollouts accepted less than25percent of planned work.',health=health))
        return 'optimizer_health_stop'
    # Leaving this function releases old accepted/collection/batch/result locals
    # before another collect or checkpoint. No raw device arrays enter history.
    return None


def run_until_barrier(s,max_updates=None):
    """Return on export barrier; NEVER wait for browser/ACK inside a busy cell."""
    from datetime import datetime,timezone
    import time,traceback
    if s.get('pending_export') is not None:return _aot_write(s)
    if s['terminal']:return _aot_write(s)
    if max_updates is not None and (type(max_updates) is not int or max_updates<1):raise ValueError('Positive optional validation update bound required.')
    try:
        _aot_verify_resume_grant(s)
    except BaseException as exc:
        s.update(terminal=True,status='resume_verification_failed',stop_reason='resume_verification_failed',error=dict(type=type(exc).__name__,message=str(exc)))
        _aot_write(s);return _aot_status(s)
    s['resume_grant']=None
    if s['segment_started'] is None:s['segment_started']=time.monotonic()
    s['status']='running';start_update=s['accepted_update']
    try:
        while True:
            _aot_identity_guard(s)
            reason=_aot_limit_reason(s)
            if reason:return _aot_checkpoint_barrier(s,reason,terminal=True)
            # Export deadlines crossed while browser was uploading are checked
            # before another collect or optimizer workload can launch.
            if _aot_elapsed(s)>=s['next_checkpoint_elapsed']:
                return _aot_checkpoint_barrier(s,'periodic90',terminal=False)
            if _aot_elapsed(s)>=s['next_export_elapsed']:
                return _aot_checkpoint_barrier(s,'supplemental15',terminal=False)
            health_stop=_aot_step(s)
            if health_stop:return _aot_checkpoint_barrier(s,health_stop,terminal=True)
            _aot_heartbeat(s,'accepted_update',accepted=s['last_metrics']['work']['accepted'],planned=s['last_metrics']['work']['planned'],loss=s['last_metrics']['optimizer'].get('loss'))
            reason=_aot_limit_reason(s)
            if reason:return _aot_checkpoint_barrier(s,reason,terminal=True)
            if _aot_elapsed(s)>=s['next_checkpoint_elapsed']:
                return _aot_checkpoint_barrier(s,'periodic90',terminal=False)
            if _aot_elapsed(s)>=s['next_export_elapsed']:
                return _aot_checkpoint_barrier(s,'supplemental15',terminal=False)
            if max_updates is not None and s['accepted_update']-start_update>=max_updates:
                return _aot_checkpoint_barrier(s,'validation',terminal=False)
            _aot_event(s,'accepted_update',metrics=s['last_metrics'],health=s['last_health'])
            _aot_write(s)
    except BaseException as exc:
        reason=(str(exc).split(':',1)[0] if isinstance(exc,AOTLongRunBoundaryStop) and str(exc).split(':',1)[0] in ('segment_complete','authorization_expired') else 'error')
        if reason in ('segment_complete','authorization_expired'):
            s['boundary_diagnostic']=dict(type=type(exc).__name__,message=str(exc))
            s.pop('active_device_result',None);s['recovery_sink'].pop('active_device_result',None)
        else:s['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        s['terminal']=True;s['status']='stopped_at_boundary' if reason!='error' else 'stopped_first_error';s['stop_reason']=reason
        try:return _aot_checkpoint_barrier(s,reason,terminal=True)
        except BaseException as save_error:
            s.setdefault('error',dict(type=type(save_error).__name__,message='Checkpoint/package preservation failed after '+reason))['checkpoint_or_package_error']=type(save_error).__name__+': '+str(save_error)
            s['status']='save_or_package_failed';s['recovery_sink'].update(state=s['state'],cfg=dict(s['cfg']),history=s['history'],session=s)
            try:_aot_write(s)
            except BaseException:pass
            return _aot_status(s)


def prepare_aot_long_run(*,original_path,production_path,candidate_package_path,coordination,output_dir,recovery_sink):
    """One fresh exact-native AOT preparation; no installer/teacher/real init."""
    from datetime import datetime,timezone
    from pathlib import Path
    import copy,hashlib,importlib.metadata,io,json,math,os,subprocess,sys,time,traceback,zipfile
    if not isinstance(recovery_sink,dict) or recovery_sink:
        raise ValueError('Supply a new empty explicit recovery_sink.')
    pins=dict(original_sha256='56da4fa7bc6e501249cecbd4e41768b27f3930b09fc8938c0fb8f1201a19c28e',
        production_sha256='16ee24065655d9d13d13896befe50772c65aa7dbcfda85e336868f42cb7ea494',
        candidate_package_sha256='2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13')
    stack=dict(jax='0.11.1',jaxlib='0.11.1',flax='0.12.10',optax='0.2.8')
    if any(coordination.get(k) is not True for k in ('run_authorized','parent_review_complete','runtime_exclusive','candidate_native_export_verified')):
        raise RuntimeError('Parent internal review/exclusive/durable-native coordination required; existing user authorization suffices.')
    if coordination.get('source_pins')!=pins:raise RuntimeError('Exact original/production/native source pins required.')
    receipt=coordination.get('candidate_external_receipt',{})
    if (receipt.get('library_receipt')!='LOCAL_VERIFIED_UPDATE674'
            or receipt.get('archive_sha256')!=pins['candidate_package_sha256']):
        raise RuntimeError('Verified immutable durable update674 external package receipt required.')
    prep_expiry=_aot_utc(coordination['prepare_valid_until_utc'])
    if prep_expiry>datetime(2026,10,19,15,49,14,tzinfo=timezone.utc):
        raise ValueError('Fixed launch window ends2026-10-19T15:49:14Z; narrow Kaggle migration authorized2026-10-18T22:33:33Z.')
    auth_expiry=_aot_utc(coordination['authorization_valid_until_utc'])
    if auth_expiry<=datetime.now(timezone.utc):raise RuntimeError('Run authorization expired.')
    if auth_expiry<prep_expiry:raise ValueError('Run authorization must cover the entire preparation window.')
    if type(coordination.get('max_segments')) is not int or not 1<=coordination['max_segments']<=8:
        raise ValueError('Explicit max_segments must be1..8; never automatic segment continuation.')
    if coordination.get('supplemental_export_interval_seconds')!=900 or type(coordination.get('supplemental_export_interval_seconds')) is not int:
        raise ValueError('Reviewed supplemental export interval must remain900seconds.')
    wait=coordination.get('export_wait_timeout_seconds')
    if type(wait) is not int or not 1<=wait<=3600:raise ValueError('Bounded export wait must be1..3600seconds.')
    platform_preflight=preflight_kaggle_nojax(coordination.get('selected_gpu_uuid'))
    selected_platform=platform_preflight['platform']
    installed={name:importlib.metadata.version(name) for name in stack}
    if installed!=stack:raise RuntimeError('Host stack mismatch; no installer is run:'+str(installed))
    if any(os.environ.get(key) is not None for key in ('XLA_FLAGS','CUBLAS_WORKSPACE_CONFIG','TF_DETERMINISTIC_OPS','JAX_DEFAULT_MATMUL_PRECISION')):
        raise RuntimeError('Original unset compiler/determinism/precision environment required; no flags changed.')
    for key,allowed,default in (('XLA_PYTHON_CLIENT_PREALLOCATE',('false',),'false'),
                                ('XLA_PYTHON_CLIENT_MEM_FRACTION',('0.96','.96'),'0.96')):
        if key in os.environ and os.environ[key] not in allowed:raise RuntimeError('Allocator environment mismatch:'+key)
        if key not in os.environ:os.environ[key]=default
    def guard(label):
        if datetime.now(timezone.utc)>=prep_expiry:raise RuntimeError('Preparation expiry before '+label+'; no further operation.')
    guard('host input validation')
    notebooks=[]
    for key,name in (('original_sha256',original_path),('production_sha256',production_path),('candidate_package_sha256',candidate_package_path)):
        info=_aot_file_identity(name)
        if info['sha256']!=pins[key]:raise RuntimeError('Pinned bytes mismatch:'+key)
        if key!='candidate_package_sha256':notebooks.append(json.loads(Path(name).read_text()))
    original,production=notebooks
    if len(original['cells'])!=38 or len(production['cells'])!=38:raise RuntimeError('Expected immutable38-cell notebooks.')
    if any(original['cells'][i]['source']!=production['cells'][i]['source'] for i in (6,8,10)):
        raise RuntimeError('Learning/rules/opponents source differs.')
    # The exact immutable external PACKAGE is the cryptographic root of trust;
    # derive its enclosed native identity without inventing an unavailable hash.
    native_bytes,manifest,history=_aot_validate_candidate_package(candidate_package_path)
    cfg=manifest['cfg']
    import numpy as np
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    checkpoint_path=root/'input_native_update674.zip'
    temporary=root/'input_native_update674.zip.tmp';temporary.write_bytes(native_bytes);temporary.replace(checkpoint_path)
    derived_native=_aot_file_identity(checkpoint_path);del native_bytes
    prep=dict(status='preparing',source_pins=pins,stack=installed,platform=selected_platform,platform_preflight=platform_preflight,stages=[],real_initialization_calls=0,teacher_calls=0,explicit_AOT_compiles=0)
    def write_prep():
        temporary=root/'preparation.json.tmp';temporary.write_text(json.dumps(prep,indent=2,allow_nan=False));temporary.replace(root/'preparation.json')
    write_prep()
    try:
        guard('selected physical UUID before JAX import')
        if any(name=='jax' or name=='jaxlib' or name.startswith(('jax.','jaxlib.','jax_plugins.')) for name in sys.modules):
            raise RuntimeError('JAX imported after host preflight; UUID visibility must be selected before any JAX import.')
        os.environ['CUDA_VISIBLE_DEVICES']=selected_platform['selected_gpu_uuid']
        prep['selected_CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
        write_prep()
        guard('JAX import')
        import jax
        import jax.numpy as jnp
        import flax
        from flax import linen as nn
        import optax
        if ({'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),'flax':flax.__version__,'optax':optax.__version__}!=stack
                or jax.config.jax_enable_x64 or jax.config.jax_disable_jit or jax.config.jax_default_matmul_precision is not None):
            raise RuntimeError('Imported stack/config/precision mismatch; no changes.')
        guard('device discovery')
        devices=jax.devices()
        if len(devices)!=1 or jax.default_backend()!='gpu' or 'T4' not in str(devices[0].device_kind):raise RuntimeError('One actual T4 required.')
        prep['plugins']={}
        for package in ('jax-cuda12-plugin','jax-cuda12-pjrt','jax-cuda13-plugin','jax-cuda13-pjrt'):
            try:prep['plugins'][package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:pass
        guard('UUID-bound runtime definitions')
        ns=dict(__name__='spades_kaggle_aot_long_run_isolated',Path=Path,os=os,time=time,json=json,math=math,np=np,jax=jax,jnp=jnp,flax=flax,nn=nn,optax=optax,
            _SPADES_KAGGLE_PLATFORM=copy.deepcopy(selected_platform))
        exec(compile(''.join(production['cells'][12]['source']),'pinned_kaggle_runtime_cell12.py','exec'),ns)
        selected_snapshot=ns['_gpu_memory_snapshot']()
        if (selected_snapshot.get('gpu_uuid')!=selected_platform['selected_gpu_uuid']
                or selected_snapshot.get('gpu_pci_bus_id')!=selected_platform['selected_gpu_pci_bus_id']
                or selected_snapshot.get('board_total_bytes')!=selected_platform['selected_gpu_total_bytes']):
            raise RuntimeError('Selected physical UUID/PCI board mapping failed before any AOT compile.')
        prep['selected_board_snapshot']=selected_snapshot
        prep['jax_backend_platform_version']=str(getattr(devices[0].client,'platform_version','unavailable'))
        opts=dict(ns['GPU_TUNING'])
        if any(opts.get(k)!=v for k,v in dict(target_gib=14.,reserve_mib=1024,safety_factor=1.45,extra_guard_mib=256).items()):raise RuntimeError('Pinned memory settings changed.')
        def admit(estimate,label):
            guard(label+' before admission');record=ns['_gpu_memory_admission'](estimate,opts,label)
            details=record.get('budget_details',{});snapshot=record.get('snapshot',{})
            if (record.get('status')!='admitted' or record.get('overlap_credit_bytes')!=0 or snapshot.get('platform')!='gpu'
                    or details.get('pool_counter_source')!='pool_bytes' or details.get('pool_counter_status')!='validated_current_total'
                    or 'pool_bytes' not in snapshot.get('jax_memory_stats_raw',{})):
                raise RuntimeError('Current validated pool/board admission required.')
            guard(label+' after admission');return record
        admit(1,'immutable definition constants')
        for index in (6,8,10):
            guard('definition cell'+str(index));exec(compile(''.join(original['cells'][index]['source']),'immutable_cell'+str(index)+'.py','exec'),ns)
        exec(compile(''.join(production['cells'][18]['source']),'pinned_runtime_cell18.py','exec'),ns)
        if ns['TRAINING_REVISION']!='strong-teacher-hand-gae-v3' or len(ns['HEURISTIC_NAMES'])!=13:raise RuntimeError('Rules/training revision mismatch.')
        # make_train_step only defines the immutable JIT functions. No numerical
        # initialize/teacher/finish/wrapper call is made; init is abstract only.
        guard('abstract native template')
        trainer=ns['make_train_step'](cfg)
        abstract=jax.eval_shape(lambda:ns['initialize_training'](cfg))
        after_collect,batch,_=jax.eval_shape(trainer.collect,abstract)
        S,B=ns['_gpu_tree_bytes'](abstract),ns['_gpu_tree_bytes'](batch)
        admit(S,'AOT collect preparation')
        guard('collect lowering');lowered=trainer.collect.lower(abstract)
        guard('collect compile after lowering');collect_handle=lowered.compile();del lowered
        C=ns['_gpu_compiled_bytes'](collect_handle.memory_analysis());prep['explicit_AOT_compiles']=1;write_prep()
        admit(C['total_bytes']+S+65536,'AOT optimize preparation')
        guard('optimizer lowering');lowered=trainer.optimize.lower(after_collect,batch)
        guard('optimizer compile after lowering');optimize_handle=lowered.compile();del lowered
        O=ns['_gpu_compiled_bytes'](optimize_handle.memory_analysis());prep['explicit_AOT_compiles']=2;write_prep()
        estimate=max(C['total_bytes'],O['total_bytes'])+S+65536
        admit(estimate,'sequential AOT selected')
        original_load_admit=ns['gpu_admit_checkpoint_load']
        def load_admit(saved_cfg,size):
            guard('native transfer before admission');outcome=original_load_admit(saved_cfg,size);guard('native transfer after admission');return outcome
        ns['gpu_admit_checkpoint_load']=load_admit
        guard('exact durable674 native candidate load')
        state,saved_cfg,saved_history,metadata=ns['load_checkpoint'](checkpoint_path)
        _aot_sync(dict(ns=ns),state)
        if saved_cfg!=cfg or int(state.updates)!=674 or state.env_steps!=309198848 or not bool(state.bootstrap_complete) or int(state.bootstrap_updates)!=0:
            raise RuntimeError('Loaded native state changed durable update674 config/counters.')
        profile=dict(device=devices[0],optimize_input_specs=_aot_input_specs(jax,np,((after_collect,batch),{})),options=opts,execution_handles=dict(collect=collect_handle,optimize=optimize_handle),
            state_bytes=S,batch_bytes=B,collect=C,optimize=O,estimated_peak_bytes=estimate,
            retained_rollback_states=1,metric_allowance_bytes=65536,JIT_trajectory_equivalence='not established')
        recovery_sink.update(state=state,cfg=copy.deepcopy(cfg),history=saved_history)
        s=_aot_new_session(ns=ns,state=state,cfg=saved_cfg,history=saved_history,profile=profile,
            coordination=coordination,output_dir=root,recovery_sink=recovery_sink,root_precreated=True)
        s['input_native']=derived_native;s['input_package']=_aot_file_identity(candidate_package_path)
        prep.update(status='prepared',state_bytes=S,batch_bytes=B,estimated_peak_bytes=estimate,collect=C,optimize=O,
            observed_update=674,observed_env_steps=309198848,history_rows=len(saved_history),
            journal_scope='Only actual top-level optimizer_work rows; fixed_budget_ppo history remains unmodified, no inferred prior health/lifetime work.')
        write_prep()
        guard('initial branch checkpoint/export barrier')
        _aot_checkpoint_barrier(s,'initial',terminal=False)
        return s
    except BaseException as exc:
        prep.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),retry_attempted=False)
        recovery_sink['preparation_error']=prep
        try:write_prep()
        except BaseException:pass
        raise
