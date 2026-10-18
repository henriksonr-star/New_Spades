"""Host-only Spades V3 deferred-launch operator export content audit. No JAX, network or Library ACK."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
from datetime import datetime,timezone
import numpy as np

BASELINE_SHA='2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13'
PINS={'original_sha256':'56da4fa7bc6e501249cecbd4e41768b27f3930b09fc8938c0fb8f1201a19c28e','production_sha256':'f818c9008adf383835db51d34470f6c933ebec9e3853d0a9c74642b2bcf822b8','candidate_package_sha256':BASELINE_SHA}
RESTORATION=dict(source_update=674,source_env_steps=309198848,previous_last_observed_update=1075,previous_last_observed_env_steps=493158400,lost_unexported_updates=401,lost_unexported_decisions=183959552,new_authorized_session=True,old_segment_clock_recovered=False,playing_strength_evaluated=False)
STACK=dict(jax='0.11.1',flax='0.12.10',optax='0.2.8')
NONTERMINAL={'initial','validation','periodic90','supplemental15'}
TERMINAL={'segment_complete','optimizer_health_stop','authorization_expired','error'}

def require(condition,message):
    if not condition:raise ValueError(message)

def identity(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def finite_json(value):
    if isinstance(value,dict):
        for item in value.values():finite_json(item)
    elif isinstance(value,list):
        for item in value:finite_json(item)
    elif isinstance(value,float):require(math.isfinite(value),'Nonfinite JSON value.')

def read_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key:'+key);result[key]=value
        return result
    def bad(value):raise ValueError('Nonfinite JSON token:'+value)
    result=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad);finite_json(result);return result

def zip_members(raw,names):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(sorted(z.namelist())==sorted(names),'ZIP members differ or contain duplicates.')
        require(all(i.file_size<=128*1024**2 for i in z.infolist()),'Unexpected oversized ZIP member.')
        require(z.testzip() is None,'ZIP CRC failure.')
        return {name:z.read(name) for name in names}

def same_json(left,right):return json.dumps(left,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(right,sort_keys=True,separators=(',',':'),allow_nan=False)

def exact_int(value,minimum=0):return type(value) is int and value>=minimum

def number(value):return type(value) in (int,float) and math.isfinite(value)

def utc(value):
    require(isinstance(value,str),'Explicit UTC string required.')
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None and result.utcoffset().total_seconds()==0,'Explicit UTC timezone required.')
    return result

def preflight_npz(payload,specs):
    names=[f'leaf_{i:05d}.npy' for i in range(214)]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(sorted(archive.namelist())==names,'NPZ214 unique member names.')
        for i,spec in enumerate(specs):
            require(isinstance(spec,dict) and set(spec)=={'shape','dtype'} and isinstance(spec['shape'],list) and all(exact_int(v) for v in spec['shape']),'Leaf spec schema.')
            dtype=np.dtype(spec['dtype']);require(not dtype.hasobject,'Object dtype forbidden.')
            expected=math.prod(spec['shape'])*dtype.itemsize;entry=archive.getinfo(names[i])
            require(entry.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and expected<=entry.file_size<=expected+65536,'NPZ leaf byte bound/compression.')
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                require(version in ((1,0),(2,0)),'Unexpected NPY header version.')
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,actual_dtype=reader(stream)
                require(list(shape)==spec['shape'] and str(actual_dtype)==spec['dtype'] and not actual_dtype.hasobject,'NPY header differs before allocation.')
                require(stream.tell()+expected==entry.file_size,'NPY header/payload size mismatch.')

def native_content(raw,baseline_manifest=None):
    parts=zip_members(raw,('manifest.json','history.json','state.npz'))
    m=read_json(parts['manifest.json']);history=read_json(parts['history.json'])
    require(isinstance(m,dict) and type(m.get('format')) is int and m['format']==1,'Native format.')
    require(m.get('ruleset')=='modified-spades-500-blind-swap-team-failure-v1','Native ruleset.')
    require(m.get('training_revision')=='strong-teacher-hand-gae-v3' and type(m.get('training_state_schema')) is int and m['training_state_schema']==2,'Native revision/schema.')
    require(m.get('versions')==STACK,'Native stack.')
    require(all(exact_int(m.get(k)) for k in ('updates','env_steps')),'Native counter types.')
    require(number(m.get('cumulative_seconds')) and m['cumulative_seconds']>=0,'Native cumulative seconds.')
    utc(m.get('created_utc'))
    require(m.get('sha256')==hashlib.sha256(parts['state.npz']).hexdigest(),'Native payload SHA256.')
    require(isinstance(history,list) and all(isinstance(row,dict) for row in history),'Native history list.')
    cfg=m.get('cfg');require(isinstance(cfg,dict),'Native cfg object.')
    require((cfg.get('seed'),cfg.get('n_env'),cfg.get('rollout_len'),cfg.get('width'),cfg.get('minibatch_size'),cfg.get('ppo_epochs'))==(42,3584,128,256,57344,3),'Native saved configuration dimensions.')
    specs=m.get('leaves');require(isinstance(specs,list) and len(specs)==214,'Native214 specs.')
    if baseline_manifest is not None:
        require(same_json(cfg,baseline_manifest['cfg']),'Full saved cfg differs from trusted674.')
        require(same_json(specs,baseline_manifest['leaves']),'Exact214 leaf specs differ from trusted674.')
    preflight_npz(parts['state.npz'],specs)
    with np.load(io.BytesIO(parts['state.npz']),allow_pickle=False) as arrays:
        require(sorted(arrays.files)==[f'leaf_{i:05d}' for i in range(214)],'Native214 leaf names/duplicates.')
        for i,spec in enumerate(specs):
            require(isinstance(spec,dict) and set(spec)=={'shape','dtype'} and isinstance(spec['shape'],list) and all(exact_int(v) for v in spec['shape']),'Leaf spec schema.')
            a=arrays[f'leaf_{i:05d}']
            require(list(a.shape)==spec['shape'] and str(a.dtype)==spec['dtype'] and not a.dtype.hasobject,'Leaf shape/dtype:'+str(i))
            require(not np.issubdtype(a.dtype,np.inexact) or bool(np.isfinite(a).all()),'Nonfinite native leaf:'+str(i))
        indices=(207,208,209,210,211,212,213);dtypes=('int32','int32','int32','bool','uint32','uint32','int32')
        values=[]
        for i,dtype in zip(indices,dtypes):
            a=arrays[f'leaf_{i:05d}'];require(a.shape==() and str(a.dtype)==dtype,'Counter scalar schema.');values.append(a.item())
        update,decisions,bootstrap,complete,low,high,base=values
        lifetime=((int(high)<<32)|int(low))+(int(update)-int(base))*int(decisions)
        require(update==m['updates'] and decisions==458752 and bootstrap==0 and complete is True and 0<=base<=update and lifetime==m['env_steps'],'Actual native scalar counters/lifetime.')
    return m,history

def baseline_trust_check(info):
    require(info['sha256']==BASELINE_SHA,'Exact trusted674 baseline package SHA256 required.')

def package_content(path,expected_identity=None):
    raw=Path(path).read_bytes()
    if expected_identity is not None:
        require(len(raw)==expected_identity['bytes'] and hashlib.sha256(raw).hexdigest()==expected_identity['sha256'],'Package changed between identity and parse snapshot.')
    parts=zip_members(raw,('native_checkpoint.zip','export_manifest.json'))
    external=read_json(parts['export_manifest.json']);require(isinstance(external,dict),'External manifest object.')
    n=external.get('native');require(isinstance(n,dict) and set(n)=={'path','bytes','sha256'},'External native record schema.')
    require(isinstance(n['path'],str) and n['path'] and exact_int(n['bytes'],1) and n['bytes']==len(parts['native_checkpoint.zip']) and n['sha256']==hashlib.sha256(parts['native_checkpoint.zip']).hexdigest(),'Enclosed native identity.')
    return external,parts['native_checkpoint.zip']

def continuation_arrays(native,baseline,initial):
    current_parts=zip_members(native,('manifest.json','history.json','state.npz'))
    baseline_parts=zip_members(baseline,('manifest.json','history.json','state.npz'))
    indices=range(214) if initial else range(208,214)
    with np.load(io.BytesIO(current_parts['state.npz']),allow_pickle=False) as current, np.load(io.BytesIO(baseline_parts['state.npz']),allow_pickle=False) as old:
        for i in indices:
            key=f'leaf_{i:05d}'
            require(current[key].tobytes(order='C')==old[key].tobytes(order='C'),'Baseline continuation array bytes changed:'+key)

def audit_history(history,base_history,cfg,update,reason,terminal,elapsed,segment_number,max_segments):
    require(same_json(history[:len(base_history)],base_history),'Trusted baseline history prefix changed.')
    new=history[len(base_history):];require(new,'Missing V2 history barrier.')
    arms=[(i,row) for i,row in enumerate(new) if row.get('kind')=='aot_explicit_next_segment']
    require(len(arms)==segment_number-1,'Segment number lacks exact explicit-next-segment history.')
    for expected,(index,arm) in enumerate(arms,2):
        require(type(arm.get('previous_segment')) is int and arm['previous_segment']==expected-1 and type(arm.get('segment_number')) is int and arm['segment_number']==expected and type(arm.get('max_segments')) is int and arm['max_segments']==max_segments and arm.get('automatic') is False and arm.get('no_promotion') is True,'Explicit segment authorization record.')
        barriers=[row for row in new[:index] if row.get('kind')=='aot_export_barrier']
        require(bool(barriers),'Missing previous final barrier.')
        final=barriers[-1]
        require(final.get('reason')=='segment_complete' and final.get('terminal') is True and number(final.get('elapsed_segment_seconds')) and final['elapsed_segment_seconds']>=10800 and type(arm.get('updates')) is int and arm['updates']==final.get('update') and arm.get('env_steps')==309198848+(arm['updates']-674)*458752,'Explicit segment lacks healthy three-hour final.')
        final_index=max(i for i,row in enumerate(new[:index]) if row.get('kind')=='aot_export_barrier')
        acks=[row for row in new[final_index+1:index] if row.get('kind')=='external_export_ack']
        require(bool(acks),'Missing external final acknowledgement.')
        ack=acks[-1]
        require(ack.get('reason')=='segment_complete' and ack.get('nonce')==arm.get('final_receipt_nonce') and isinstance(ack.get('nonce'),str) and bool(ack['nonce']) and ack.get('verified') is True and isinstance(ack.get('library_receipt'),str) and bool(ack['library_receipt'].strip()),'Explicit segment final receipt mismatch.')
    last=new[-1]
    require(last.get('kind')=='aot_export_barrier' and last.get('reason')==reason and type(last.get('update')) is int and last['update']==update and type(last.get('terminal')) is bool and last['terminal']==terminal and last.get('no_promotion') is True,'Last native barrier does not bind current export.')
    require(number(last.get('elapsed_segment_seconds')) and 0<=last['elapsed_segment_seconds']<=elapsed,'Barrier elapsed.')
    if reason=='segment_complete':require(last['elapsed_segment_seconds']>=10800,'Early three-hour final.')
    cadence={k:cfg[k] for k in ('n_env','rollout_len','minibatch_size','ppo_epochs')}
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    planned=cadence['ppo_epochs']*(cadence['n_env']*cadence['rollout_len']//cadence['minibatch_size'])
    work=[];metrics=[]
    for row in new:
        if row.get('kind')=='optimizer_work':
            require(row.get('optimizer_work_version')==1 and type(row.get('optimizer_work_version')) is int and same_json(row.get('cadence'),cadence) and row.get('config_signature')==signature,'New optimizer work version/cfg/cadence.')
            require(all(exact_int(row.get(k)) for k in ('env_steps','update','planned','attempted','accepted','skipped','rejected')),'New work exact counts.')
            require(row['planned']==planned and 1<=row['accepted']==row['attempted']<=planned and row['rejected']==0 and row['skipped']==planned-row['accepted'] and type(row.get('kl_early_stop')) is bool and (row['accepted']==planned or row['kl_early_stop']),'Rejected/inconsistent new optimizer work.')
            require(row['env_steps']==309198848+(row['update']-674)*458752,'New optimizer work lifetime decisions.')
            for name,expected_fraction in (('attempted_fraction',row['attempted']/planned),('accepted_fraction',row['accepted']/planned),('skipped_fraction',row['skipped']/planned)):
                require(number(row.get(name)) and row[name]==expected_fraction,'New work fraction:'+name)
            work.append(row)
        elif row.get('kind')=='aot_raw_metrics':
            require(exact_int(row.get('update'),675),'New metrics update.')
            c=row.get('collection');o=row.get('optimizer');require(isinstance(c,dict) and isinstance(o,dict),'New raw metrics objects.')
            require(all(number(v) for v in c.values()) and all(number(v) for v in o.values()) and c.get('illegal_actions')==0,'New raw metrics illegal/nonfinite.')
            require(all(o.get(k)==1 for k in ('finite','diagnostics_finite','gradients_finite','grad_norm_finite','params_finite','optimizer_finite')),'New raw optimizer finite gates.')
            metrics.append(row)
    expected=list(range(675,update+1))
    require([r['update'] for r in work]==expected and [r['update'] for r in metrics]==expected,'New accepted work/metric coverage missing or out of order.')
    for w,r in zip(work,metrics):
        o=r['optimizer'];require(o.get('attempted_updates')==w['attempted'] and o.get('accepted_updates')==w['accepted'] and o.get('rejected_updates')==0 and o.get('kl_early_stop')==int(w['kl_early_stop']),'Raw metrics disagree with work.')
    expected_metrics=(dict(update=work[-1]['update'],collection=metrics[-1]['collection'],optimizer=metrics[-1]['optimizer'],work=work[-1]) if work else None)
    return dict(_status_last_metrics=expected_metrics,barrier_elapsed=last['elapsed_segment_seconds'],new_accepted_updates=len(work),new_accepted_minibatches=sum(r['accepted'] for r in work),new_raw_metrics=len(metrics))

def audit_export(package_path,status_path,coordination_path,baseline_package_path):
    package_info=identity(package_path);baseline_info=identity(baseline_package_path)
    baseline_trust_check(baseline_info)
    base_external,base_native=package_content(baseline_package_path,baseline_info);bm,bh=native_content(base_native)
    require(base_external.get('format')=='spades-aot-external-export-v1' and base_external.get('updates')==674 and base_external.get('env_steps')==309198848 and base_external.get('reason')=='periodic90' and bm['updates']==674 and bm['env_steps']==309198848,'Trusted baseline674 provenance.')
    status=read_json(Path(status_path).read_bytes());coord=read_json(Path(coordination_path).read_bytes())
    require(isinstance(status,dict) and isinstance(coord,dict),'Status/coordination objects.')
    require(all(coord.get(k) is True for k in ('run_authorized','parent_review_complete','runtime_exclusive','candidate_native_export_verified')),'Fixed coordination approvals.')
    require(coord.get('source_pins')==PINS and type(coord.get('max_segments')) is int and 1<=coord['max_segments']<=8 and type(coord.get('supplemental_export_interval_seconds')) is int and coord['supplemental_export_interval_seconds']==900,'Fixed coordination pins/cadence/cap.')
    require(type(coord.get('export_wait_timeout_seconds')) is int and 1<=coord['export_wait_timeout_seconds']<=3600,'Fixed export wait timeout.')
    prep=utc(coord.get('prepare_valid_until_utc'));auth=utc(coord.get('authorization_valid_until_utc'))
    require(prep<=datetime(2026,10,19,15,49,14,tzinfo=timezone.utc) and auth>=prep,'Fixed preparation/authorization UTC windows.')
    receipt=coord.get('candidate_external_receipt',{})
    require(receipt.get('library_receipt')=='LOCAL_VERIFIED_UPDATE674' and receipt.get('archive_sha256')==BASELINE_SHA,'Fixed initial external baseline receipt.')
    require(status.get('status')=='export_required','Actual flat status must be pending export_required.')
    require(package_info['sha256']==status.get('package_sha256') and type(status.get('package_bytes')) is int and package_info['bytes']==status['package_bytes'],'Outer bytes/SHA differ from status.')
    require(isinstance(status.get('package_path'),str) and status['package_path'],'Remote package path required; local download path is separate.')
    external,native=package_content(package_path,package_info)
    keys={'format','nonce','run_id','reason','native','updates','env_steps','RUN_NAME','source_pins','candidate_label','segment_number','max_segments','restoration','original_JIT_continuity'}
    require(set(external)==keys and external['format']=='spades-aot-external-export-v2','Exact V2 external schema.')
    require(external['source_pins']==PINS and external['source_pins']==coord['source_pins'],'V2 source pins.')
    require(same_json(external['restoration'],RESTORATION) and same_json(status.get('restoration'),RESTORATION),'Exact rollback/restoration provenance.')
    require(type(external['segment_number']) is int and type(external['max_segments']) is int and 1<=external['segment_number']<=external['max_segments']==coord['max_segments'],'Segment cap/number.')
    for k in ('nonce','run_id','reason','updates','env_steps','RUN_NAME','segment_number','max_segments'):
        require(type(status.get(k)) is type(external[k]) and status[k]==external[k],'Status/manifest field:'+k)
    require(all(isinstance(external[k],str) and external[k] for k in ('nonce','run_id')) and external['RUN_NAME']=='modified_spades_run_001','Run identity.')
    require(external['candidate_label']=='durable update674 experimental AOT continuation; unpromoted' and external['original_JIT_continuity']=='failed; unresolved' and status.get('original_combined_JIT_continuity')=='failed; unresolved','Candidate/unresolved-JIT disclosures.')
    require(status.get('route')=='experimental_sequential_AOT' and status.get('promotion') is False and type(status.get('evaluation_calls')) is int and status['evaluation_calls']==0,'Route/no promotion/evaluation.')
    reason=external['reason'];require(reason in NONTERMINAL|TERMINAL and type(status.get('terminal')) is bool,'Barrier reason/runtime terminal.')
    barrier_terminal=reason in TERMINAL
    wait_expired=status.get('stop_reason')=='export_wait_expired'
    if wait_expired:require(status['terminal'] is True,'Export wait expiry must remain terminal.')
    else:
        require(status['terminal']==barrier_terminal,'Barrier/runtime terminal mismatch.')
        require(status.get('stop_reason')==(reason if barrier_terminal else None),'Status stop reason.')
    n=external['native']
    for field,key in (('native_path','path'),('native_sha256','sha256'),('native_bytes','bytes')):
        require(type(status.get(field)) is type(n[key]) and status[field]==n[key],'Remote native record:'+field)
    require(status.get('last_native')==n,'Status last_native record.')
    m,h=native_content(native,bm)
    require(type(external['updates']) is int and external['updates']>=674 and external['updates']==m['updates'] and type(external['env_steps']) is int and external['env_steps']==m['env_steps'],'Manifest/native counters.')
    require(m['env_steps']==309198848+(m['updates']-674)*458752,'New decisions differ from exact baseline continuation.')
    if reason=='initial':require(m['updates']==674 and external['segment_number']==1,'Initial must be unchanged674 in segment1.')
    continuation_arrays(native,base_native,reason=='initial')
    elapsed=status.get('elapsed_segment_seconds');require(number(elapsed) and elapsed>=0,'Status elapsed.')
    history=audit_history(h,bh,m['cfg'],m['updates'],reason,barrier_terminal,elapsed,external['segment_number'],external['max_segments'])
    require(same_json(status.get('last_metrics'),history.pop('_status_last_metrics')),'Status last_metrics differs from actual native history.')
    lower=history['barrier_elapsed'];total_before=(external['segment_number']-1)*10800.
    require(total_before+lower<=m['cumulative_seconds']<=total_before+elapsed,'Native cumulative time outside actual barrier/status interval.')
    if reason=='initial':require(lower==0 and elapsed==0,'Initial elapsed must be zero.')
    if reason=='periodic90':require(5400<=lower<10800,'Periodic90 milestone outside segment.')
    return dict(status='PASS',audit_scope='CONTENT_ONLY',verified_content=True,library_upload_verified=False,ack_authorized=False,scope='Local V2 content verification only; separate external Library upload/readback and exact receipt ACK required.',package=package_info,remote_package_path=status['package_path'],nonce=external['nonce'],run_id=external['run_id'],native=n,updates=m['updates'],env_steps=m['env_steps'],segment_number=external['segment_number'],max_segments=external['max_segments'],reason=reason,terminal=status['terminal'],barrier_terminal=barrier_terminal,resumable=False,export_wait_expired=wait_expired,baseline=baseline_info,full_cfg_matches_baseline=True,leaf_specs_match_baseline=True,native_finite_leaves=214,history=history,original_JIT_continuity='failed; unresolved',playing_strength_evaluated=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','status','coordination','baseline-package','report'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    if Path(args.report).exists() or Path(args.report).is_symlink():p.error('--report must be a new path; existing files are never overwritten.')
    try:report=audit_export(args.package,args.status,args.coordination,args.baseline_package);code=0
    except Exception as e:report=dict(status='FAIL',audit_scope='CONTENT_ONLY',verified_content=False,library_upload_verified=False,ack_authorized=False,error_type=type(e).__name__,error=str(e));code=1
    with Path(args.report).open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
