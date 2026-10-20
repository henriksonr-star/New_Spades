"""Definitions-only, separately frozen diversity evaluation benchmark.

No import-time JAX, kernel, training, network or provider action. The original
full-match evaluation math is retained; only scripted-opponent dispatch changes.
Caller must supply actual operator memory/clock/UUID/authorization callbacks.
"""
import hashlib
import json
import math
from pathlib import Path

OPPONENT_MANIFEST_SHA256='20e4727fdf4d37d035c9ec1bc5d770eb3510f93c9bd20c680c512ceee5f78f1d'
HELDOUT_PROTOCOL_SHA256='0b5970eb566bfcf3db55ffc5559a80e27c072babffe5e1c64f92d3aa5ed2942d'
TRAIN_IDS=tuple(range(13))+tuple(13+64*f+v for f in range(3) for v in range(32))
HELDOUT_IDS=tuple(13+64*f+v for f in range(3,5) for v in range(32,64))
DEVELOPMENT_IDS=(5,6,7,11)+tuple(13+64*f+v for f in range(3) for v in (0,8,16,24))

def _require(condition,message):
    if not condition:raise ValueError(message)

def _identity(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def _write_json(path,value):
    path=Path(path);_require(not path.exists() and not path.is_symlink(),'Evaluation artifact path already exists.')
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return _identity(path)

def _preserve_kernel_outputs(path,outputs):
    """Preserve numeric device results before shape/finite/winner validation."""
    import numpy as np
    import zipfile
    values=outputs if isinstance(outputs,(tuple,list)) else (outputs,)
    arrays={f'output_{i:02d}':np.asarray(value) for i,value in enumerate(values)}
    _require(all(not value.dtype.hasobject for value in arrays.values()),'Object dtype cannot be preserved without unsafe pickle.')
    path=Path(path)
    with path.open('xb') as stream:np.savez(stream,**arrays)
    with zipfile.ZipFile(path) as archive:_require(archive.testzip() is None,'Transport output archive CRC failure.')
    return _identity(path)

def evaluation_plan(mode,seed_name):
    """Fixed benchmark entrypoint; arbitrary or heldout development IDs refused."""
    if mode=='development':
        _require(seed_name=='development','Fixed development seed name required.')
        return dict(mode=mode,seed_name=seed_name,seed=2026101921,bootstrap_seed=2026101922,style_ids=list(DEVELOPMENT_IDS),pairs_per_style=16,games_per_style=32,max_hands=160)
    _require(mode in ('holdout','legacy_regression'),'Unknown evaluation mode.')
    _require(seed_name in ('primary','confirmation'),'Fixed final seed name required.')
    seeds=({'primary':2026101901,'confirmation':2026101907} if mode=='holdout' else {'primary':2026101911,'confirmation':2026101917})
    boot=({'primary':2026101902,'confirmation':2026101908} if mode=='holdout' else {'primary':2026101912,'confirmation':2026101918})
    pairs=16 if mode=='holdout' else 32
    return dict(mode=mode,seed_name=seed_name,seed=seeds[seed_name],bootstrap_seed=boot[seed_name],style_ids=list(HELDOUT_IDS if mode=='holdout' else range(13)),pairs_per_style=pairs,games_per_style=2*pairs,max_hands=160)

def build_diversity_evaluation_kernel(ns,width=256):
    """Define a new kernel in an isolated namespace; do not replace legacy helpers."""
    required=('jax','jnp','np','reset','observe','legal_mask','actor','phase','heuristic_action','policy_logits','batch_step','trick_winner')
    _require(type(width) is int and width==256,'Saved full model width256 required.')
    _require(all(k in ns for k in required),'Complete reviewed evaluation namespace required.')
    namespace=dict(ns)
    _require(hashlib.sha256(FROZEN_OPPONENT_SOURCE.encode()).hexdigest()==FROZEN_OPPONENT_SOURCE_SHA256,'Frozen benchmark opponent source changed.')
    exec(compile(FROZEN_OPPONENT_SOURCE,'frozen_evaluation_opponents.py','exec'),namespace)
    exec(compile(KERNEL_FACTORY_SOURCE,'frozen_diversity_evaluation_kernel.py','exec'),namespace)
    return namespace['_diversity_evaluation_kernel'](width,False,True)

def validate_output(outputs,teams,*,max_hands=160):
    """Pure host validation/scoring. Always retain raw output before reliability failure."""
    import numpy as np
    _require(isinstance(outputs,(tuple,list)) and len(outputs)==7,'Exactly seven evaluation outputs required.')
    completed,winners,hands,scores,made,illegal,score_sum=[np.asarray(v) for v in outputs]
    teams=np.asarray(teams);n=len(teams)
    _require(teams.shape==(n,) and str(teams.dtype)=='int32' and teams.tolist()==[0,1]*(n//2),'Exact duplicate-pair learner team roles required.')
    _require(completed.shape==winners.shape==hands.shape==made.shape==illegal.shape==score_sum.shape==(n,) and scores.shape==(n,2),'Evaluation output shapes differ.')
    _require(str(completed.dtype)=='bool' and all(str(v.dtype)=='int32' for v in (winners,hands,scores,made,illegal)) and str(score_sum.dtype)=='float32','Evaluation output dtypes differ.')
    _require(all(bool(np.isfinite(v).all()) for v in (hands,scores,made,illegal,score_sum)),'Nonfinite evaluation output.')
    _require(bool(((hands>=1)&(hands<=max_hands)).all()) and bool((illegal>=0).all()) and bool(((made>=0)&(made<=hands)).all()),'Evaluation counters invalid.')
    _require(bool(np.isin(winners[completed],[0,1,2]).all()),'Invalid completed engine winner.')
    tied=scores[:,0]==scores[:,1]
    _require(bool(np.array_equal(winners[completed]==2,tied[completed])) and bool(np.array_equal(winners[completed&(~tied)],scores[completed&(~tied)].argmax(axis=1))),'Engine winner/draw disagrees with final scores.')
    _require(bool(np.array_equal(score_sum[completed],scores[np.arange(n),teams][completed])),'Accumulated learner hand scores disagree with final score.')
    outcomes=np.where(completed,(winners==teams).astype(float)+.5*(winners==2),np.nan)
    return dict(completed=completed.tolist(),winners=winners.tolist(),hands=hands.tolist(),scores=scores.tolist(),successful_contracts=made.tolist(),illegal_actions=illegal.tolist(),hand_score_sum=score_sum.tolist(),learner_teams=teams.tolist(),outcomes=[float(x) if np.isfinite(x) else None for x in outcomes],complete=bool(completed.all()),legal=bool((illegal==0).all()),finite=True)

def analyze_paired_raw(raw,plan):
    """Fixed-ID stratified paired-cluster bootstrap; policies share each resample."""
    import numpy as np
    expected=set(plan['style_ids']);_require(set(raw)=={'baseline','candidate'} and all(set(raw[a])==expected for a in raw),'Exact benchmark ID/arm coverage required.')
    rng=np.random.default_rng(plan['bootstrap_seed']);replicates=10000;boots=[];rows=[]
    for style in plan['style_ids']:
        a,b=raw['baseline'][style],raw['candidate'][style]
        _require(a['pair_root_keys']==b['pair_root_keys'] and a['learner_teams']==b['learner_teams'],'Paired evaluation roots/roles differ.')
        _require(all(item.get('complete') is True and item.get('legal') is True and item.get('finite') is True for item in (a,b)),'Incomplete/illegal/nonfinite campaign; no complete-case inference.')
        x=np.asarray(a['outcomes'],dtype=float).reshape(plan['pairs_per_style'],2);y=np.asarray(b['outcomes'],dtype=float).reshape(plan['pairs_per_style'],2)
        _require(bool(np.isfinite(x).all() and np.isfinite(y).all()),'Missing paired outcomes.')
        delta=(y-x).mean(axis=1);indices=rng.integers(0,len(delta),size=(replicates,len(delta)));boot=delta[indices].mean(axis=1);boots.append(boot)
        low,high=np.quantile(boot,[.025,.975]);family=-1 if style<13 else (style-13)//64
        rows.append(dict(style_id=style,family=family,complete_pairs=len(delta),baseline_win_credit=float(x.mean()),candidate_win_credit=float(y.mean()),mean_paired_difference=float(delta.mean()),nominal95_paired_cluster_interval=[float(low),float(high)]))
    matrix=np.stack(boots,axis=1);mean_boot=matrix.mean(axis=1);low,high=np.quantile(mean_boot,[.025,.975]);families=[]
    for family in sorted({r['family'] for r in rows}):
        selected=[i for i,r in enumerate(rows) if r['family']==family];values=matrix[:,selected].mean(axis=1);lo,hi=np.quantile(values,[.025,.975]);families.append(dict(family=family,style_count=len(selected),mean_paired_difference=float(np.mean([rows[i]['mean_paired_difference'] for i in selected])),nominal95_paired_cluster_interval=[float(lo),float(hi)],clear_harm=bool(hi<0)))
    return dict(baseline_win_credit=float(np.mean([r['baseline_win_credit'] for r in rows])),candidate_win_credit=float(np.mean([r['candidate_win_credit'] for r in rows])),aggregate_difference=float(np.mean([r['mean_paired_difference'] for r in rows])),nominal95_stratified_paired_cluster_interval=[float(low),float(high)],root_clusters=sum(r['complete_pairs'] for r in rows),per_style=rows,per_family=families,bootstrap_replicates=replicates,bootstrap_seed=plan['bootstrap_seed'],training_seed_uncertainty_included=False,playing_strength_claim=False)

def _read_json(path):
    def pairs(items):
        result={}
        for k,v in items:
            _require(k not in result,'Duplicate evaluation JSON key.');result[k]=v
        return result
    def invalid(value):raise ValueError('Nonfinite evaluation JSON token.')
    result=json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)
    def finite(v):
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
        elif isinstance(v,float):_require(math.isfinite(v),'Nonfinite evaluation JSON value.')
    finite(result);return result

def audit_evaluation_report(report_path,*,expected_mode='development',expected_cfg=None,expected_cycle=None,expected_checkpoint_records=None):
    """Host-only raw-file/hash/paired-score/statistics audit before cycle review.

    This reads original live raw-file paths. External evidence archives need a
    separate extractor/path map; never replace the recorded identity silently.
    It cannot attest GPU execution or Library upload and never authorizes ACKs.
    """
    report=_read_json(report_path);_require(isinstance(report,dict),'Evaluation report object required.')
    _require(report.get('format')=='spades-diversity-evaluation-report-v1' and report.get('mode')==expected_mode and report.get('status')=='complete' and all(report.get(k) is True for k in ('complete','legal','finite')),'Completed exact-mode evaluation report required.')
    plan=evaluation_plan(expected_mode,report.get('seed_name'));_require(_canonical(report.get('plan'))==_canonical(plan),'Frozen evaluation plan differs.')
    _require(report.get('opponent_manifest_sha256')==OPPONENT_MANIFEST_SHA256 and report.get('heldout_protocol_sha256')==HELDOUT_PROTOCOL_SHA256 and report.get('evaluation_source_sha256')==_identity(__file__)['sha256'] and report.get('frozen_opponent_source_sha256')==FROZEN_OPPONENT_SOURCE_SHA256,'Frozen evaluation source/manifest/protocol mismatch.')
    cfg=report.get('cfg');_require(isinstance(cfg,dict) and report.get('config_signature')==hashlib.sha256(_canonical(cfg).encode()).hexdigest(),'Full evaluation config signature differs.')
    if expected_cfg is not None:_require(_canonical(cfg)==_canonical(expected_cfg),'Evaluation config differs from saved training config.')
    if expected_cycle is not None:_require(type(report.get('cycle_number')) is int and report['cycle_number']==expected_cycle,'Evaluation cycle differs.')
    records=report.get('checkpoint_records');_require(isinstance(records,dict) and set(records)=={'baseline','candidate'},'Checkpoint arm coverage differs.')
    if expected_checkpoint_records is not None:
        for arm in ('baseline','candidate'):
            for key in ('bytes','sha256'):_require(type(records[arm].get(key)) is type(expected_checkpoint_records[arm].get(key)) and records[arm][key]==expected_checkpoint_records[arm][key],'Evaluation checkpoint binding differs:'+arm+':'+key)
    bindings=report.get('parameter_native_bindings');_require(isinstance(bindings,dict) and set(bindings)=={'baseline','candidate'},'Actual parameter/native bindings required.')
    import io,zipfile
    import numpy as np
    for arm in ('baseline','candidate'):
        actual=_identity(records[arm]['path']);_require(actual['bytes']==records[arm]['bytes'] and actual['sha256']==records[arm]['sha256'],'Native file changed after evaluation.')
        with zipfile.ZipFile(records[arm]['path']) as archive:
            m=json.loads(archive.read('manifest.json'));payload=archive.read('state.npz')
        _require(m.get('sha256')==hashlib.sha256(payload).hexdigest() and _canonical(m.get('cfg'))==_canonical(cfg) and len(m.get('leaves',[]))==214,'Actual native payload/fullcfg/schema differs.')
        digest=hashlib.sha256()
        with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
            for i in range(44):
                leaf=arrays[f'leaf_{i:05d}'];spec=m['leaves'][i]
                _require(list(leaf.shape)==spec['shape'] and str(leaf.dtype)==spec['dtype'] and not leaf.dtype.hasobject and bool(np.isfinite(leaf).all()),'Native model parameter spec/finite failure.')
                digest.update(_canonical(spec).encode());digest.update(leaf.tobytes(order='C'))
        binding=bindings[arm];_require(isinstance(binding,dict) and _canonical(binding.get('native'))==_canonical(actual) and type(binding.get('parameter_leaves')) is int and binding['parameter_leaves']==44 and binding.get('parameter_digest_sha256')==digest.hexdigest() and binding.get('parameter_bytes_match_native') is True,'Recomputed native parameter digest/binding differs.')
    files=report.get('raw_files');_require(isinstance(files,list) and len(files)==2*len(plan['style_ids']),'Exact raw evaluation file count required.')
    transports=report.get('transport_files');_require(isinstance(transports,list) and len(transports)==len(files) and len({info['path'] for info in transports})==len(files),'Exact unique transport snapshot coverage required.')
    _require(report.get('transport_manifest_sha256')==hashlib.sha256(_canonical(transports).encode()).hexdigest(),'Transport snapshot manifest hash differs.')
    _require(report.get('raw_manifest_sha256')==hashlib.sha256(_canonical(files).encode()).hexdigest(),'Raw evaluation manifest hash differs.')
    raw={'baseline':{},'candidate':{}};paths=set()
    def integers(v):
        if isinstance(v,list):return all(integers(x) for x in v)
        return type(v) is int
    for info in files:
        _require(isinstance(info,dict) and set(info)=={'path','bytes','sha256'} and isinstance(info['path'],str) and info['path'] not in paths,'Unique exact raw-file identities required.');paths.add(info['path'])
        actual=_identity(info['path']);_require(_canonical(actual)==_canonical(info),'Actual raw file identity changed.')
        row=_read_json(info['path']);arm=row.get('arm');style=row.get('style_id')
        _require(arm in raw and type(style) is int and style in plan['style_ids'] and style not in raw[arm],'Duplicate/missing/unknown raw arm/style.')
        _require(row.get('mode')==expected_mode and row.get('seed_name')==plan['seed_name'] and type(row.get('seed')) is int and row['seed']==plan['seed'] and _canonical(row.get('checkpoint'))==_canonical(records[arm]),'Raw mode/seed/checkpoint binding differs.')
        n=plan['games_per_style'];teams=row.get('learner_teams');_require(teams==[0,1]*plan['pairs_per_style'] and integers(teams),'Raw exact duplicate roles differ.')
        transport=row.get('transport_snapshot');_require(transport in transports and _canonical(_identity(transport['path']))==_canonical(transport),'Actual kernel transport snapshot changed.')
        with np.load(transport['path'],allow_pickle=False) as saved:
            _require(sorted(saved.files)==[f'output_{i:02d}' for i in range(7)],'Transport must retain exactly seven numeric outputs.')
            recomputed=validate_output(tuple(saved[f'output_{i:02d}'] for i in range(7)),np.asarray(teams,np.int32),max_hands=plan['max_hands'])
        _require(all(_canonical(row.get(k))==_canonical(v) for k,v in recomputed.items()),'Raw JSON differs from preserved actual transport arrays.')
        _require(isinstance(row.get('completed'),list) and len(row['completed'])==n and all(type(v) is bool and v for v in row['completed']),'Unresolved/invalid completion flags.')
        for key in ('winners','hands','successful_contracts','illegal_actions'):
            _require(isinstance(row.get(key),list) and len(row[key])==n and integers(row[key]),'Raw exact integer array:'+key)
        _require(all(v in (0,1,2) for v in row['winners']) and all(1<=v<=plan['max_hands'] for v in row['hands']) and all(v==0 for v in row['illegal_actions']) and all(v>=0 for v in row['successful_contracts']),'Raw winner/hands/illegal/counter reliability failure.')
        _require(isinstance(row.get('scores'),list) and len(row['scores'])==n and all(isinstance(x,list) and len(x)==2 and integers(x) for x in row['scores']),'Raw score array differs.')
        _require(isinstance(row.get('hand_score_sum'),list) and len(row['hand_score_sum'])==n and all(type(v) in (int,float) and math.isfinite(v) for v in row['hand_score_sum']),'Raw hand scores differ.')
        for winner,scores,team,score_sum,made,hands in zip(row['winners'],row['scores'],teams,row['hand_score_sum'],row['successful_contracts'],row['hands']):
            _require((winner==2)==(scores[0]==scores[1]) and (winner==2 or winner==(0 if scores[0]>scores[1] else 1)),'Raw engine winner/draw disagrees with final scores.')
            _require(score_sum==scores[team] and made<=hands,'Raw accumulated hand score/contract counter differs.')
        expected=[float(w==t)+.5*(w==2) for w,t in zip(row['winners'],teams)]
        _require(_canonical(row.get('outcomes'))==_canonical(expected) and all(row.get(k) is True for k in ('complete','legal','finite')),'Raw win credit/reliability flags disagree.')
        roots=row.get('pair_root_keys');_require(isinstance(roots,list) and len(roots)==plan['pairs_per_style'] and all(isinstance(x,list) and len(x)==2 and integers(x) and all(0<=v<2**32 for v in x) for x in roots),'Exact uint32 pair-root keys required.')
        raw[arm][style]=row
    roots_seen=set()
    for style in plan['style_ids']:
        keys=[tuple(x) for x in raw['baseline'][style]['pair_root_keys']]
        _require(len(set(keys))==len(keys) and not(set(keys)&roots_seen),'Duplicate pair roots across fixed ID strata.');roots_seen.update(keys)
    analysis=analyze_paired_raw(raw,plan)
    _require(_canonical(report.get('analysis'))==_canonical(analysis) and report.get('aggregate_difference')==analysis['aggregate_difference'],'Recomputed paired statistics differ from report.')
    return dict(status='PASS',audit_scope='CONTENT_ONLY',report=_identity(report_path),mode=expected_mode,cycle_number=report['cycle_number'],raw_file_count=len(files),transport_file_count=len(transports),games=2*len(plan['style_ids'])*plan['games_per_style'],root_clusters=analysis['root_clusters'],root_generation_independently_recomputed=False,root_audit_scope='Root keys are paired and unique across IDs; this host-only audit does not independently reimplement JAX Threefry fold_in/split.',library_upload_verified=False,review_authorized=False,playing_strength_claim=False)

def package_evaluation_evidence(report_path,output_path):
    """Preserve a verified report and every raw row in one new CRC-checked ZIP."""
    import zipfile
    report=_read_json(report_path);audit_evaluation_report(report_path,expected_mode=report['mode'])
    output=Path(output_path);_require(not output.exists() and not output.is_symlink(),'Never overwrite evaluation evidence.')
    members={'evaluation_report.json':Path(report_path).read_bytes()}
    for i,info in enumerate(report['raw_files']):members[f'raw_{i:03d}.json']=Path(info['path']).read_bytes()
    for i,info in enumerate(report['transport_files']):members[f'transport_{i:03d}.npz']=Path(info['path']).read_bytes()
    manifest={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in members.items()}
    members['evaluation_archive_manifest.json']=(json.dumps(dict(format='spades-diversity-evaluation-evidence-v1',members=manifest),indent=2)+'\n').encode()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in members.items():archive.writestr(name,data)
    with zipfile.ZipFile(output) as archive:
        _require(archive.testzip() is None and set(archive.namelist())==set(members),'Evaluation evidence archive CRC/member mismatch.')
        for name,data in members.items():_require(archive.read(name)==data,'Evaluation evidence member bytes differ.')
    return _identity(output)

def _bind_params_to_native(ns,params,record,cfg,*,baseline):
    """Bind actual policy leaves to exact native bytes, before evaluation kernels."""
    import io,zipfile
    import numpy as np
    actual=_identity(record['path'])
    _require(actual['bytes']==record['bytes'] and actual['sha256']==record['sha256'],'Actual native checkpoint file identity differs.')
    _require(tuple(ns['TrainingState']._fields[:1])==('params',),'Reviewed TrainingState.params must be first native subtree.')
    with zipfile.ZipFile(record['path']) as archive:
        _require(set(archive.namelist())=={'manifest.json','history.json','state.npz'} and len(archive.namelist())==3 and archive.testzip() is None,'Exact native checkpoint container/CRC required.')
        manifest=json.loads(archive.read('manifest.json'));payload=archive.read('state.npz')
    revision='strong-teacher-hand-gae-v3' if baseline else 'strong-teacher-hand-gae-v3+opponent-diversity-v1'
    _require(manifest.get('training_revision')==revision and manifest.get('training_state_schema')==2 and len(manifest.get('leaves',[]))==214 and _canonical(manifest.get('cfg'))==_canonical(cfg),'Native checkpoint source/schema/fullcfg binding differs.')
    _require(manifest.get('sha256')==hashlib.sha256(payload).hexdigest(),'Native array payload SHA differs.')
    if baseline:_require(manifest.get('updates')==674 and manifest.get('env_steps')==309198848,'Frozen baseline must be exact674.')
    leaves=ns['jax'].tree_util.tree_leaves(params);_require(len(leaves)==44,'Exact44 model parameter leaves required.')
    digest=hashlib.sha256()
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        for index,leaf in enumerate(leaves):
            actual_leaf=np.asarray(ns['jax'].device_get(leaf));native_leaf=arrays[f'leaf_{index:05d}'];spec=manifest['leaves'][index]
            _require(actual_leaf.shape==native_leaf.shape and actual_leaf.dtype==native_leaf.dtype and list(actual_leaf.shape)==spec['shape'] and str(actual_leaf.dtype)==spec['dtype'] and not actual_leaf.dtype.hasobject and bool(np.isfinite(actual_leaf).all()) and actual_leaf.tobytes(order='C')==native_leaf.tobytes(order='C'),'Policy parameters do not match exact native leaf:'+str(index))
            digest.update(_canonical(spec).encode());digest.update(actual_leaf.tobytes(order='C'))
    return dict(native=actual,parameter_leaves=44,parameter_digest_sha256=digest.hexdigest(),parameter_bytes_match_native=True)

def evaluate_diversity_pair(*,ns,baseline_params,candidate_params,cfg,mode,seed_name,output_dir,checkpoint_records,precompile_estimate_bytes,admit,launch_guard,cycle_number=1,final_authorization=None):
    """Explicit bounded launch API; callbacks must enforce real operator gates.

    `launch_guard(label)` enforces selected UUID, unchanged source/cfg, current
    authorization and global clock. `admit(bytes,label)` must return actual
    validated pool/board admission. Neither callback may be a self-attestation.
    This function never trains, ACKs a checkpoint, changes configuration, or
    automatically runs another evaluation campaign.
    """
    import numpy as np
    plan=evaluation_plan(mode,seed_name)
    _require(callable(admit) and callable(launch_guard),'Actual operator admission/launch callbacks required.')
    _require(type(precompile_estimate_bytes) is int and precompile_estimate_bytes>0,'Explicit reviewed precompile memory estimate required.')
    _require(type(cycle_number) is int and 1<=cycle_number<=12,'Cycle must be1..12.')
    _require(cfg.get('width')==256 and cfg.get('n_env')==3584 and cfg.get('rollout_len')==128,'Exact saved full configuration required.')
    _require(isinstance(checkpoint_records,dict) and set(checkpoint_records)=={'baseline','candidate'},'Exact baseline/candidate checkpoint bindings required.')
    for record in checkpoint_records.values():
        _require(isinstance(record,dict) and isinstance(record.get('sha256'),str) and len(record['sha256'])==64 and all(c in '0123456789abcdef' for c in record['sha256']) and type(record.get('bytes')) is int and record['bytes']>0 and isinstance(record.get('path'),str) and record['path'],'Actual native checkpoint path/bytes/SHA records required.')
    if mode!='development':
        a=final_authorization or {}
        _require(all(a.get(k) is True for k in ('global_training_stopped','selection_frozen','no_previous_holdout')) and a.get('candidate_sha256')==checkpoint_records['candidate']['sha256'] and isinstance(a.get('library_receipt'),str) and a['library_receipt'].strip(),'Separate final-selection/no-previous-holdout authorization required.')
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    report=dict(format='spades-diversity-evaluation-report-v1',mode=mode,seed_name=seed_name,cycle_number=cycle_number,complete=False,legal=False,finite=False,status='preparing',checkpoint_records=checkpoint_records,config_signature=hashlib.sha256(_canonical(cfg).encode()).hexdigest(),cfg=cfg,evaluation_source=_identity(__file__),evaluation_source_sha256=_identity(__file__)['sha256'],frozen_opponent_source_sha256=FROZEN_OPPONENT_SOURCE_SHA256,opponent_manifest_sha256=OPPONENT_MANIFEST_SHA256,heldout_protocol_sha256=HELDOUT_PROTOCOL_SHA256,plan=plan,raw_files=[],transport_files=[],scope='Explicit evaluation output; separate external Library upload/readback and review receipt required. No training, promotion or automatic continuation.',library_upload_verified=False,review_authorized=False)
    def gate(estimate,label):
        launch_guard(label+' before admission');record=admit(estimate,label)
        _require(isinstance(record,dict) and record.get('status')=='admitted' and record.get('overlap_credit_bytes')==0 and record.get('snapshot',{}).get('platform')=='gpu' and record.get('budget_details',{}).get('pool_counter_source')=='pool_bytes' and record.get('budget_details',{}).get('pool_counter_status')=='validated_current_total','Actual validated GPU pool/board admission required.')
        launch_guard(label+' after admission');return record
    try:
        launch_guard('evaluation parameter/native binding')
        report['parameter_native_bindings']={arm:_bind_params_to_native(ns,params,checkpoint_records[arm],cfg,baseline=arm=='baseline') for arm,params in (('baseline',baseline_params),('candidate',candidate_params))}
        jax,jnp=ns['jax'],ns['jnp'];kernel=build_diversity_evaluation_kernel(ns,int(cfg['width']))
        style=plan['style_ids'][0];roots=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(plan['seed']),style),plan['pairs_per_style']);keys=jnp.repeat(roots,2,axis=0);teams=jnp.tile(jnp.asarray([0,1],jnp.int32),plan['pairs_per_style'])
        gate(precompile_estimate_bytes,'evaluation before lowering');launch_guard('evaluation lowering');lowered=kernel.lower(baseline_params,baseline_params,keys,teams,jnp.int32(style),jnp.int32(plan['max_hands']))
        gate(precompile_estimate_bytes,'evaluation before compile');launch_guard('evaluation compile');handle=lowered.compile();del lowered
        profile=ns['_gpu_compiled_bytes'](handle.memory_analysis());estimate=profile.get('total_bytes');_require(type(estimate) is int and estimate>0,'Actual evaluation compiler memory profile required.');report['compiler_profile']=profile
        raw={'baseline':{},'candidate':{}};all_roots=set()
        for style in plan['style_ids']:
            roots=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(plan['seed']),style),plan['pairs_per_style']);keys=jnp.repeat(roots,2,axis=0);teams=jnp.tile(jnp.asarray([0,1],jnp.int32),plan['pairs_per_style'])
            root_values=np.asarray(jax.device_get(roots));root_tuples=[tuple(int(v) for v in row) for row in root_values]
            _require(len(set(root_tuples))==len(root_tuples) and not(set(root_tuples)&all_roots),'Duplicate root keys across style strata.');all_roots.update(root_tuples)
            for arm,params in (('baseline',baseline_params),('candidate',candidate_params)):
                label=f'evaluation {mode} {seed_name} {style} {arm}';gate(estimate,label);launch_guard(label+' kernel');outputs=handle(params,params,keys,teams,jnp.int32(style),jnp.int32(plan['max_hands']));outputs=jax.device_get(outputs)
                transport=_preserve_kernel_outputs(root/f'transport_{mode}_{seed_name}_{style}_{arm}.npz',outputs);report['transport_files'].append(transport)
                row=validate_output(outputs,np.asarray(jax.device_get(teams)),max_hands=plan['max_hands']);row.update(mode=mode,seed_name=seed_name,seed=plan['seed'],style_id=style,arm=arm,pair_root_keys=root_values.tolist(),checkpoint=checkpoint_records[arm],transport_snapshot=transport)
                info=_write_json(root/f'raw_{mode}_{seed_name}_{style}_{arm}.json',row);report['raw_files'].append(info);raw[arm][style]=row
                launch_guard(label+' after fence and raw preservation')
                _require(row['complete'] and row['legal'] and row['finite'],'Evaluation invalid/incomplete; raw evidence retained, no inference/retry.')
        analysis=analyze_paired_raw(raw,plan);report.update(status='complete',complete=True,legal=True,finite=True,analysis=analysis,aggregate_difference=analysis['aggregate_difference'],raw_manifest_sha256=hashlib.sha256(_canonical(report['raw_files']).encode()).hexdigest(),transport_manifest_sha256=hashlib.sha256(_canonical(report['transport_files']).encode()).hexdigest())
    except BaseException as exc:
        report.update(status='failed',complete=False,legal=False,finite=False,error=dict(type=type(exc).__name__,message=str(exc)))
        _write_json(root/'evaluation_report.json',report);raise
    _write_json(root/'evaluation_report.json',report)
    return report

KERNEL_FACTORY_SOURCE='def _diversity_evaluation_kernel(width, snapshot_opponent, deterministic):\n    """One cached executable per architecture/opponent kind/action convention."""\n    inference_cfg = {\'width\': width}\n\n    def choose(logits, masks, keys):\n        logits = jnp.where(masks, logits, -jnp.inf)\n        if deterministic:\n            return jnp.argmax(logits, axis=-1).astype(jnp.int32)\n        return jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)\n\n    @jax.jit\n    def run(params, other_params, pair_keys, learner_teams, style, max_hands):\n        count = learner_teams.shape[0]\n        reset_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(pair_keys)\n        states = jax.vmap(reset)(reset_keys)\n        # The hand clock is independent of bidding, trick winners, and policy randomness.\n        # Both sides of a duplicate pair get the same reset and every subsequent deal key.\n        carry = (\n            states,\n            jnp.zeros(count, bool),                    # completed matches\n            jnp.full(count, -1, jnp.int32),             # true engine winners only\n            jnp.zeros(count, jnp.int32),               # played hands\n            jnp.zeros((count, 2), jnp.int32),           # last scores before auto-reset\n            jnp.zeros(count, jnp.int32),               # successful learner contracts\n            jnp.zeros(count, jnp.int32),               # all illegal actions (must be 0)\n            jnp.zeros(count, jnp.float32),             # learner hand score sum\n            jnp.array(0, jnp.int32),                   # synchronized hand index\n        )\n\n        def condition(c):\n            return (c[-1] < max_hands) & (~jnp.all(c[1]))\n\n        def hand(c):\n            state, completed, winners, hands, scores, made, illegal, score_sum, h = c\n            active = ~completed\n            deal_keys = jax.vmap(lambda k: jax.random.fold_in(k, h + 1))(pair_keys)\n\n            def turn(s, t):\n                action_keys = jax.vmap(\n                    lambda k: jax.random.fold_in(k, 1_000_000 + h * 64 + t)\n                )(pair_keys)\n                observations = jax.vmap(observe)(s)\n                masks = jax.vmap(legal_mask)(s)\n                players = jax.vmap(actor)(s)\n                learner_actions = choose(policy_logits(params, observations, inference_cfg), masks, action_keys)\n                opponent_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(action_keys)\n                if snapshot_opponent:\n                    opponent_actions = choose(\n                        policy_logits(other_params, observations, inference_cfg), masks, opponent_keys\n                    )\n                else:\n                    opponent_actions = jax.vmap(diverse_heuristic_action, in_axes=(0, 0, None))(\n                        s, opponent_keys, style\n                    )\n                actions = jnp.where(players % 2 == learner_teams, learner_actions, opponent_actions)\n                # Keep every member on the same scheduled turn so the engine can\n                # branch once per batch. Completed matches may play ignored extra\n                # hands; active masks below freeze all recorded results and metrics.\n                nxt, _, _, info = batch_step(s, actions, deal_keys)\n                compact = (info[\'match_done\'], info[\'winner\'], info[\'final_scores\'],\n                           info[\'contract_success\'], info[\'illegal\'], info[\'score_delta\'])\n                return nxt, compact\n\n            state, info = jax.lax.scan(turn, state, jnp.arange(64, dtype=jnp.int32))\n            ended = info[0][-1] & active\n            winners = jnp.where(ended, info[1][-1], winners)\n            scores = jnp.where(active[:, None], info[2][-1], scores)\n            row = jnp.arange(count)\n            made += (info[3][-1][row, learner_teams] & active).astype(jnp.int32)\n            illegal += jnp.sum(info[4].astype(jnp.int32), axis=0) * active\n            score_sum += jnp.where(active, info[5][-1][row, learner_teams], 0)\n            hands += active.astype(jnp.int32)\n            return state, completed | ended, winners, hands, scores, made, illegal, score_sum, h + 1\n\n        result = jax.lax.while_loop(condition, hand, carry)\n        return result[1:8]\n    return run'

FROZEN_OPPONENT_SOURCE_SHA256='75649c722d2843641322084e157a6c6816bd8a28a32ced416522b9a732d9da55'
FROZEN_OPPONENT_SOURCE='"""Additional observation-limited opponents; append after original cell8.\n\nUses the supplied JAX/rules/original heuristic namespace. No game-rule, learning,\nstate-schema, original-name or original-function replacement occurs here.\nEncoded IDs persist in the existing scalar league_style for an entire match.\n"""\n\nDIVERSITY_OPPONENT_REVISION = \'spades-opponent-diversity-v1\'\nDIVERSITY_STYLE_BASE = 13\nDIVERSITY_FAMILY_STRIDE = 64\nDIVERSITY_TRAIN_FAMILIES = (\'trump_management\', \'void_building\', \'tempo_ducking\')\nDIVERSITY_HOLDOUT_FAMILIES = (\'crossruff_pressure\', \'score_threshold_defense\')\nDIVERSITY_TRAIN_VARIANTS = tuple(range(32))\nDIVERSITY_HOLDOUT_VARIANTS = tuple(range(32, 64))\nDIVERSITY_TRAIN_STYLE_IDS = tuple(range(13)) + tuple(\n    13 + family * 64 + variant for family in range(3) for variant in range(32))\nDIVERSITY_HOLDOUT_STYLE_IDS = tuple(\n    13 + family * 64 + variant for family in range(3, 5) for variant in range(32, 64))\nDIVERSITY_MAX_STYLE_ID = 332\nDIVERSITY_OPPONENT_MANIFEST = {\n    \'revision\': DIVERSITY_OPPONENT_REVISION,\n    \'legacy_ids\': list(range(13)),\n    \'encoding\': {\'base\': 13, \'family_stride\': 64, \'maximum_id\': 332},\n    \'train_families\': list(DIVERSITY_TRAIN_FAMILIES),\n    \'holdout_families\': list(DIVERSITY_HOLDOUT_FAMILIES),\n    \'train_family_indices\': [0, 1, 2], \'holdout_family_indices\': [3, 4],\n    \'train_variants\': list(range(32)), \'holdout_variants\': list(range(32, 64)),\n    \'train_style_ids\': list(DIVERSITY_TRAIN_STYLE_IDS),\n    \'holdout_style_ids\': list(DIVERSITY_HOLDOUT_STYLE_IDS),\n    \'parameter\': \'p=(variant+0.5)/64\',\n    \'parameter_quantization\': \'32 midpoint bins per split; q=(variant%32+0.5)/32\',\n    \'used_parameter_bands\': {\n        \'train\': {\'bid_bias_interval\': [-0.4, 0.4], \'bid_bias_attainable\': [-0.3875, 0.3875],\n                  \'risk_scale_interval\': [0.8, 1.2], \'risk_scale_attainable\': [0.80625, 1.19375],\n                  \'urgency_scale_interval\': [0.8, 1.2], \'urgency_scale_attainable\': [0.80625, 1.19375]},\n        \'holdout_crossruff_pressure\': {\'bid_bias_interval\': [-1.0, -0.7], \'bid_bias_attainable\': [-0.9953125, -0.7046875],\n                  \'risk_scale_interval\': [1.35, 1.7], \'risk_scale_attainable\': [1.35546875, 1.69453125],\n                  \'urgency_scale_interval\': [1.35, 1.7], \'urgency_scale_attainable\': [1.35546875, 1.69453125]},\n        \'holdout_score_threshold_defense\': {\'bid_bias_interval\': [0.7, 1.0], \'bid_bias_attainable\': [0.7046875, 0.9953125],\n                  \'risk_scale_interval\': [1.35, 1.7], \'risk_scale_attainable\': [1.35546875, 1.69453125],\n                  \'urgency_scale_interval\': [1.35, 1.7], \'urgency_scale_attainable\': [1.35546875, 1.69453125]},\n    },\n    \'family_rules\': {\n        \'trump_management\': \'lead trump from long holdings when contract needs tricks; conserve high trump; cheap ruffs\',\n        \'void_building\': \'clear short side suits while holding trump; use low long-suit entries\',\n        \'tempo_ducking\': \'duck early seats; cheapest last-seat win when needed; urgency threshold switches intent\',\n        \'crossruff_pressure\': \'HELDOUT partner-known-void leads interact with opponent-void avoidance and master cashing\',\n        \'score_threshold_defense\': \'HELDOUT public score and bag thresholds switch denial versus avoiding bags\',\n    },\n    \'train_parameter_interval\': [0.0078125, 0.4921875],\n    \'holdout_parameter_interval\': [0.5078125, 0.9921875],\n    \'scripted_sampling\': {\'legacy_probability\': 0.5, \'new_probability\': 0.5,\n                          \'legacy_uniform_count\': 13, \'new_uniform_family_count\': 3,\n                          \'new_uniform_variants_per_family\': 32, \'assignment_scope\': \'match\'},\n    \'invalid_action\': -1,\n    \'information\': \'acting hand and public bids/tricks/scores/bags/table/played ownership/voids only; blind uses no hands\',\n    \'search\': \'none; fixed52-card feature scores; no lookahead/neural inference\',\n    \'strength\': \'unmeasured; diversity does not imply stronger opponents\',\n}\n\n\ndef decode_opponent_style_id(style):\n    """JAX scalar decode: legacy family=-1, variant=legacy ID; invalid=False.\n\n    Caller state stores int32 IDs. No clipping converts invalid/reserved IDs\n    into an admitted training style. Host/native boundaries validate the dtype.\n    """\n    raw = jnp.asarray(style)\n    integer_dtype = jnp.issubdtype(raw.dtype, jnp.integer)\n    value = raw.astype(jnp.int32)\n    offset = value - DIVERSITY_STYLE_BASE\n    family = offset // DIVERSITY_FAMILY_STRIDE\n    variant = offset % DIVERSITY_FAMILY_STRIDE\n    legacy = (value >= 0) & (value < DIVERSITY_STYLE_BASE)\n    train = (family >= 0) & (family < 3) & (variant < 32)\n    holdout = (family >= 3) & (family < 5) & (variant >= 32)\n    return (jnp.where(legacy, -1, family).astype(jnp.int32),\n            jnp.where(legacy, value, variant).astype(jnp.int32),\n            (legacy | train | holdout) & integer_dtype)\n\n\ndef is_train_opponent_style_id(style):\n    family, variant, valid = decode_opponent_style_id(style)\n    return valid & ((family == -1) | ((family >= 0) & (family < 3) & (variant < 32)))\n\n\ndef is_holdout_opponent_style_id(style):\n    family, variant, valid = decode_opponent_style_id(style)\n    return valid & (family >= 3) & (family < 5) & (variant >= 32)\n\n\ndef sample_train_opponent_style_ids(key, n):\n    """Half original13, half balanced3x32 train variants; never holdout IDs.\n\n    The caller supplies its existing per-match style key and keeps the encoded\n    scalar until that match finishes. This does not add a persistent RNG stream.\n    """\n    if not isinstance(n, int) or isinstance(n, bool) or n < 0:\n        raise ValueError(\'Static nonnegative Python integer population size required.\')\n    mix_key, legacy_key, family_key, variant_key = jax.random.split(key, 4)\n    legacy = jax.random.randint(legacy_key, (n,), 0, 13, dtype=jnp.int32)\n    family = jax.random.randint(family_key, (n,), 0, 3, dtype=jnp.int32)\n    variant = jax.random.randint(variant_key, (n,), 0, 32, dtype=jnp.int32)\n    new_style = 13 + family * 64 + variant\n    return jnp.where(jax.random.bernoulli(mix_key, 0.5, (n,)), legacy,\n                     new_style).astype(jnp.int32)\n\n\ndef diversity_style_parameters(family, variant):\n    """Actual finite profile parameters; train and heldout bands are disjoint."""\n    q = ((variant % 32).astype(jnp.float32) + 0.5) / 32.\n    train = family < 3\n    bid_bias = jnp.where(train, -0.4 + 0.8 * q,\n                        jnp.where(family == 3, -1.0 + 0.3 * q, 0.7 + 0.3 * q))\n    scale = jnp.where(train, 0.8 + 0.4 * q, 1.35 + 0.35 * q)\n    return dict(p=(variant.astype(jnp.float32) + 0.5) / 64.,\n                bid_bias=bid_bias, risk_scale=scale, urgency_scale=scale)\n\n\ndef _diversity_acting_hand(state):\n    """The only private-card read in the new play/pass/bid implementation."""\n    return state.hands[actor(state)]\n\n\ndef _diversity_public_play_features(state):\n    """Shared cheap52-card features from own hand and public game state."""\n    me = actor(state)\n    partner = (me + 2) % 4\n    team = me % 2\n    ids = jnp.arange(52, dtype=jnp.int32)\n    suits, ranks = ids // 13, ids % 13\n    rank = ranks.astype(jnp.float32) / 12.\n    hand = _diversity_acting_hand(state)\n    lengths = hand.reshape(4, 13).sum(axis=1)\n    leading = state.trick_pos == 0\n    led = jnp.maximum(state.table[0], 0) // 13\n    table = jnp.maximum(state.table, 0)\n    table_strength = jnp.where(table // 13 == 3, 26 + table % 13,\n                               jnp.where(table // 13 == led, 13 + table % 13, -1))\n    table_strength = jnp.where(state.table >= 0, table_strength, -100)\n    current_winner = jnp.where(leading, -1, state.table_players[jnp.argmax(table_strength)])\n    candidate_strength = jnp.where(suits == 3, 26 + ranks,\n                                   jnp.where(suits == led, 13 + ranks, -1))\n    wins = leading | (candidate_strength > jnp.max(table_strength))\n    partner_winning = current_winner == partner\n    unseen = ((state.played_by < 0) & ~hand).reshape(4, 13)\n    higher = jnp.flip(jnp.cumsum(jnp.flip(unseen, axis=1), axis=1), axis=1) - unseen\n    master = higher.reshape(52) == 0\n    seats = jnp.arange(4)\n    own_team = seats % 2 == team\n    normal_bid = jnp.maximum(state.bids, 0)\n    normal_tricks = jnp.where(state.bids > 0, state.tricks, 0)\n    our_need = jnp.maximum(jnp.sum(jnp.where(own_team, normal_bid - normal_tricks, 0)), 0)\n    their_need = jnp.maximum(jnp.sum(jnp.where(~own_team, normal_bid - normal_tricks, 0)), 0)\n    remaining = jnp.maximum(13 - jnp.maximum(state.turn - 12, 0) // 4, 1)\n    live_nil = (state.bids <= 0) & (state.tricks == 0)\n    own_failed = jnp.any(own_team & (state.bids <= 0) & (state.tricks > 0))\n    their_failed = jnp.any(~own_team & (state.bids <= 0) & (state.tricks > 0))\n    our_need = jnp.where(own_failed, 0, our_need)\n    their_need = jnp.where(their_failed, 0, their_need)\n    partner_nil = live_nil[partner] & ~own_failed\n    own_nil = live_nil[me] & ~own_failed\n    opposing_nil_winning = jnp.any((~own_team) & live_nil & (seats == current_winner)) & ~their_failed\n    opponent_void_count = jnp.sum(state.voids & (~own_team)[:, None], axis=0)\n    partner_void = state.voids[partner]\n    return dict(rank=rank, suits=suits, lengths=lengths, leading=leading,\n        wins=wins, partner_winning=partner_winning, master=master,\n        our_need=our_need, their_need=their_need, remaining=remaining,\n        own_nil=own_nil, partner_nil=partner_nil, opposing_nil_winning=opposing_nil_winning,\n        opponent_void_count=opponent_void_count, partner_void=partner_void,\n        score_gap=state.scores[team] - state.scores[1 - team],\n        own_score=state.scores[team], opponent_score=state.scores[1 - team],\n        own_bags=state.bags[team],\n        me=me, partner=partner, team=team, trick_pos=state.trick_pos)\n\n\ndef _diversity_trump_management(f, p):\n    """Draw trump from long holdings; conserve high trump; cheap ruffing."""\n    trump = f[\'suits\'] == 3\n    need = f[\'our_need\'] > 0\n    draw = f[\'leading\'] & (f[\'lengths\'][3] >= 4) & need\n    # Targeted trump leads differ from a universal high/low-card endpoint.\n    score = (2.2 + 1.4 * p) * f[\'wins\'] * need\n    score -= (0.35 + 0.9 * p) * f[\'rank\'] + (1.1 + p) * trump\n    score += draw * trump * (1.4 + 1.8 * f[\'master\'])\n    score -= (~f[\'leading\']) * f[\'partner_winning\'] * f[\'wins\'] * (2.5 + p)\n    score += (~f[\'leading\']) * trump * f[\'wins\'] * need * (1.0 - f[\'rank\'])\n    return score\n\n\ndef _diversity_void_building(f, p):\n    """Clear short side suits to create ruffs; prefer low entries to long suits."""\n    side = f[\'suits\'] != 3\n    singleton = (f[\'lengths\'][f[\'suits\']] == 1) & side\n    short = jnp.maximum(4 - f[\'lengths\'][f[\'suits\']], 0).astype(jnp.float32)\n    can_ruff = f[\'lengths\'][3] >= 2\n    score = (1.4 + p) * f[\'wins\'] * (f[\'our_need\'] > 0) - 0.3 * f[\'rank\']\n    score += can_ruff * side * (0.35 + 0.7 * p) * short\n    score += singleton * can_ruff * (1.2 + p)\n    score += f[\'leading\'] * side * (~singleton) * (0.12 * f[\'lengths\'][f[\'suits\']] - 0.8 * f[\'rank\'])\n    score -= (f[\'suits\'] == 3) * (0.8 + 0.6 * p)\n    score -= (~f[\'leading\']) * f[\'partner_winning\'] * f[\'wins\'] * 2.4\n    return score\n\n\ndef _diversity_tempo_ducking(f, p):\n    """Duck early seats; take a cheap last-seat trick when the contract needs it."""\n    last = f[\'trick_pos\'] == 3\n    pressure = f[\'our_need\'] / f[\'remaining\']\n    take = (f[\'our_need\'] > 0) & (last | (pressure > 0.55 + 0.35 * p))\n    score = f[\'wins\'] * jnp.where(take, 3.0 + p, -1.5 - 2.0 * p)\n    score -= (0.4 + 0.5 * p) * f[\'rank\'] + 0.5 * (f[\'suits\'] == 3)\n    score += f[\'leading\'] * (0.24 * f[\'lengths\'][f[\'suits\']] - (0.9 + p) * f[\'rank\'])\n    score -= f[\'partner_winning\'] * f[\'wins\'] * (2.0 + p)\n    return score\n\n\ndef _diversity_crossruff_pressure(f, p):\n    """HELDOUT: public partner-void leads and opponent-void avoidance interact."""\n    side = f[\'suits\'] != 3\n    partner_ruff_entry = side & f[\'partner_void\'][f[\'suits\']]\n    safe_side = side & (f[\'opponent_void_count\'][f[\'suits\']] == 0)\n    score = (2.0 + p) * f[\'wins\'] * (f[\'our_need\'] > 0) - 0.45 * f[\'rank\']\n    score += f[\'leading\'] * partner_ruff_entry * (2.8 + 2.2 * p) * (1.0 - 0.5 * f[\'rank\'])\n    score += f[\'leading\'] * safe_side * f[\'master\'] * (1.4 + p)\n    score -= f[\'leading\'] * side * f[\'opponent_void_count\'][f[\'suits\']] * (0.6 + p)\n    score += (~f[\'leading\']) * (f[\'suits\'] == 3) * f[\'wins\'] * (1.0 - f[\'rank\']) * (1.8 + p)\n    score -= f[\'partner_winning\'] * f[\'wins\'] * 3.0\n    return score\n\n\ndef _diversity_score_threshold_defense(f, p):\n    """HELDOUT: score/bag threshold switches between denial and bag avoidance."""\n    opponent_close = f[\'opponent_score\'] >= 300.0 + 100.0 * p\n    own_close = f[\'own_score\'] >= 330.0 + 80.0 * p\n    losing = f[\'score_gap\'] < -(20.0 + 80.0 * p)\n    bag_danger = f[\'own_bags\'] >= 5.0 + 3.0 * p\n    deny = (f[\'their_need\'] > 0) & (opponent_close | losing)\n    take = (f[\'our_need\'] > 0) | deny\n    avoid = (~take) & (bag_danger | own_close)\n    score = f[\'wins\'] * jnp.where(take, 2.5 + 2.0 * p, jnp.where(avoid, -4.0, -0.5))\n    # Cash public masters only under score pressure; otherwise save high cards.\n    score += f[\'leading\'] * f[\'master\'] * deny * (2.0 + p)\n    score -= (0.3 + p) * f[\'rank\'] + 0.45 * (f[\'suits\'] == 3)\n    score -= f[\'partner_winning\'] * f[\'wins\'] * (1.5 + 2.0 * p)\n    return score\n\n\ndef _diversity_blind_action(state, key, family, variant):\n    """No legal_mask/hand/private-card read in blind polling."""\n    me = actor(state)\n    team = me % 2\n    p = (variant.astype(jnp.float32) + 0.5) / 64.\n    deficit = state.scores[1 - team] - state.scores[team]\n    base = jnp.array([0.015, 0.025, 0.008, 0.04, 0.01], jnp.float32)[family]\n    comeback = (deficit >= 180) & (state.scores[1 - team] >= 350)\n    chance = base + (0.04 + 0.08 * p) * comeback\n    chance *= jnp.where(state.bids[(me + 2) % 4] == -1, 0.15, 1.)\n    declare = state.eligible[team] & (jax.random.uniform(key) < chance)\n    return jnp.where(declare, 67, 66).astype(jnp.int32)\n\n\ndef _diversity_pass_action(state, key, family, variant, legal):\n    hand = _diversity_acting_hand(state)\n    me = actor(state)\n    ids = jnp.arange(52)\n    ranks, suits = ids % 13, ids // 13\n    lengths = hand.reshape(4, 13).sum(axis=1)\n    p = (variant.astype(jnp.float32) + 0.5) / 64.\n    export_danger = ranks.astype(jnp.float32) + (5.0 + 2.0 * p) * (suits == 3)\n    export_danger += (lengths[suits] <= 2) * (ranks >= 9) * (1.0 + p)\n    support_gift = -ranks.astype(jnp.float32) - (0.2 + p) * (suits == 3)\n    support_gift += (family == 1) * (lengths[suits] == 1) * 0.2\n    score = jnp.where(state.bids[me] == -1, export_danger, support_gift)\n    score += 0.015 * jax.random.uniform(key, (52,))\n    chosen = jnp.argmax(jnp.where(legal[:52], score, -1.e9))\n    return jnp.where(jnp.any(legal[:52]), chosen, 66).astype(jnp.int32)\n\n\ndef _diversity_bid_action(state, key, family, variant, legal):\n    me = actor(state)\n    estimate, nil_risk = estimate_bid_and_nil_risk(_diversity_acting_hand(state))\n    parameters = diversity_style_parameters(family, variant)\n    bias = parameters[\'bid_bias\']\n    bid = jnp.clip(jnp.floor(estimate + 0.35 + bias), 1, 13).astype(jnp.int32)\n    partner_bid = state.bids[(me + 2) % 4]\n    bid = jnp.minimum(bid, jnp.maximum(1, 13 - jnp.maximum(partner_bid, 0)))\n    threshold = 0.75 * parameters[\'risk_scale\']\n    threshold *= jnp.where((partner_bid == 0) | (partner_bid == -1), 0.25, 1.)\n    chosen = jnp.where(nil_risk < threshold, 52, 52 + bid)\n    return jnp.where(legal[66], 66, chosen).astype(jnp.int32)\n\n\ndef _diversity_play_action(state, key, family, variant, legal):\n    f = _diversity_public_play_features(state)\n    p = (variant.astype(jnp.float32) + 0.5) / 64.\n    score = jax.lax.switch(family, (_diversity_trump_management, _diversity_void_building,\n        _diversity_tempo_ducking, _diversity_crossruff_pressure,\n        _diversity_score_threshold_defense), f, p)\n    parameters = diversity_style_parameters(family, variant)\n    score += (parameters[\'urgency_scale\'] - 1.0) * f[\'wins\'] * (f[\'our_need\'] > 0)\n    # Shared contract safeguards use only public bids/tricks/table and our hand.\n    score += f[\'partner_nil\'] * f[\'wins\'] * (5.5 + p)\n    score += f[\'opposing_nil_winning\'] * (~f[\'wins\']) * 5.0\n    danger = f[\'rank\'] + 0.65 * (f[\'suits\'] == 3) + 0.25 * f[\'master\']\n    nil_score = -9.0 * f[\'wins\'] + (1.1 + 0.5 * p) * danger * (~f[\'wins\'])\n    # A lead provisionally wins regardless of rank. Score its exposure directly\n    # so live nil leads do not collapse to a constant followed by tie noise.\n    nil_lead_score = -3.0 * f[\'rank\'] - 2.0 * (f[\'suits\'] == 3) - 0.5 * f[\'master\']\n    nil_score = jnp.where(f[\'leading\'], nil_lead_score, nil_score)\n    score = jnp.where(f[\'own_nil\'], nil_score, score)\n    score += 0.025 * jax.random.uniform(key, (52,))\n    return jnp.argmax(jnp.where(legal[:52], score, -1.e9)).astype(jnp.int32)\n\n\ndef _diversity_nonblind_action(state, key, family, variant):\n    legal = legal_mask(state)\n    chosen = jax.lax.switch(jnp.maximum(phase(state) - 1, 0),\n        (_diversity_pass_action, _diversity_bid_action, _diversity_play_action),\n        state, key, family, variant, legal)\n    return jnp.where(legal[chosen], chosen, jnp.argmax(legal)).astype(jnp.int32)\n\n\ndef diverse_heuristic_action(state, key, style):\n    """Exact legacy delegation or legal new action; invalid IDs yield-1.\n\n    Invalid IDs are a fail-fast sentinel, never silently clamped to a valid\n    opponent. Native/runtime boundaries must reject them before workloads.\n    """\n    family, variant, valid = decode_opponent_style_id(style)\n    def new_action(_):\n        return jax.lax.cond(phase(state) == 0, _diversity_blind_action,\n                           _diversity_nonblind_action, state, key, family, variant)\n    chosen = jax.lax.cond(family == -1,\n        lambda _: heuristic_action(state, key, style), new_action, operand=None)\n    return jnp.where(valid, chosen, jnp.int32(-1)).astype(jnp.int32)\n'
