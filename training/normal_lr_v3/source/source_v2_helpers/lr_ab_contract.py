"""Fixed-source1201 opponent-expansion LR A/B contract. Host-only.

Production admission always uses the original byte-pinned source1201 roots.
The separate semantic functions support explicitly synthetic host tests only.
"""
import hashlib,importlib.util,io,json,math,re,types,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LEGACY_AUDITOR_SHA256='a354a362dad26b68dc8ad3030654adc9d7646a362b2a0ca024b7977ce04b9fb7'
SOURCE_PACKAGE_BYTES=34691129
SOURCE_PACKAGE_SHA256='865ee9cf3ee5f288bacdd0d092238daa9eeef81e4a8f887384f76f2ffaa75d0f'
SOURCE_NATIVE_BYTES=42929088
SOURCE_NATIVE_SHA256='93504f4d9dfb7080ff08b6eacf544aa55971997c9a45a1d06e8d111307fbf68e'
SOURCE_UPDATES=1201
SOURCE_ENV_STEPS=550961152
DECISIONS_PER_UPDATE=458752
DURATION_SECONDS=6600
ARM_TRAINING_SECONDS=1800
PLANNED_ROLLOUTS=128
LR_FACTOR=1.5
SOURCE_NATIVE_REVISION='strong-teacher-hand-gae-v3+opponent-diversity-v1+advantage-gated-anchor-v1'
NATIVE_REVISION=SOURCE_NATIVE_REVISION+'+opponent-training-expansion-v2'
TRAINING_PROFILE='training_expansion_v2'
AUTH_CAMPAIGN_ID='spades-source1201-expansion3-lr150-ab-20261020-v2'
AUTH_RUN_ID='source1201-expansion3-lr150-ab-20261020-v2'
AUTH_CREATED_UTC='2026-10-20T00:18:37Z'
AUTH_ACTIVATION_DEADLINE_UTC='2026-10-21T00:18:37Z'
EXPANSION_MANIFEST_SHA256='2497c7fd54ee54ed26a3131ece87addbf4c2032170c7b1a7434528e6ea2781d4'
EXPANSION_SOURCE_SHA256='71de314f6258d236ba2b2c9cf9f527182d11b3670038fd91c74df147656b143f'
TRAIN_STYLE_IDS=tuple(range(13))+tuple(13+64*f+v for f in (0,1,2,5,6,7) for v in range(32))
HELDOUT_STYLE_IDS=tuple(13+64*f+v for f in (3,4) for v in range(32,64))
SAMPLING_SHA256='824c41ee9318509901af82a1ca99d4814c88e2ac8112efcbc03cf9e13cc709a0'
AUTH_KEYS={'format','authorized','campaign_id','run_id','created_utc','activation_deadline_utc','authorization_reference','duration_seconds','arm_training_seconds','planned_rollouts_per_arm','lr_multiplier','automatic_extension','source_pins','arm_configuration_sha256','opponent_training_profile','training_profile_sha256'}
CONTEXT_KEYS={'utc_now','monotonic_now','host_boot_id','selected_gpu_uuid','selected_gpu_pci_bus_id'}
CLOCK_KEYS={'format','authorization_sha256','campaign_id','run_id','start_utc','start_monotonic','deadline_utc','deadline_monotonic','duration_seconds','host_boot_id','selected_gpu_uuid','selected_gpu_pci_bus_id','maximum_clock_drift_seconds'}

def require(condition,message):
    if not condition:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None

def integer(value,minimum=0):return type(value) is int and value>=minimum

def number(value):return type(value) in (int,float) and math.isfinite(value)

def utc(value):
    require(isinstance(value,str),'Explicit UTC timestamp required.');v=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(v.tzinfo is not None and v.utcoffset().total_seconds()==0,'Explicit UTC timezone required.');return v

def read_json_bytes(raw):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'Duplicate JSON key.');d[k]=v
        return d
    def bad(value):raise ValueError('Nonfinite JSON value.')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    def finite(v):
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
        elif isinstance(v,float):require(math.isfinite(v),'Nonfinite JSON number.')
    finite(value);return value

def read_json(path):return read_json_bytes(Path(path).read_bytes())

def file_identity(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'Regular nonsymlink input file required.');h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def content_identity(role,raw):return dict(role=role,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def _load_pinned(name,filename,expected):
    path=ROOT/filename;require(path.is_file() and not path.is_symlink(),'Pinned helper must be a regular source file.')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Pinned helper bytes differ:'+filename)
    module=types.ModuleType(name);module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module

def native_auditor():return _load_pinned('spades_fresh_frozen_native_auditor','audit_export_original.py',LEGACY_AUDITOR_SHA256)

def expansion_opponent_record():
    """New training identities; the frozen benchmark record is stored separately."""
    return dict(protocol='spades-training-opponent-expansion-v2',profile=TRAINING_PROFILE,manifest_sha256=EXPANSION_MANIFEST_SHA256,training_style_ids=list(TRAIN_STYLE_IDS),heldout_style_ids=list(HELDOUT_STYLE_IDS),native_league_style_leaf=156,old_families_unchanged=True,holdout_used_for_training=False,strength_evaluated=False)

def training_profile_metadata(cfg):
    """Assignment probabilities derive only from the actual saved configuration.

    Scripted match assignments persist until completion or truncation. This is
    neither a decision/time distribution nor proof that snapshots were present;
    unavailable historical assignments fall back to self-play in the source.
    """
    require(isinstance(cfg,dict) and all(k in cfg and number(cfg[k]) and 0<=cfg[k]<=1 for k in ('selfplay_prob','snapshot_prob')),'Actual saved source selfplay_prob/snapshot_prob required; defaults are forbidden.')
    selfplay=cfg['selfplay_prob'];snapshot=cfg['snapshot_prob'];require(selfplay+snapshot<=1,'Actual saved outer assignment probabilities must sum to at most one.')
    scripted=1.-selfplay-snapshot
    return dict(format='spades-opponent-training-profile-v2',profile=TRAINING_PROFILE,opponent_record=expansion_opponent_record(),scripted_sampling=dict(legacy_probability=.25,old_train_probability=.25,new_probability=.5,legacy_uniform_count=13,old_uniform_family_count=3,new_uniform_family_count=3,uniform_variants_per_family=32,assignment_scope='match'),saved_outer_configuration=dict(selfplay_prob=selfplay,snapshot_prob=snapshot),effective_assignment_probabilities=dict(selfplay=selfplay,historical_snapshot=snapshot,scripted=scripted,legacy_scripted=scripted*.25,old_family_scripted=scripted*.25,new_family_scripted=scripted*.5,each_legacy_style=scripted/52,each_old_family=scripted/12,each_old_family_variant=scripted/384,each_new_family=scripted/6,each_new_family_variant=scripted/192),historical_snapshot_unavailable_fallback='selfplay',assignment_scope='match; not decision or time fractions',frozen_benchmark_unchanged=True,playing_strength_claim=False)

def audit_training_style_array(array,cfg):
    import numpy as np
    require(array.shape==(cfg['n_env'],) and str(array.dtype)=='int32','Expanded native league_style leaf156 shape/dtype.')
    require(bool(np.isin(array,TRAIN_STYLE_IDS).all()),'Expanded native league_style contains heldout/reserved/negative/unknown training IDs.')

def _context(context):
    require(isinstance(context,dict) and set(context)==CONTEXT_KEYS,'Exact actual fresh execution context required.')
    require(number(context['monotonic_now']) and context['monotonic_now']>=0,'Finite nonnegative monotonic clock required.')
    require(isinstance(context['host_boot_id'],str) and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',context['host_boot_id']) is not None,'Canonical actual boot UUID required.')
    require(isinstance(context['selected_gpu_uuid'],str) and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',context['selected_gpu_uuid']) is not None,'Canonical selected physical GPU UUID required.')
    require(isinstance(context['selected_gpu_pci_bus_id'],str) and re.fullmatch(r'(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',context['selected_gpu_pci_bus_id']) is not None,'Actual selected physical PCI identity required.')
    utc(context['utc_now']);return context

def validate_lr_ab_authorization(auth,*,expected_source_pins=None):
    require(isinstance(auth,dict) and set(auth)==AUTH_KEYS and auth['format']=='spades-lr-ab-expansion-authorization-v2','Exact expanded LR A/B authorization schema required.')
    require(auth['authorized'] is True and auth['automatic_extension'] is False,'Explicit new LR A/B authorization without automatic extension required.')
    for k,wanted in [('duration_seconds',DURATION_SECONDS),('arm_training_seconds',ARM_TRAINING_SECONDS),('planned_rollouts_per_arm',PLANNED_ROLLOUTS)]:require(type(auth[k]) is int and auth[k]==wanted,'Fixed LR A/B duration/rollout budget differs:'+k)
    require(type(auth['lr_multiplier']) is float and auth['lr_multiplier']==LR_FACTOR,'Fixed1.5 learning-rate multiplier required.')
    for k in ('campaign_id','run_id'):require(isinstance(auth[k],str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',auth[k]) is not None,'Safe nonempty new campaign/run identity required.')
    require(auth['campaign_id']==AUTH_CAMPAIGN_ID and auth['run_id']==AUTH_RUN_ID and auth['created_utc']==AUTH_CREATED_UTC and auth['activation_deadline_utc']==AUTH_ACTIVATION_DEADLINE_UTC,'Only the dedicated newly authorized expansion campaign/run/window is admitted.')
    require(utc(auth['created_utc'])<utc(auth['activation_deadline_utc']) and isinstance(auth['authorization_reference'],str) and bool(auth['authorization_reference'].strip()),'Current grant window/reference required.')
    require(auth['opponent_training_profile']==TRAINING_PROFILE and sha(auth['training_profile_sha256']),'Explicit new expanded training-profile authorization required.')
    pins=auth['source_pins'];required={'original_sha256','production_sha256','operator_sha256','contract_sha256','evaluator_sha256','sampling_sha256','plasticity_sha256','training_opponents_sha256','training_opponent_manifest_sha256'}
    require(isinstance(pins,dict) and required<=set(pins) and all(isinstance(k,str) and re.fullmatch(r'[a-z][a-z0-9_]*_sha256',k) is not None and sha(v) for k,v in pins.items()),'Exact reviewed source pin map required.')
    require(pins['original_sha256']=='3236d2c95d4d1a95bf31a5cb7299853be6871783691829ee775fbb38366d1389' and pins['production_sha256']=='ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8' and pins['sampling_sha256']==SAMPLING_SHA256 and pins['contract_sha256']==file_identity(__file__)['sha256'],'Original/base sampling or actual LR A/B contract source differs.')
    require(pins['training_opponents_sha256']==EXPANSION_SOURCE_SHA256 and pins['training_opponent_manifest_sha256']==EXPANSION_MANIFEST_SHA256,'Reviewed exact expansion implementation/manifest pins differ.')
    if expected_source_pins is not None:require(canonical(pins)==canonical(expected_source_pins),'Caller-reviewed exact source pin map differs.')
    configs=auth['arm_configuration_sha256'];require(isinstance(configs,dict) and set(configs)=={'control','candidate'} and all(sha(v) for v in configs.values()) and configs['control']!=configs['candidate'],'Two actual distinct full arm configuration hashes required.')
    return read_json_bytes(canonical(auth))

def make_lr_ab_clock(auth,context):
    validate_lr_ab_authorization(auth);_context(context);start=utc(context['utc_now'])
    require(utc(auth['created_utc'])<=start<utc(auth['activation_deadline_utc']),'LR A/B grant activation window expired or not active.')
    return dict(format='spades-lr-ab-clock-v1',authorization_sha256=digest(auth),campaign_id=auth['campaign_id'],run_id=auth['run_id'],start_utc=context['utc_now'],start_monotonic=context['monotonic_now'],deadline_utc=(start+timedelta(seconds=DURATION_SECONDS)).isoformat(),deadline_monotonic=context['monotonic_now']+DURATION_SECONDS,duration_seconds=DURATION_SECONDS,host_boot_id=context['host_boot_id'],selected_gpu_uuid=context['selected_gpu_uuid'],selected_gpu_pci_bus_id=context['selected_gpu_pci_bus_id'],maximum_clock_drift_seconds=5)

def validate_lr_ab_clock(auth,clock):
    validate_lr_ab_authorization(auth)
    require(isinstance(clock,dict) and set(clock)==CLOCK_KEYS and clock['format']=='spades-lr-ab-clock-v1' and clock['authorization_sha256']==digest(auth) and clock['campaign_id']==auth['campaign_id'] and clock['run_id']==auth['run_id'],'Exact immutable LR A/B clock binding differs.')
    require(type(clock['duration_seconds']) is int and clock['duration_seconds']==DURATION_SECONDS and type(clock['maximum_clock_drift_seconds']) is int and clock['maximum_clock_drift_seconds']==5,'Fixed LR A/B total/drift budget differs.')
    start=utc(clock['start_utc']);require(utc(auth['created_utc'])<=start<utc(auth['activation_deadline_utc']) and utc(clock['deadline_utc'])==start+timedelta(seconds=DURATION_SECONDS),'Absolute LR A/B interval differs.')
    require(number(clock['start_monotonic']) and clock['start_monotonic']>=0 and number(clock['deadline_monotonic']) and abs(clock['deadline_monotonic']-clock['start_monotonic']-DURATION_SECONDS)<1e-6,'Monotonic LR A/B interval differs.')
    _context(dict(utc_now=clock['start_utc'],monotonic_now=clock['start_monotonic'],host_boot_id=clock['host_boot_id'],selected_gpu_uuid=clock['selected_gpu_uuid'],selected_gpu_pci_bus_id=clock['selected_gpu_pci_bus_id']));return clock

def validate_lr_ab_context(auth,clock,context,*,previous_context=None):
    validate_lr_ab_clock(auth,clock);_context(context);now=utc(context['utc_now']);mono=context['monotonic_now'];start=utc(clock['start_utc'])
    require(all(context[k]==clock[k] for k in ('host_boot_id','selected_gpu_uuid','selected_gpu_pci_bus_id')),'LR A/B boot/physical device identity changed.')
    require(start<=now<utc(clock['deadline_utc']) and clock['start_monotonic']<=mono<clock['deadline_monotonic'],'LR A/B total budget expired or clock rolled back.')
    require(abs((now-start).total_seconds()-(mono-clock['start_monotonic']))<=5.,'LR A/B UTC/monotonic drift exceeds five seconds.')
    if previous_context is not None:
        validate_lr_ab_context(auth,clock,previous_context)
        require(all(previous_context[k]==context[k] for k in ('host_boot_id','selected_gpu_uuid','selected_gpu_pci_bus_id')) and utc(previous_context['utc_now'])<=now and previous_context['monotonic_now']<=mono,'LR A/B persisted high-water rollback.')
    return dict(status='PASS_lr_ab_context',authorization_sha256=digest(auth),clock_sha256=digest(clock),context=dict(context))

def build_arm_configs(source_cfg):
    require(isinstance(source_cfg,dict) and type(source_cfg.get('lr')) is float and source_cfg['lr']==0.0002,'Actual source1201 lr must be exactly0.0002.')
    require(all(source_cfg.get(k)==v for k,v in dict(seed=42,n_env=3584,rollout_len=128,width=256,minibatch_size=57344,ppo_epochs=3,teacher_anchor_mode='advantage_gated_v1',anchor_conflict_floor=.25,anchor_advantage_scale=1.).items()),'Actual source1201 dimensions/adaptive configuration differ.')
    require('opponent_training_profile' not in source_cfg,'Source1201 must precede the newly authorized opponent profile.')
    training_profile_metadata(source_cfg)
    control=read_json_bytes(canonical(source_cfg));control['opponent_training_profile']=TRAINING_PROFILE
    candidate=read_json_bytes(canonical(control));candidate['lr']=0.0003
    return dict(control=control,candidate=candidate)

def arm_migration_record(source_cfg,arm_cfg,arm):
    require(arm in ('control','candidate') and canonical(arm_cfg)==canonical(build_arm_configs(source_cfg)[arm]),'Only shared opponent expansion plus fixed LR arm configuration allowed.')
    changes={'opponent_training_profile':dict(before=None,after=TRAINING_PROFILE)}
    if arm=='candidate':changes['lr']=dict(before=.0002,after=.0003)
    return dict(kind='lr_ab_configuration',format='spades-lr-ab-expansion-configuration-v2',arm=arm,source_cfg_sha256=digest(source_cfg),arm_cfg_sha256=digest(arm_cfg),changes=changes,lr_multiplier=1. if arm=='control' else LR_FACTOR,shared_opponent_training_profile=TRAINING_PROFILE,training_profile_sha256=digest(training_profile_metadata(source_cfg)),source_revision=SOURCE_NATIVE_REVISION,arm_revision=NATIVE_REVISION,all_other_configuration_unchanged=True,between_arm_only_lr_differs=True,optimizer_state_retained=True,rng_state_retained=True,parameter_reset=False)

def _native_content(raw,*,expected_cfg=None,source_manifest=None,expanded=False):
    """Pure content layer for both production admission and separate synthetic tests."""
    import numpy as np
    old=native_auditor();parts=old.zip_members(raw,('manifest.json','history.json','state.npz'));m=read_json_bytes(parts['manifest.json']);h=read_json_bytes(parts['history.json'])
    require(isinstance(m,dict) and type(m.get('format')) is int and m['format']==1 and m.get('ruleset')=='modified-spades-500-blind-swap-team-failure-v1','Native format/ruleset differs.')
    require(m.get('training_revision')==(NATIVE_REVISION if expanded else SOURCE_NATIVE_REVISION) and type(m.get('training_state_schema')) is int and m['training_state_schema']==2 and m.get('versions')==old.STACK,'Exact adaptive native revision/schema/stack required.')
    require(isinstance(m.get('cfg'),dict) and isinstance(h,list) and all(isinstance(r,dict) for r in h),'Native cfg/history schemas required.')
    if expanded:
        require(m['cfg'].get('opponent_training_profile')==TRAINING_PROFILE and canonical(m.get('opponent_diversity'))==canonical(expansion_opponent_record()) and canonical(m.get('frozen_opponent_benchmark'))==canonical(old.OPPONENT_DIVERSITY) and canonical(m.get('opponent_training_profile'))==canonical(training_profile_metadata(m['cfg'])),'Truthful expanded native training profile and separate frozen benchmark required.')
    else:
        require('opponent_training_profile' not in m['cfg'] and 'opponent_training_profile' not in m and 'frozen_opponent_benchmark' not in m and canonical(m.get('opponent_diversity'))==canonical(old.OPPONENT_DIVERSITY),'Exact original source1201 opponents required.')
    if expected_cfg is not None:require(canonical(m['cfg'])==canonical(expected_cfg),'Native full arm cfg differs.')
    require(all(integer(m.get(k)) for k in ('updates','env_steps')) and number(m.get('cumulative_seconds')) and m['cumulative_seconds']>=0,'Native counters/cumulative types differ.');utc(m.get('created_utc'))
    require(m.get('sha256')==hashlib.sha256(parts['state.npz']).hexdigest(),'Native NPZ hash differs.')
    specs=m.get('leaves');require(isinstance(specs,list) and len(specs)==214,'Native214 schema required.')
    if source_manifest is not None:require(canonical(specs)==canonical(source_manifest['leaves']),'Native214 source specs changed.')
    old.preflight_npz(parts['state.npz'],specs)
    with np.load(io.BytesIO(parts['state.npz']),allow_pickle=False) as arrays:
        require(sorted(arrays.files)==[f'leaf_{i:05d}' for i in range(214)],'Native214 unique arrays required.')
        for i,spec in enumerate(specs):
            a=arrays[f'leaf_{i:05d}'];require(list(a.shape)==spec['shape'] and str(a.dtype)==spec['dtype'] and not a.dtype.hasobject,'Native leaf shape/dtype differs:'+str(i));require(not np.issubdtype(a.dtype,np.inexact) or bool(np.isfinite(a).all()),'Nonfinite native leaf:'+str(i))
        (audit_training_style_array if expanded else old.audit_training_style_array)(arrays['leaf_00156'],m['cfg'])
        vals=[]
        for i,dtype in zip((207,208,209,210,211,212,213),('int32','int32','int32','bool','uint32','uint32','int32')):
            a=arrays[f'leaf_{i:05d}'];require(a.shape==() and str(a.dtype)==dtype,'Native lifetime counter schema.');vals.append(a.item())
        update,decisions,bootstrap,complete,low,high,base=vals
        require(update==m['updates'] and decisions==DECISIONS_PER_UPDATE and bootstrap==0 and complete is True and 0<=base<=update and ((int(high)<<32)|int(low))+(int(update)-int(base))*int(decisions)==m['env_steps'],'Native actual lifetime counters differ.')
    return m,h

def _health(rows):
    window=rows[-3:];planned=sum(r['planned'] for r in window);accepted=sum(r['accepted'] for r in window)
    consecutive=(len(window)==3 and all(window[i]['update']+1==window[i+1]['update'] for i in range(2)) and all(r['cadence']==window[0]['cadence'] and r['config_signature']==window[0]['config_signature'] for r in window) and not any(r.get('continuity_break_after',False) for r in window[:-1]))
    allkl=bool(consecutive and all(r['kl_early_stop'] for r in window))
    return dict(window_rollouts=len(window),planned=planned,accepted=accepted,aggregate_accepted_fraction=accepted/planned if planned else 0.,consecutive_same_cadence=bool(consecutive),all_three_kl_stopped=allkl,should_stop=bool(allkl and 4*accepted<planned))

def _work_audit(history,cfg,start_update,end_update):
    cadence={k:cfg[k] for k in ('n_env','rollout_len','minibatch_size','ppo_epochs')};planned=cadence['ppo_epochs']*(cadence['n_env']*cadence['rollout_len']//cadence['minibatch_size']);work=[];metrics=[];prior=[];stopped=False
    for row in history:
        kind=row.get('kind');u=row.get('update')
        if kind=='optimizer_work' and type(u) is int and u<=start_update:prior.append(row);continue
        if kind not in ('optimizer_work','aot_raw_metrics','optimizer_health_stop') or not(type(u) is int and u>start_update):continue
        require(u<=end_update,'Work/metric update exceeds native state.')
        if kind=='optimizer_work':
            require(not stopped,'Optimizer continued after unacknowledged health stop.')
            require(type(row.get('optimizer_work_version')) is int and row['optimizer_work_version']==1 and canonical(row.get('cadence'))==canonical(cadence) and row.get('config_signature')==digest(cfg),'Actual work cfg/cadence/version differs.')
            require(all(integer(row.get(k)) for k in ('env_steps','update','planned','attempted','accepted','skipped','rejected')),'Work exact counts required.')
            require(row['planned']==planned and 1<=row['accepted']==row['attempted']<=planned and row['rejected']==0 and row['skipped']==planned-row['accepted'] and type(row.get('kl_early_stop')) is bool and (row['accepted']==planned or row['kl_early_stop']),'Rejected/inconsistent optimizer work.')
            require(row['env_steps']==SOURCE_ENV_STEPS+(row['update']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE,'Work lifetime decisions differ.')
            for k,v in [('attempted_fraction',row['attempted']/planned),('accepted_fraction',row['accepted']/planned),('skipped_fraction',row['skipped']/planned)]:require(number(row.get(k)) and row[k]==v,'Work fraction differs:'+k)
            work.append(row)
        elif kind=='aot_raw_metrics':
            c=row.get('collection');o=row.get('optimizer');require(isinstance(c,dict) and isinstance(o,dict) and all(number(v) for v in c.values()) and all(number(v) for v in o.values()) and c.get('illegal_actions')==0,'Raw metrics illegal/nonfinite.')
            require(all(o.get(k)==1 for k in ('finite','diagnostics_finite','gradients_finite','grad_norm_finite','params_finite','optimizer_finite')),'Raw optimizer finite gate failed.');metrics.append(row)
        else:
            health=_health(prior+work);stopid=hashlib.sha256(json.dumps(dict(update=u,signature=digest(cfg),health=health),sort_keys=True).encode()).hexdigest()
            expected=dict(kind='optimizer_health_stop',optimizer_health_version=1,update=u,config_signature=digest(cfg),stop_id=stopid,reason='Three consecutive KL-stopped rollouts accepted less than25percent of planned work.',health=health)
            require(not stopped and health['should_stop'] and canonical(row)==canonical(expected),'Exact optimizer health-stop record differs.');stopped=True
    expected=list(range(start_update+1,end_update+1));require([r['update'] for r in work]==expected and [r['update'] for r in metrics]==expected,'Actual work/raw metric coverage missing, duplicate or reordered.')
    for w,r in zip(work,metrics):
        o=r['optimizer'];require(o.get('attempted_updates')==w['attempted'] and o.get('accepted_updates')==w['accepted'] and o.get('rejected_updates')==0 and o.get('kl_early_stop')==int(w['kl_early_stop']),'Raw/work accounting disagrees.')
    health=_health(prior+work)
    return dict(accepted_rollouts=len(work),accepted_minibatches=sum(w['accepted'] for w in work),raw_metrics=len(metrics),health=health,health_stop_recorded=stopped,last_metrics=dict(update=work[-1]['update'],collection=metrics[-1]['collection'],optimizer=metrics[-1]['optimizer'],work=work[-1]) if work else None)

def _zip_content(raw,*,core,prefixes=()):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos=z.infolist();names=[i.filename for i in infos];require(len(names)==len(set(names)) and set(core)<=set(names),'ZIP required members/duplicates differ.')
        for i in infos:
            n=i.filename;require(n in core or any(n.startswith(p) for p in prefixes),'Unknown ZIP member:'+n)
            require(not n.startswith('/') and '\\' not in n and all(v not in ('','..','.') for v in n.split('/')) and not i.is_dir() and not(i.flag_bits&1) and (i.external_attr>>16)&0o170000!=0o120000,'Unsafe ZIP member path/type/encryption.')
            require(i.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and i.file_size<=256*1024**2,'Unexpected ZIP member compression/size.')
        require(sum(i.file_size for i in infos)<=1536*1024**2 and z.testzip() is None,'ZIP total bound or CRC differs.');return {n:z.read(n) for n in names}

def _check_member_hashes(parts,manifest):
    members=manifest.get('members');require(isinstance(members,dict) and set(members)==set(parts)-{'manifest.json'},'ZIP member hash manifest differs.')
    for n,r in members.items():require(isinstance(r,dict) and set(r)=={'bytes','sha256'} and type(r['bytes']) is int and r['bytes']==len(parts[n]) and r['sha256']==hashlib.sha256(parts[n]).hexdigest(),'Member identity differs:'+n)

def validate_source_artifacts(source1201_bundle_path):
    """Production root admission has no custom expected roots or bypass switch.

    Prior strict evaluation-archive PASS is deliberately not inferred. This is
    independent content validation of the exact trusted current checkpoint.
    """
    raw=Path(source1201_bundle_path).read_bytes()
    require(len(raw)==SOURCE_PACKAGE_BYTES and hashlib.sha256(raw).hexdigest()==SOURCE_PACKAGE_SHA256,'Exact actual source1201 bundle trust root required.')
    parts=_zip_content(raw,core={'manifest.json','status.json','authorization.json','provenance.json','native_checkpoint.zip'},prefixes=('evidence/',));x=read_json_bytes(parts['manifest.json']);s=read_json_bytes(parts['status.json']);p=read_json_bytes(parts['provenance.json']);a=read_json_bytes(parts['authorization.json']);_check_member_hashes(parts,x)
    native=parts['native_checkpoint.zip'];require(len(native)==SOURCE_NATIVE_BYTES and hashlib.sha256(native).hexdigest()==SOURCE_NATIVE_SHA256,'Exact actual source1201 native trust root required.')
    m,h=_native_content(native);configs=build_arm_configs(m['cfg'])
    require(x.get('format')=='spades-fresh-onehour-bundle-v1' and x.get('reason')==x.get('checkpoint_role')=='final' and s.get('terminal') is True and x.get('updates')==s.get('updates')==m['updates']==SOURCE_UPDATES and x.get('env_steps')==s.get('env_steps')==m['env_steps']==SOURCE_ENV_STEPS,'Source1201 final native/status counters differ.')
    require(canonical(s.get('cfg'))==canonical(m['cfg'])==canonical(p.get('candidate_cfg')) and a.get('configuration_sha256')==digest(m['cfg']),'Source1201 full saved adaptive configuration differs.')
    require(h and h[-1].get('kind')=='fresh_checkpoint' and h[-1].get('reason')=='final' and h[-1].get('updates')==SOURCE_UPDATES and h[-1].get('env_steps')==SOURCE_ENV_STEPS and h[-1].get('terminal') is True,'Source1201 final history barrier differs.')
    journal=_work_audit(h,m['cfg'],906,SOURCE_UPDATES)
    require(not journal['health']['should_stop'] and not journal['health_stop_recorded'],'Source1201 original-config health stop requires separate review before any LR migration.')
    require(canonical(s.get('last_metrics'))==canonical(journal['last_metrics']),'Source1201 native/status last metrics differ.')
    ids=dict(source1201_package=content_identity('source1201_package',raw),source1201_native=content_identity('source1201_native',native))
    record=dict(package_identity=ids['source1201_package'],native_identity=ids['source1201_native'],native_bytes=native,native_manifest=m,native_history=h,external_manifest=x,status=s,provenance=p,authorization=a)
    return dict(source1201=record,source_cfg=m['cfg'],source_identities=ids,source_history_sha256=digest(h),source_health=journal['health'],source_journal=journal,historical_clock=x.get('fresh_run_clock'),source_validation_scope='PINNED_CURRENT_CHECKPOINT_CONTENT_ONLY',prior_strict_evaluation_archive_pass_inferred=False,prior_evaluation_evidence_scope='Prior archive gap and separately derived policy-digest/rebind audit are distinct; neither is replaced by this checkpoint-content check.')

def validate_arm_native(raw,*,source_artifacts,arm,expected_config=None):
    cfg=build_arm_configs(source_artifacts['source_cfg'])[arm] if arm in ('control','candidate') else None;require(cfg is not None,'Known LR arm required.')
    if expected_config is not None:require(canonical(expected_config)==canonical(cfg),'Caller full arm configuration differs.')
    source=source_artifacts['source1201'];m,h=_native_content(raw,expected_cfg=cfg,source_manifest=source['native_manifest'],expanded=True)
    require(SOURCE_UPDATES<=m['updates']<=SOURCE_UPDATES+PLANNED_ROLLOUTS and m['env_steps']==SOURCE_ENV_STEPS+(m['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE,'Arm counter/128-rollout bound differs.')
    require(m['cumulative_seconds']>=source['native_manifest']['cumulative_seconds'],'Arm cumulative time rolled back.')
    native_auditor().continuation_arrays(raw,source['native_bytes'],m['updates']==SOURCE_UPDATES)
    return m,h

def _audit_arm_history(history,source,arm,cfg,auth,clock):
    base=source['source1201']['native_history'];require(canonical(history[:len(base)])==canonical(base),'Immutable exact source1201 history prefix changed.')
    tail=history[len(base):];expected=dict(kind='lr_ab_branch_start',arm=arm,authorization_sha256=digest(auth),lr_ab_clock=clock,source_identities=source['source_identities'],migration=arm_migration_record(source['source_cfg'],cfg,arm))
    require(len(tail)>=2 and canonical(tail[0])==canonical(expected),'Exact LR branch-start/source/migration record required.')
    require(all(r.get('kind') in ('optimizer_work','aot_raw_metrics','optimizer_health_stop','lr_ab_checkpoint') for r in tail[1:]),'Unexpected LR branch history row.')
    for row in tail[1:]:
        if row.get('kind')=='lr_ab_checkpoint':require(row.get('arm')==arm and canonical(row.get('lr_ab_clock'))==canonical(clock) and integer(row.get('updates'),SOURCE_UPDATES) and row['updates']<=SOURCE_UPDATES+PLANNED_ROLLOUTS and row.get('env_steps')==SOURCE_ENV_STEPS+(row['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and isinstance(row.get('nonce'),str) and bool(row['nonce']) and type(row.get('terminal')) is bool,'LR checkpoint history counters/clock/arm differ.')
    require(tail[-1].get('kind')=='lr_ab_checkpoint','Native must end at immutable LR checkpoint record.')
    return tail[-1]

def _audit_bundle_parts(parts,source,*,expected_authorization=None):
    """Pure semantic layer; production entry always first admits fixed source bytes."""
    x=read_json_bytes(parts['manifest.json']);s=read_json_bytes(parts['status.json']);p=read_json_bytes(parts['provenance.json']);a=read_json_bytes(parts['authorization.json']);validate_lr_ab_authorization(a);_check_member_hashes(parts,x)
    if expected_authorization is not None:require(canonical(a)==canonical(expected_authorization),'Caller reviewed LR A/B authorization differs.')
    keys={'format','campaign_id','run_id','nonce','reason','phase','evaluated_native_sha256','members'}
    require(isinstance(x,dict) and set(x)==keys and x['format']=='spades-lr-ab-expansion-bundle-v2','Exact expanded LR A/B portable manifest required.')
    require(isinstance(s,dict) and isinstance(p,dict) and s.get('format')=='spades-lr-ab-expansion-status-v2','Expanded LR A/B status/provenance required.')
    for k in ('campaign_id','run_id'):require(x[k]==s.get(k)==a[k],'LR A/B campaign/run binding differs.')
    require(isinstance(x['nonce'],str) and bool(x['nonce']) and isinstance(x['reason'],str) and bool(x['reason']) and x['phase']==s.get('phase'),'LR A/B phase/nonce/reason missing.')
    require(s.get('nonce')==x['nonce'] and s.get('reason')==x['reason'],'Actual status/manifest nonce or reason differs.')
    clock=validate_lr_ab_clock(a,s.get('lr_ab_clock'));require(canonical(p.get('lr_ab_clock'))==canonical(clock),'LR A/B provenance clock differs.')
    require(type(s.get('terminal')) is bool and s.get('training_enabled') is (not s['terminal']) and s.get('promotion') is False and s.get('playing_strength_claim') is False,'LR A/B terminal/no-promotion fields differ.')
    elapsed=s.get('elapsed_seconds');require(number(elapsed) and elapsed>=0,'Actual LR A/B elapsed duration required.')
    configs=build_arm_configs(source['source_cfg']);require(canonical(s.get('source_cfg'))==canonical(source['source_cfg']) and canonical(s.get('arm_configs'))==canonical(configs) and a['arm_configuration_sha256']=={arm:digest(cfg) for arm,cfg in configs.items()},'Exact source/two-arm authorized configs differ.')
    profile=training_profile_metadata(source['source_cfg']);require(canonical(s.get('training_profile'))==canonical(profile) and canonical(p.get('training_profile'))==canonical(profile) and a['training_profile_sha256']==digest(profile),'Actual saved-cfg expanded training profile/proportions differ.')
    require(p.get('format')=='spades-lr-ab-expansion-provenance-v2' and canonical(p.get('source_identities'))==canonical(source['source_identities']) and p.get('source_history_sha256')==source['source_history_sha256'],'Pinned source1201 portable lineage differs.')
    require(canonical(p.get('source_cfg'))==canonical(source['source_cfg']) and canonical(p.get('arm_configs'))==canonical(configs) and canonical(p.get('source_health'))==canonical(source['source_health']) and p.get('historical_ledgers_unchanged') is True and p.get('optimizer_and_rng_reset') is False and p.get('playing_strength_claim') is False,'Truthful source cfg/health/no-reset provenance required.')
    platform=native_auditor().audit_platform(p.get('platform'),dict(selected_gpu_uuid=clock['selected_gpu_uuid']));require(platform['selected_gpu_pci_bus_id']==clock['selected_gpu_pci_bus_id'],'New Kaggle platform/clock PCI differs.')
    pin_names=dict(operator_sha256='spades_lr_ab_run.py',contract_sha256='lr_ab_contract.py',evaluator_sha256='lr_ab_evaluation.py',plasticity_sha256='plasticity.py',helper_sha256='pinned_fresh_helpers.py',legacy_operator_sha256='spades_legacy_aot.py',original_sha256='original_source.ipynb',production_sha256='production_source.ipynb',frozen_evaluator_sha256='diversity_evaluation.py',sampling_sha256='sampling_policy.py',training_opponents_sha256='training_opponents_v2.py',training_opponent_manifest_sha256='training_opponent_manifest_v2.json')
    records=p.get('source_pins');require(isinstance(records,dict),'Actual reviewed source file identities required.')
    for key,value in a['source_pins'].items():require(key in pin_names and isinstance(records.get(pin_names[key]),dict) and integer(records[pin_names[key]].get('bytes'),1) and records[pin_names[key]].get('sha256')==value,'Actual source/auth pin differs:'+key)
    source_native=parts.get('checkpoints/source1201_native.zip');require(source_native==source['source1201']['native_bytes'],'Exact source1201 native must be portable in every bundle.')
    arm_records=s.get('arm_records');require(isinstance(arm_records,dict) and set(arm_records)<={'control','candidate'},'Known LR A/B arm records required.')
    require(canonical(p.get('branch_214_equalities'))==canonical({arm:r.get('equality') for arm,r in arm_records.items()}),'Provenance/arm initial214 equality proofs differ.')
    actual={};arm_reports={};unsaved_arms={}
    for arm,record in arm_records.items():
        require(isinstance(record,dict) and canonical(record.get('cfg'))==canonical(configs[arm]),'Arm full recorded cfg differs.')
        require(integer(record.get('updates'),SOURCE_UPDATES) and record['updates']<=SOURCE_UPDATES+PLANNED_ROLLOUTS and record.get('env_steps')==SOURCE_ENV_STEPS+(record['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and type(record.get('completed_rollouts')) is int and record['completed_rollouts']==record['updates']-SOURCE_UPDATES,'Reported arm RAM counter schema/bounds differ.')
        name='checkpoints/'+arm+'_native.zip';info=record.get('native')
        if info is None:
            require(name not in parts and record.get('accepted_state_saved') is False and record.get('durable_updates') is None and record.get('durable_env_steps') is None,'Unrecorded/never-saved arm native metadata differs.')
            unsaved_arms[arm]=dict(latest_reported_ram_updates=record.get('updates'),latest_reported_ram_env_steps=record.get('env_steps'),ram_state_independently_verified=False,accepted_state_saved=False);continue
        require(name in parts and isinstance(info,dict) and type(info.get('bytes')) is int and info['bytes']==len(parts[name]) and info.get('sha256')==hashlib.sha256(parts[name]).hexdigest(),'Exact portable current arm native identity differs.')
        m,h=validate_arm_native(parts[name],source_artifacts=source,arm=arm);last=_audit_arm_history(h,source,arm,configs[arm],a,clock)
        require(last['updates']==m['updates'] and last['env_steps']==m['env_steps'] and record.get('durable_updates')==m['updates'] and record.get('durable_env_steps')==m['env_steps'],'Saved arm native/history/durable counters differ.')
        require(integer(record.get('updates'),m['updates']) and record['updates']<=SOURCE_UPDATES+PLANNED_ROLLOUTS and record.get('env_steps')==SOURCE_ENV_STEPS+(record['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and type(record.get('completed_rollouts')) is int and record['completed_rollouts']==record['updates']-SOURCE_UPDATES,'Latest reported arm RAM counters differ.')
        current_saved=record['updates']==m['updates'];require(type(record.get('accepted_state_saved')) is bool and record['accepted_state_saved']==current_saved,'Latest accepted-state durability must be explicit and truthful.')
        equality=dict(native_state_leaves=214,all_array_bytes_equal=True,source_native_sha256=source['source1201']['native_identity']['sha256']);require(canonical(record.get('equality'))==canonical(equality),'Exact initial214 restoration proof differs.')
        require(canonical(record.get('source_health'))==canonical(source['source_health']) and canonical(record.get('migration'))==canonical(arm_migration_record(source['source_cfg'],configs[arm],arm)),'Original-config health or shared expansion/LR migration differs.')
        work=_work_audit(h,configs[arm],SOURCE_UPDATES,m['updates'])
        if current_saved:require(canonical(record.get('last_metrics'))==canonical(work['last_metrics']) and canonical(record.get('health'))==canonical(work['health']),'Saved arm actual raw metrics/rolling health differs.')
        else:unsaved_arms[arm]=dict(latest_reported_ram_updates=record['updates'],latest_reported_ram_env_steps=record['env_steps'],ram_state_independently_verified=False,accepted_state_saved=False)

        t=record.get('training_clock');require(isinstance(t,dict) and set(t)=={'start_monotonic','stop_monotonic','elapsed_seconds','maximum_seconds'} and type(t['maximum_seconds']) is int and t['maximum_seconds']==ARM_TRAINING_SECONDS and number(t['elapsed_seconds']) and t['elapsed_seconds']>=0,'Exact1800-second arm training clock required.')
        if t['start_monotonic'] is None:require(t['stop_monotonic'] is None and t['elapsed_seconds']==0 and m['updates']==SOURCE_UPDATES,'Unstarted arm cannot have work or elapsed time.')
        else:
            require(number(t['start_monotonic']) and clock['start_monotonic']<=t['start_monotonic']<clock['deadline_monotonic'],'Arm clock starts outside global budget.')
            if t['stop_monotonic'] is not None:require(number(t['stop_monotonic']) and t['stop_monotonic']>=t['start_monotonic'] and abs(t['stop_monotonic']-t['start_monotonic']-t['elapsed_seconds'])<=1e-6,'Stopped arm elapsed accounting differs.')
        added=m['cumulative_seconds']-source['source1201']['native_manifest']['cumulative_seconds'];require(0<=added<=elapsed+1e-6,'Arm cumulative time outside source/experiment interval.')
        actual[arm]=hashlib.sha256(parts[name]).hexdigest();arm_reports[arm]=dict(updates=m['updates'],env_steps=m['env_steps'],completed_rollouts=m['updates']-SOURCE_UPDATES,latest_reported_ram_updates=record['updates'],accepted_state_saved=current_saved,work=work,training_clock=t,initial214_array_bytes_verified=m['updates']==SOURCE_UPDATES,lifetime_base_counter_bytes_unchanged=True)
    expected_checkpoint_names={'checkpoints/source1201_native.zip'}|{'checkpoints/'+arm+'_native.zip' for arm,record in arm_records.items() if record.get('native') is not None}
    require({name for name in parts if name.startswith('checkpoints/')}==expected_checkpoint_names,'Unknown native or native exists without matching durable arm record.')
    evaluated=x['evaluated_native_sha256'];require(isinstance(evaluated,dict),'Explicit evaluated-native map required.')
    evaluation=None;incomplete_evaluation=None
    if evaluated:
        require(set(actual)=={'control','candidate'} and all(r['completed_rollouts']==PLANNED_ROLLOUTS and r['accepted_state_saved'] for r in arm_reports.values()),'Both exact128-rollout endpoints required before evaluation.')
        expected={'source':hashlib.sha256(source_native).hexdigest(),**actual};require(evaluated==expected,'Evaluated checkpoint identities differ from exact enclosed endpoint natives.')
        archives=[(n,b) for n,b in parts.items() if n.startswith('evidence/') and n.endswith('.zip') and _is_lr_ab_evidence(b)]
        require(len(archives)<=1,'Duplicate portable LR A/B evaluation evidence archives.')
        if not archives:
            require(not(isinstance(s.get('evaluation'),dict) and s['evaluation'].get('complete') is True),'Claimed complete evaluation lacks portable evidence.')
            incomplete_evaluation='Intended exact checkpoint selection preserved; evaluation archive not yet available.'
        else:
            name,evidence=archives[0]
            with zipfile.ZipFile(io.BytesIO(evidence)) as z:
                entries=z.infolist();names=[i.filename for i in entries]
                require(len(names)==len(set(names)) and all('/' not in n and '\\' not in n and n not in ('.','..') for n in names) and all(i.file_size<=256*1024**2 for i in entries) and sum(i.file_size for i in entries)<=512*1024**2 and z.testzip() is None,'Portable evaluation ZIP bounds/names/CRC differ.')
                em=read_json_bytes(z.read('evaluation_archive_manifest.json'));members=em.get('members');require(isinstance(members,dict) and set(names)==set(members)|{'evaluation_archive_manifest.json'},'Portable evaluation transport member coverage differs.')
                for member,info in members.items():
                    body=z.read(member);require(isinstance(info,dict) and set(info)=={'bytes','sha256'} and type(info['bytes']) is int and info['bytes']==len(body) and info['sha256']==hashlib.sha256(body).hexdigest(),'Portable evaluation transport hash differs:'+member)
                require(z.read('source_native.zip')==source_native and z.read('control_native.zip')==parts['checkpoints/control_native.zip'] and z.read('candidate_native.zip')==parts['checkpoints/candidate_native.zip'],'EXACT evaluated native ZIP bytes differ from exported endpoints.')
                evaluation_report=read_json_bytes(z.read('evaluation_report.json'))
            if evaluation_report.get('complete') is not True:
                require(not(isinstance(s.get('evaluation'),dict) and s['evaluation'].get('complete') is True),'Status claims complete evaluation but archive is partial.')
                incomplete_evaluation='Partial evaluation retained; transport identities only, no outcome inference.'
            else:
                module=_load_pinned('spades_lr_ab_portable_evaluation','lr_ab_evaluation.py',a['source_pins']['evaluator_sha256'])
                import tempfile
                with tempfile.TemporaryDirectory(prefix='spades_lr_ab_content_') as folder:
                    path=Path(folder)/'evaluation.zip';path.write_bytes(evidence)
                    evaluation=module.audit_lr_ab_evaluation(path,expected_experiment_id=a['campaign_id'],expected_run_id=a['run_id'],expected_arm_config_signatures=a['arm_configuration_sha256'],frozen_source_path=ROOT/'diversity_evaluation.py')
                require(evaluation.get('status')=='PASS' and evaluation.get('exact_evaluated_natives_embedded') is True,'Complete portable LR A/B evaluation audit failed.')
    complete_claim=(x['phase']=='complete' or s.get('stop_reason')=='bounded_lr_ab_complete' or (isinstance(s.get('evaluation'),dict) and s['evaluation'].get('complete') is True))
    require(not complete_claim or (evaluation is not None and not unsaved_arms and s['terminal'] is True),'Claimed completed LR A/B experiment requires full1536-game portable audit and fully saved endpoints.')
    return dict(status='PASS',audit_scope='CONTENT_ONLY',verified_content=True,ack_authorized=False,library_upload_verified=False,launch_authorized=False,resumable=False,automatic_extension=False,playing_strength_claim=False,lr_generalization_claim=False,campaign_id=a['campaign_id'],run_id=a['run_id'],source_identities=source['source_identities'],source_validation_scope=source.get('source_validation_scope','PINNED_CURRENT_CHECKPOINT_CONTENT_ONLY'),prior_strict_evaluation_archive_pass_inferred=False,authorization_sha256=digest(a),lr_ab_clock=clock,training_profile=profile,arm_configuration_sha256=a['arm_configuration_sha256'],arms=arm_reports,unsaved_reported_ram_states=unsaved_arms,all_reported_accepted_states_saved=not unsaved_arms,incomplete_evaluation_scope=incomplete_evaluation,lr_ab_evaluation_complete=evaluation is not None,evaluation=evaluation,exact_evaluated_natives_embedded=evaluation is not None,metadata_attestation_scope='Content only; recorded clocks/platform/initial GPU restore proof do not independently attest execution.')

def _is_lr_ab_evidence(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:return read_json_bytes(z.read('evaluation_archive_manifest.json')).get('format')=='spades-lr-ab-expansion-per-arm-evidence-v2'
    except (ValueError,KeyError,zipfile.BadZipFile):return False

def audit_lr_ab_bundle(bundle_path,*,source1201_bundle_path,expected_authorization=None):
    source=validate_source_artifacts(source1201_bundle_path)
    raw=Path(bundle_path).read_bytes();parts=_zip_content(raw,core={'manifest.json','status.json','authorization.json','provenance.json','checkpoints/source1201_native.zip'},prefixes=('checkpoints/','evidence/'))
    result=_audit_bundle_parts(parts,source,expected_authorization=expected_authorization);result['bundle']=content_identity('lr_ab_bundle',raw);return result
