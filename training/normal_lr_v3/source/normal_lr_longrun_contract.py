"""Host-only source, native-state and durable-budget contract for normal LR v3.

Journal validation checks content and continuity. It cannot independently attest
that a JSON head is the current head of an external store. Production launch
also requires an independently retrieved, externally acknowledged current claim.
"""
import hashlib,io,json,math,re,types,zipfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ACTIVE_TRAINING_SECONDS=43200
SEGMENT_ACTIVE_SECONDS=21600
MAX_SEGMENTS=3
LEGACY_MAX_SEGMENTS=2
LEGACY_CONTRACT_FILENAME='legacy_normal_lr_longrun_contract_v3.py'
LEGACY_CONTRACT_SHA256='0383c488fbd5cf023a254e913f80fef0d31bedc906883ce24d06efa6b14fa8d7'
HOURLY_WALL_SECONDS=3600
NATIVE_CHECKPOINT_SECONDS=900
NORMAL_LR=.0002
SOURCE_UPDATES=1329
SOURCE_ENV_STEPS=609681408
SOURCE_NATIVE_BYTES=43309324
SOURCE_NATIVE_SHA256='e125a9a946bbd50185829cde5c4e577d4198d39b3ecede06da9aa44a77e496f6'
DECISIONS_PER_UPDATE=458752
TRAINING_PROFILE='normal_lr_formula_longrun_v3'
SOURCE_NATIVE_REVISION='strong-teacher-hand-gae-v3+opponent-diversity-v1+advantage-gated-anchor-v1+opponent-training-expansion-v2'
NATIVE_REVISION=SOURCE_NATIVE_REVISION+'+formula-opponent-v3'
V2_CONTRACT_SHA256='21098503226f081a8e3a823fe1ae05ea924b43eb8f30f1713da63f2febb4b53f'
SOURCE_AUDITOR_FILENAME='normal_lr_source_audit.py'
SOURCE_AUDITOR_SHA256='1ad1c15742b3aa68a14c599148bd15ca6dc2d6e3832eca5507d87ead9566452c'
SOURCE_APPROVAL_RECEIPT_SHA256='6d3ca601b7c6c862519351840c1ff451cf6ff1cc9a4283d538ff77eae3bcb5dc'
FORMULA_SOURCE_SHA256='4a90b163832781f80420fee8c7b74e21185350638b0c5c059a28f2dd258f0df6'
FORMULA_MANIFEST_SHA256='569af9f881ffcfd9cddfd28f148275a95e207da7f786bc95f6c2a81d4ff05553'
TRAIN_STYLE_IDS=tuple(range(13))+tuple(13+64*f+v for f in (0,1,2,5,6,7,8) for v in range(32))
HELDOUT_STYLE_IDS=tuple(13+64*f+v for f in (3,4) for v in range(32,64))

def require(condition,message):
    if not condition:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def integer(value,minimum=0):return type(value) is int and value>=minimum
def number(value):return type(value) in (int,float) and math.isfinite(value)
def utc(value):
    require(isinstance(value,str),'Explicit UTC timestamp required.')
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds()==0,'UTC timezone required.')
    return parsed

def read_json_bytes(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'Duplicate JSON key.');value[key]=item
        return value
    def bad(value):raise ValueError('Nonfinite JSON value.')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    canonical(value)
    return value

def read_json(path):return read_json_bytes(Path(path).read_bytes())
def file_identity(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'Regular nonsymlink file required.')
    raw=p.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def content_identity(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def _identity(value):
    require(isinstance(value,dict) and set(value)=={'bytes','sha256'} and integer(value['bytes'],1) and sha(value['sha256']),'Exact positive byte identity required.')
    return value
def _safe_id(value):return isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',value) is not None
def _copy(value):return read_json_bytes(canonical(value))
def _load_pinned(name,filename,expected):
    require(sha(expected),'Reviewed helper pin unavailable; production admission is blocked.')
    path=ROOT/filename;require(path.is_file() and not path.is_symlink(),'Pinned regular helper required.')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Pinned helper changed:'+filename)
    module=types.ModuleType(name);module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module
def source_v2_contract():return _load_pinned('spades_normal_lr_source_v2_contract','source_v2_contract.py',V2_CONTRACT_SHA256)
def source_auditor():return _load_pinned('spades_normal_lr_source_auditor',SOURCE_AUDITOR_FILENAME,SOURCE_AUDITOR_SHA256)
def legacy_contract():return _load_pinned('spades_normal_lr_legacy_contract',LEGACY_CONTRACT_FILENAME,LEGACY_CONTRACT_SHA256)
def validate_source_artifacts(native_path,source_approval_receipt_path):
    report=source_auditor().validate_source_artifacts(native_path,source_approval_receipt_path)
    support=Path(report['attached_support_root']);status_path=support/'status.json';require(status_path.is_file() and not status_path.is_symlink(),'Prior status must remain a regular file.');status_raw=status_path.read_bytes()
    require(content_identity(status_raw)==report['support_file_identities']['status'],'Admitted prior status changed before external-verification flag check.')
    status=read_json_bytes(status_raw)
    require(status.get('external_verification_required_before_new_run') is True,'Prior result requires explicit external verification before a new run; the actual saved status flag must be present and true.')
    result_path=support/'lr_ab_result.json';result_identity=None
    if result_path.exists():
        require(result_path.is_file() and not result_path.is_symlink(),'Prior result must be a regular file.')
        result_raw=result_path.read_bytes();require(read_json_bytes(result_raw).get('external_verification_required_before_new_run') is True,'Prior result external-verification requirement differs from saved status.');result_identity=content_identity(result_raw)
    report.update(prior_external_verification_required_before_new_run=True,prior_external_verification_acknowledged=False,prior_status_flag_identity=content_identity(status_raw),prior_result_flag_identity=result_identity,source_content_pass_is_external_ack=False)
    return report
AMENDED_FEASIBILITY_FORMAT='spades-kaggle-current-segment-feasibility-v3'
AMENDED_FEASIBILITY_ADDED_KEYS={'campaign_active_seconds','maximum_segments','maximum_segment_active_seconds','segment_active_seconds','segment_index','schedule_amendment_sha256','free_compute_only'}
def validate_feasibility_receipt(path,*,expected_external_receipt_sha256=None,now_utc=None,
        expected_schedule_amendment_sha256=None,expected_segment_index=None,expected_segment_active_seconds=None):
    """Validate historical facts unchanged or one explicitly amended current claim.

    The caller supplies the fresh file digest AND the schedule/index/allowance
    from its validated journal. Neither trust root is read from the receipt.
    """
    require(Path(path).is_file() and not Path(path).is_symlink(),'Regular nonsymlink feasibility receipt required.');raw=Path(path).read_bytes()
    receipt=read_json_bytes(raw)
    if receipt.get('format')!=AMENDED_FEASIBILITY_FORMAT:
        require(expected_schedule_amendment_sha256 is None and expected_segment_index is None and expected_segment_active_seconds is None,'Amended execution requires a current-segment feasibility receipt, not the old two-segment plan.')
        return source_auditor().validate_feasibility_receipt(path,expected_external_receipt_sha256=expected_external_receipt_sha256,now_utc=now_utc)
    require(sha(expected_external_receipt_sha256) and content_identity(raw)['sha256']==expected_external_receipt_sha256,'Exact externally approved current feasibility receipt digest required.')
    old=source_auditor();keys=(old.FEASIBILITY_KEYS-{'authorized_active_segments_seconds'})|AMENDED_FEASIBILITY_ADDED_KEYS
    require(set(receipt)==keys and receipt.get('status')=='parent_reported_current_ui_facts' and receipt.get('provider')=='kaggle','Exact amended current-segment feasibility schema required.')
    require(sha(expected_schedule_amendment_sha256) and receipt['schedule_amendment_sha256']==expected_schedule_amendment_sha256,'Current feasibility must bind the independently validated scheduling amendment.')
    require(type(expected_segment_index) is int and expected_segment_index in (2,3) and receipt['segment_index']==expected_segment_index and type(receipt['segment_index']) is int,'Exact current amended segment index required.')
    active=receipt['segment_active_seconds']
    require(number(expected_segment_active_seconds) and 0<expected_segment_active_seconds<=SEGMENT_ACTIVE_SECONDS and number(active) and type(active) is not bool and active==expected_segment_active_seconds,'Exact current claim allowance, never a new or enlarged allowance, required.')
    require(type(receipt['campaign_active_seconds']) is int and receipt['campaign_active_seconds']==ACTIVE_TRAINING_SECONDS and type(receipt['maximum_segments']) is int and receipt['maximum_segments']==MAX_SEGMENTS and type(receipt['maximum_segment_active_seconds']) is int and receipt['maximum_segment_active_seconds']==SEGMENT_ACTIVE_SECONDS and receipt['free_compute_only'] is True,'Fixed total/per-segment bounds and free-only compute required.')
    now=utc(now_utc) if now_utc is not None else datetime.now(timezone.utc);recorded,expires=utc(receipt['recorded_utc']),utc(receipt['valid_until_utc'])
    require(recorded<=now<expires and 0<(expires-recorded).total_seconds()<=3600,'Current account facts require a live at-most-one-hour validity window.')
    require(receipt['facts_source']=='trusted_parent_runtime_worker_current_kaggle_ui_report' and all(isinstance(receipt[k],str) and bool(receipt[k].strip()) for k in ('parent_report_reference','account_scope_reference')),'Actual trusted-parent account/UI fact provenance required.')
    require(receipt['source_notebook_url']==old.SOURCE_NOTEBOOK_URL and type(receipt['source_notebook_output_version']) is int and receipt['source_notebook_output_version']==360228607,'Original source output identity is immutable during continuation.')
    fields=('maximum_session_wall_seconds','reported_quota_total_seconds','reported_quota_used_seconds','available_gpu_quota_seconds','planned_setup_compile_reserve_seconds','planned_export_reserve_seconds','planned_other_nontraining_reserve_seconds')
    require(all(type(receipt.get(k)) is int and receipt[k]>=0 for k in fields),'Exact nonnegative provider/quota/reserve seconds required.')
    require(0<receipt['maximum_session_wall_seconds']<=43200 and 0<=receipt['reported_quota_used_seconds']<receipt['reported_quota_total_seconds'] and receipt['available_gpu_quota_seconds']==receipt['reported_quota_total_seconds']-receipt['reported_quota_used_seconds'],'Positive bounded session/quota and exact reported arithmetic required.')
    require(receipt['planned_setup_compile_reserve_seconds']>0 and receipt['planned_export_reserve_seconds']>=900,'Positive preparation and at least900-second export reserves required.')
    require(receipt['durability_policy']=='hourly_local_checkpoints_and_end_segment_external_verification' and receipt['hourly_local_checkpoints_required'] is True and receipt['external_saved_checkpoint_ack_required_before_next_segment'] is True and receipt['automatic_continuation'] is False and receipt['during_run_recovery_route_verified'] is False,'Hourly local/end-segment external durability gates must remain unchanged.')
    require(receipt['official_documentation_role']=='Provider background only; actual account facts are parent-reported, not inferred from documentation.','Official documentation cannot substitute for current account facts.')
    setup=receipt['planned_setup_compile_reserve_seconds']+receipt['planned_other_nontraining_reserve_seconds'];export=receipt['planned_export_reserve_seconds'];wall=math.ceil(active)+setup+export
    require(wall<receipt['maximum_session_wall_seconds'] and wall<=receipt['available_gpu_quota_seconds'],'Current claim plus actual preparation/export reserves exceeds free quota or session cap.')
    ledger=dict(receipt_sha256=expected_external_receipt_sha256,account_scope_sha256=digest(dict(provider='kaggle',reference=receipt['account_scope_reference'])),provider='kaggle',verified_utc=receipt['recorded_utc'],valid_until_utc=receipt['valid_until_utc'],maximum_session_wall_seconds=receipt['maximum_session_wall_seconds'],available_gpu_quota_seconds=receipt['available_gpu_quota_seconds'],setup_reserve_seconds=setup,export_reserve_seconds=export,maximum_segment_wall_seconds=wall,during_run_recovery_route_verified=False,during_run_recovery_evidence=[],parent_authorized_segment_end_retrieval=True,hourly_external_durability_verified=False,durability_policy=receipt['durability_policy'])
    validate_ledger_feasibility(ledger,minimum_active_seconds=active)
    return dict(status='PASS',receipt_sha256=expected_external_receipt_sha256,ledger_feasibility=ledger,schedule_amendment_sha256=expected_schedule_amendment_sha256,segment_index=expected_segment_index,maximum_this_segment_active_training_seconds=active,campaign_active_seconds=ACTIVE_TRAINING_SECONDS,maximum_segments=MAX_SEGMENTS,free_compute_only=True,automatic_continuation=False,future_session_availability_verified=False,training_authorized=False,active_training_complete_claim=False,evidence_scope='Current one-claim free-account content evidence only; parent verifies persistence/freshness and consumes one launch.')

EXTERNAL_CLAIM_RECEIPT_KEYS={'format','authority_scope','status','campaign_id','run_id','claim_id','segment_id','segment_index','initial_charge_seconds','claim_ceiling_seconds','prior_journal_head_sha256','original_source_native','segment_source_native','source_notebook_output_version','prior_external_verification_required_before_new_run','prior_external_verification_acknowledged','source_cpu_audit_status','source_cpu_audit_report_reference','recorded_utc','activation_deadline_utc','keeper_reference','one_launch_only','automatic_resume'}
def validate_external_claim_receipt(receipt,*,source_artifacts,authorization,claim,prior_journal_head_sha256):
    require(isinstance(receipt,dict) and set(receipt)==EXTERNAL_CLAIM_RECEIPT_KEYS and receipt['format']=='spades-normal-lr-parent-external-claim-receipt-v3' and receipt['authority_scope']=='trusted_parent_manual_serialized_keeper' and receipt['status']=='prior_source_externally_verified_and_single_segment_reserved','Explicit parent manual external-verification and single-claim receipt required; content PASS alone is insufficient.')
    require(source_artifacts.get('prior_external_verification_required_before_new_run') is True and receipt['prior_external_verification_required_before_new_run'] is True and receipt['prior_external_verification_acknowledged'] is True and receipt['source_cpu_audit_status']=='PASS','The prior saved verification requirement must be acknowledged explicitly by the parent keeper.')
    original=source_artifacts.get('source',source_artifacts);original_identity=content_identity(original['native_bytes'])
    require(receipt['original_source_native']==original_identity==dict(bytes=SOURCE_NATIVE_BYTES,sha256=SOURCE_NATIVE_SHA256) and receipt['segment_source_native']==claim['source']['native'] and type(receipt['source_notebook_output_version']) is int and receipt['source_notebook_output_version']==360228607,'External acknowledgment must name the exact original CONTROL1329, selected segment source and private output version.')
    require(all(receipt[key]==authorization[key] for key in ('campaign_id','run_id')) and all(receipt[key]==claim[key] for key in ('claim_id','segment_id','segment_index','initial_charge_seconds','claim_ceiling_seconds','activation_deadline_utc')) and sha(prior_journal_head_sha256) and receipt['prior_journal_head_sha256']==prior_journal_head_sha256,'External keeper receipt must bind the exact campaign, prior head and reserved segment allowance.')
    require(utc(claim['claimed_utc'])<=utc(receipt['recorded_utc'])<utc(receipt['activation_deadline_utc']) and receipt['one_launch_only'] is True and receipt['automatic_resume'] is False,'Bounded single-launch keeper acknowledgment required.')
    require(all(isinstance(receipt[key],str) and bool(receipt[key].strip()) for key in ('source_cpu_audit_report_reference','keeper_reference')),'Actual source CPU audit proof reference and parent keeper reference required.')
    return dict(status='PASS_CONTENT_ONLY',prior_external_verification_acknowledged=True,source_cpu_audit_report_reference=receipt['source_cpu_audit_report_reference'],source_notebook_output_version=receipt['source_notebook_output_version'],external_current_head_independently_attested=False,launch_authorized_by_content=False,authority_scope='Parent manually verifies freshness, persistence and single launch; no secret/signature service or artifact CAS.')

def formula_opponent_record():
    require(sha(FORMULA_MANIFEST_SHA256),'Reviewed formula manifest pin unavailable.')
    return dict(protocol='spades-formula-opponent-v3',profile=TRAINING_PROFILE,manifest_sha256=FORMULA_MANIFEST_SHA256,training_style_ids=list(TRAIN_STYLE_IDS),heldout_style_ids=list(HELDOUT_STYLE_IDS),native_league_style_leaf=156,old_families_unchanged=True,holdout_used_for_training=False,strength_evaluated=False)

expansion_opponent_record=formula_opponent_record

def training_profile_metadata(cfg):
    require(isinstance(cfg,dict) and all(k in cfg and number(cfg[k]) and 0<cfg[k]<1 for k in ('selfplay_prob','snapshot_prob')),'Actual saved positive self-play and snapshot probabilities required; no defaults.')
    selfplay,snapshot=cfg['selfplay_prob'],cfg['snapshot_prob'];scripted=1.-selfplay-snapshot
    require(scripted>0,'Actual saved scripted probability must also be positive.')
    return dict(format='spades-normal-lr-formula-training-profile-v3',profile=TRAINING_PROFILE,opponent_record=formula_opponent_record(),scripted_sampling=dict(legacy_probability=.1875,old_train_probability=.1875,expansion_v2_probability=.375,formula_probability=.25,legacy_uniform_count=13,old_uniform_family_count=3,expansion_v2_uniform_family_count=3,formula_uniform_family_count=1,uniform_variants_per_family=32,assignment_scope='match'),saved_outer_configuration=dict(selfplay_prob=selfplay,snapshot_prob=snapshot),effective_assignment_probabilities=dict(selfplay=selfplay,historical_snapshot=snapshot,scripted=scripted,legacy_scripted=scripted*3/16,old_family_scripted=scripted*3/16,expansion_v2_scripted=scripted*3/8,formula_scripted=scripted/4,each_legacy_style=scripted*3/208,each_old_family=scripted/16,each_old_family_variant=scripted/512,each_expansion_v2_family=scripted/8,each_expansion_v2_variant=scripted/256,each_formula_variant=scripted/128),historical_snapshot_unavailable_fallback='selfplay',assignment_scope='match; not decision or time fractions',frozen_benchmark_unchanged=True,playing_strength_claim=False)

def build_training_config(source_cfg):
    require(isinstance(source_cfg,dict) and type(source_cfg.get('lr')) is float and source_cfg['lr']==NORMAL_LR,'Verified control source normal LR exactly0.0002 required.')
    require(source_cfg.get('opponent_training_profile')=='training_expansion_v2','Verified control1329 V2 training profile required.')
    require(all(source_cfg.get(k)==v for k,v in dict(seed=42,n_env=3584,rollout_len=128,width=256,minibatch_size=57344,ppo_epochs=3,teacher_anchor_mode='advantage_gated_v1',anchor_conflict_floor=.25,anchor_advantage_scale=1.).items()),'Exact source dimensions and adaptive settings required.')
    training_profile_metadata(source_cfg)
    cfg=_copy(source_cfg);cfg['opponent_training_profile']=TRAINING_PROFILE;return cfg

def migration_record(source_cfg,cfg):
    require(canonical(cfg)==canonical(build_training_config(source_cfg)),'Only explicitly authorized formula profile migration allowed.')
    return dict(kind='normal_lr_formula_configuration',format='spades-normal-lr-formula-configuration-v3',source_cfg_sha256=digest(source_cfg),cfg_sha256=digest(cfg),changes={'opponent_training_profile':dict(before='training_expansion_v2',after=TRAINING_PROFILE)},normal_lr=NORMAL_LR,lr_changed=False,source_revision=SOURCE_NATIVE_REVISION,training_revision=NATIVE_REVISION,training_profile_sha256=digest(training_profile_metadata(source_cfg)),all_other_configuration_unchanged=True,optimizer_state_retained=True,rng_state_retained=True,parameter_reset=False)

arm_migration_record=migration_record

AUTH_KEYS={'format','authorized','campaign_id','run_id','created_utc','activation_deadline_utc','authorization_reference','active_training_seconds','segment_active_seconds','maximum_segments','learning_rate','opponent_training_profile','training_profile_sha256','source_approval_receipt_sha256','source_pins','configuration_sha256','hourly_wall_checkpoint_seconds','native_checkpoint_seconds','automatic_extension','automatic_paid_compute'}
def validate_campaign_authorization(auth,*,expected_source_pins=None):
    # The immutable genesis was issued by the exact historical contract. Its
    # own code pin must never be rewritten to admit this amended implementation.
    return legacy_contract().validate_campaign_authorization(auth,expected_source_pins=expected_source_pins)

FEASIBILITY_KEYS={'receipt_sha256','account_scope_sha256','provider','verified_utc','valid_until_utc','maximum_session_wall_seconds','available_gpu_quota_seconds','setup_reserve_seconds','export_reserve_seconds','maximum_segment_wall_seconds','during_run_recovery_route_verified','during_run_recovery_evidence','parent_authorized_segment_end_retrieval','hourly_external_durability_verified','durability_policy'}
def validate_ledger_feasibility(value,*,minimum_active_seconds=SEGMENT_ACTIVE_SECONDS):
    require(isinstance(value,dict) and set(value)==FEASIBILITY_KEYS and value['provider']=='kaggle' and sha(value['receipt_sha256']) and sha(value['account_scope_sha256']),'Exact externally reviewed feasibility summary required.')
    require(utc(value['verified_utc'])<utc(value['valid_until_utc']),'Current feasibility verification interval required.')
    for key in ('maximum_session_wall_seconds','available_gpu_quota_seconds','setup_reserve_seconds','export_reserve_seconds','maximum_segment_wall_seconds'):
        require(number(value[key]) and value[key]>=0,'Finite nonnegative feasibility limit required:'+key)
    require(value['maximum_session_wall_seconds']>0 and value['available_gpu_quota_seconds']>0 and value['export_reserve_seconds']>=900 and value['maximum_segment_wall_seconds']<=min(value['maximum_session_wall_seconds'],value['available_gpu_quota_seconds']),'Verified quota/session bounds and at least900s export reserve required.')
    require(value['maximum_segment_wall_seconds']>value['setup_reserve_seconds']+value['export_reserve_seconds'],'Verified wall allocation cannot contain setup/export reserves and active work.')
    route=value['during_run_recovery_route_verified'];require(type(route) is bool and isinstance(value['during_run_recovery_evidence'],list),'Explicit hourly external durability route evidence required.')
    for identity in value['during_run_recovery_evidence']:_identity(identity)
    require(not route or bool(value['during_run_recovery_evidence']),'Verified during-run retrieval route requires actual evidence.')
    require(value['parent_authorized_segment_end_retrieval'] is True and value['hourly_external_durability_verified'] is route and value['durability_policy']=='hourly_local_checkpoints_and_end_segment_external_verification','Hourly local snapshots must not be relabeled as external VM-loss durability.')
    require(number(minimum_active_seconds) and 0<minimum_active_seconds<=SEGMENT_ACTIVE_SECONDS and value['maximum_segment_wall_seconds']>=minimum_active_seconds+value['setup_reserve_seconds']+value['export_reserve_seconds'],'Verified wall/quota allocation must include the exact bounded current claim plus preparation/export overhead.')
    return _copy(value)

SOURCE_SUMMARY_KEYS={'receipt_sha256','native','updates','env_steps','source_cfg_sha256'}
def _source_summary(value):
    require(isinstance(value,dict) and set(value)==SOURCE_SUMMARY_KEYS and sha(value['receipt_sha256']) and sha(value['source_cfg_sha256']),'Exact externally verified source summary required.')
    _identity(value['native'])
    require(integer(value['updates'],SOURCE_UPDATES) and value['env_steps']==SOURCE_ENV_STEPS+(value['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE,'Verified source lifetime counters differ.')
    return _copy(value)

def _append(journal,kind,payload):
    out=_copy(journal);entry=dict(index=len(out['events']),previous_sha256=out['head_sha256'],kind=kind,payload=_copy(payload));entry['sha256']=digest(entry);out['events'].append(entry);out['head_sha256']=entry['sha256'];return out

def create_campaign_journal(auth,source_summary,feasibility_summary):
    validate_campaign_authorization(auth);source=_source_summary(source_summary);feasibility=validate_ledger_feasibility(feasibility_summary)
    require(source['updates']==SOURCE_UPDATES and source['env_steps']==SOURCE_ENV_STEPS and source['receipt_sha256']==auth['source_approval_receipt_sha256'] and source['native']==dict(bytes=SOURCE_NATIVE_BYTES,sha256=SOURCE_NATIVE_SHA256),'Genesis must start at explicitly approved exact control1329, never fallback1201.')
    out=dict(format='spades-normal-lr-external-campaign-journal-v3',campaign_id=auth['campaign_id'],events=[],head_sha256='0'*64)
    return _append(out,'campaign_authorized',dict(authorization=auth,source=source,feasibility=feasibility))

CLAIM_KEYS={'claim_id','segment_id','segment_index','claimed_utc','activation_deadline_utc','platform_hard_end_utc','launch_deadline_utc','initial_charge_seconds','claim_ceiling_seconds','source','feasibility','external_claim_receipt_sha256'}
CHECKPOINT_KEYS={'claim_id','verified_utc','source','claim_measured_kernel_active_seconds','claim_observed_attempt_seconds','claim_charged_active_seconds','claim_blocked','unresolved_launch','kernel_event_head_sha256','segment_terminal','segment_active_target_overshoot','active_budget_exception','external_verification_receipt_sha256'}

SCHEDULING_AMENDMENT_FORMAT='spades-normal-lr-scheduling-amendment-v3'
SCHEDULING_AMENDMENT_KEYS={'format','authorized','amendment_id','campaign_id','run_id','recorded_utc','authorization_reference','prior_journal_head_sha256','original_authorization_sha256','prior_charged_active_seconds','prior_successful_kernel_seconds','prior_source','maximum_segments','segment_active_seconds','active_training_seconds','execution_source_pins','free_compute_only','automatic_resume','automatic_extension','automatic_paid_compute'}

def _validate_execution_pins(pins,auth):
    original=auth['source_pins']
    require(isinstance(pins,dict) and set(pins)==set(original) and all(sha(v) for v in pins.values()),'Exact reviewed execution source pin map required.')
    require(all(pins[k]==v for k,v in original.items() if k not in ('contract_sha256','operator_sha256')),'Scheduling amendment cannot change model/opponent/source/optimizer helpers.')
    require(pins['contract_sha256']==file_identity(__file__)['sha256'] and pins['operator_sha256']==file_identity(ROOT/'spades_normal_longrun.py')['sha256'],'Amended contract/runtime pins must name the actual reviewed execution files.')
    return _copy(pins)

def validate_scheduling_amendment(amendment,*,journal_state,prior_journal_head_sha256):
    """Validate one explicit parent scheduling adjustment, never a new budget.

    Terminal-native health and externally retained bytes are checked by the
    normal bundle/continuation admission; this metadata does not invent them.
    """
    s=journal_state;auth=s['authorization'];a=amendment
    require(isinstance(a,dict) and set(a)==SCHEDULING_AMENDMENT_KEYS and a['format']==SCHEDULING_AMENDMENT_FORMAT and a['authorized'] is True,'Exact explicitly authorized scheduling amendment required.')
    require(s.get('scheduling_amendment') is None and s['open_claim'] is None and s['segment_count'] in (1,2) and s['previous_segment_externally_verified'] is True and s.get('previous_segment_resolved_healthy') is True and not s['active_budget_exception'] and s['charged_seconds']<ACTIVE_TRAINING_SECONDS,'Amendment requires a healthy externally resolved prior segment, remaining budget and no outstanding claim or prior amendment.')
    require(_safe_id(a['amendment_id']) and a['campaign_id']==auth['campaign_id'] and a['run_id']==auth['run_id'] and sha(prior_journal_head_sha256) and a['prior_journal_head_sha256']==prior_journal_head_sha256 and a['original_authorization_sha256']==digest(auth),'Amendment must bind the unchanged original authorization/campaign and current pre-amendment head.')
    require(number(a['prior_charged_active_seconds']) and a['prior_charged_active_seconds']==s['charged_seconds'] and number(a['prior_successful_kernel_seconds']) and a['prior_successful_kernel_seconds']==s['measured_kernel_active_seconds'] and canonical(a['prior_source'])==canonical(s['source']),'Amendment cannot reset charged cost, invent training credit or roll back the verified native.')
    require(type(a['maximum_segments']) is int and a['maximum_segments']==MAX_SEGMENTS and type(a['segment_active_seconds']) is int and a['segment_active_seconds']==SEGMENT_ACTIVE_SECONDS and type(a['active_training_seconds']) is int and a['active_training_seconds']==ACTIVE_TRAINING_SECONDS,'Only a third sequential segment within the same21600/43200 bounds is permitted.')
    require(a['free_compute_only'] is True and a['automatic_resume'] is False and a['automatic_extension'] is False and a['automatic_paid_compute'] is False,'Free-only, parent-controlled, nonautomatic continuation is required.')
    require(isinstance(a['authorization_reference'],str) and bool(a['authorization_reference'].strip()) and utc(s['last_event_utc'])<=utc(a['recorded_utc'])<utc(auth['activation_deadline_utc']),'Actual parent approval reference and unchanged authorization activation window required.')
    _validate_execution_pins(a['execution_source_pins'],auth)
    return _copy(a)

def amend_campaign_schedule(journal,amendment,*,expected_head_sha256):
    state=validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256)
    validate_scheduling_amendment(amendment,journal_state=state,prior_journal_head_sha256=expected_head_sha256)
    out=_append(journal,'scheduling_amended',amendment);validate_campaign_journal(out);return out

def validate_execution_authorization(authorization,journal,*,expected_source_pins=None,expected_head_sha256=None):
    state=validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256)
    require(canonical(state['authorization'])==canonical(validate_campaign_authorization(authorization)),'Execution configuration must preserve the original genesis authorization unchanged.')
    require(state['scheduling_amendment'] is not None,'This amended implementation requires an explicit parent scheduling amendment before execution.')
    pins=_validate_execution_pins(state['execution_source_pins'],authorization)
    if expected_source_pins is not None:require(canonical(pins)==canonical(expected_source_pins),'Actual execution source closure differs from the parent-approved scheduling amendment.')
    return state

def validate_campaign_journal(journal,*,expected_head_sha256=None):
    require(isinstance(journal,dict) and set(journal)=={'format','campaign_id','events','head_sha256'} and journal['format']=='spades-normal-lr-external-campaign-journal-v3' and _safe_id(journal['campaign_id']),'Exact external campaign journal required.')
    events=journal['events'];require(isinstance(events,list) and bool(events),'Nonempty campaign journal required.')
    amendments=[i for i,event in enumerate(events) if isinstance(event,dict) and event.get('kind')=='scheduling_amended']
    require(len(amendments)<=1,'Scheduling amendment cannot be duplicated or replayed.')
    # Validate every historical event with the exact original contract before
    # interpreting the new event. Existing event bytes and their hashes remain
    # unchanged; the original authorization continues to carry its original pins.
    if not amendments:
        state=legacy_contract().validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256)
        resolutions=[e['kind'] for e in events if e.get('kind') in ('segment_resolved_healthy','segment_abandoned_unresolved')]
        state.update(execution_source_pins=_copy(state['authorization']['source_pins']),maximum_segments=LEGACY_MAX_SEGMENTS,scheduling_amendment=None,previous_segment_resolved_healthy=bool(resolutions and resolutions[-1]=='segment_resolved_healthy'))
        return state
    transition=amendments[0];require(transition>0,'Scheduling amendment cannot replace genesis.')
    prefix=dict(format=journal['format'],campaign_id=journal['campaign_id'],events=_copy(events[:transition]),head_sha256=events[transition]['previous_sha256'])
    legacy_contract().validate_campaign_journal(prefix,expected_head_sha256=prefix['head_sha256'])
    prior='0'*64;state=None;claim_ids=set();segment_ids=set()
    for index,event in enumerate(events):
        require(isinstance(event,dict) and set(event)=={'index','previous_sha256','kind','payload','sha256'} and type(event['index']) is int and event['index']==index and event['previous_sha256']==prior and event['sha256']==digest({k:v for k,v in event.items() if k!='sha256'}),'Campaign journal chain/index/hash differs.')
        kind,payload=event['kind'],event['payload'];require(isinstance(payload,dict),'Journal event payload required.')
        if index==0:
            require(kind=='campaign_authorized' and set(payload)=={'authorization','source','feasibility'},'Exact campaign genesis required.')
            auth=validate_campaign_authorization(payload['authorization']);source=_source_summary(payload['source']);feasibility=validate_ledger_feasibility(payload['feasibility'])
            require(auth['campaign_id']==journal['campaign_id'] and source['updates']==SOURCE_UPDATES and source['env_steps']==SOURCE_ENV_STEPS and source['receipt_sha256']==auth['source_approval_receipt_sha256'] and source['native']==dict(bytes=SOURCE_NATIVE_BYTES,sha256=SOURCE_NATIVE_SHA256),'Approved exact control1329 genesis binding differs.')
            state=dict(authorization=auth,source=source,account_scope_sha256=feasibility['account_scope_sha256'],charged_seconds=0.,measured_kernel_active_seconds=0.,observed_attempt_seconds=0.,failed_attempt_seconds=0.,conservative_unmeasured_seconds=0.,segment_count=0,open_claim=None,last_event_utc=auth['created_utc'],previous_segment_externally_verified=False,previous_segment_resolved_healthy=False,active_budget_exception=False,execution_source_pins=_copy(auth['source_pins']),maximum_segments=LEGACY_MAX_SEGMENTS,scheduling_amendment=None)
        elif kind=='scheduling_amended':
            amendment=validate_scheduling_amendment(payload,journal_state=state,prior_journal_head_sha256=prior)
            state.update(scheduling_amendment=amendment,execution_source_pins=_copy(amendment['execution_source_pins']),maximum_segments=MAX_SEGMENTS,last_event_utc=amendment['recorded_utc'])
        elif kind=='segment_claimed':
            require(set(payload)==CLAIM_KEYS and state['open_claim'] is None,'One exact outstanding segment claim permitted.')
            require(_safe_id(payload['claim_id']) and payload['claim_id'] not in claim_ids and _safe_id(payload['segment_id']) and payload['segment_id'] not in segment_ids and type(payload['segment_index']) is int and payload['segment_index']==state['segment_count']+1<=state['maximum_segments'],'Only new unique sequential segment identities within the issued schedule are permitted.')
            require(not state['active_budget_exception'] and (state['segment_count']==0 or state['previous_segment_externally_verified']),'Second segment requires first segment independently persisted and verified; a budget exception forbids further launch.')
            if state['scheduling_amendment'] is not None:require(state['previous_segment_resolved_healthy'] is True,'Amended continuation requires a healthy externally resolved prior segment.')
            f=validate_ledger_feasibility(payload['feasibility'],minimum_active_seconds=payload['claim_ceiling_seconds'] if state['scheduling_amendment'] is not None else SEGMENT_ACTIVE_SECONDS);source=_source_summary(payload['source']);started=utc(payload['claimed_utc']);activation=utc(payload['activation_deadline_utc']);end=utc(payload['platform_hard_end_utc']);deadline=utc(payload['launch_deadline_utc'])
            require(source==state['source'] and f['account_scope_sha256']==state['account_scope_sha256'] and sha(payload['external_claim_receipt_sha256']),'Claim must bind exact latest externally verified source/account/claim receipt.')
            require(utc(state['last_event_utc'])<=started and utc(state['authorization']['created_utc'])<=started<utc(state['authorization']['activation_deadline_utc']) and utc(f['verified_utc'])<=started<activation<=utc(f['valid_until_utc']) and activation<=deadline<end,'Claim activation/current-feasibility/clock interval differs.')
            require((end-started).total_seconds()<=f['maximum_segment_wall_seconds'] and (end-deadline).total_seconds()==f['export_reserve_seconds'],'Claim wall interval or final export reserve exceeds current verified allowance.')
            ceiling=payload['claim_ceiling_seconds'];require(number(ceiling) and number(payload['initial_charge_seconds']) and ceiling==min(SEGMENT_ACTIVE_SECONDS,ACTIVE_TRAINING_SECONDS-state['charged_seconds']) and ceiling>0 and payload['initial_charge_seconds']==state['charged_seconds'] and ceiling<=(deadline-started).total_seconds()-f['setup_reserve_seconds'],'Claim must reserve its exact bounded six-hour segment allowance before launch.')
            claim_ids.add(payload['claim_id']);segment_ids.add(payload['segment_id']);state['segment_count']+=1
            state['open_claim']=dict(payload=payload,verified_measured_seconds=0.,verified_observed_seconds=0.,verified_charged_seconds=0.,latest_verified_source=source,kernel_event_head_sha256=None,terminal_verified=False,claim_blocked=False,unresolved_launch=False)
            state['last_event_utc']=payload['claimed_utc']
        elif kind=='checkpoint_externally_verified':
            require(set(payload)==CHECKPOINT_KEYS and state['open_claim'] is not None,'Checkpoint requires active external segment claim.')
            claim=state['open_claim'];p=claim['payload'];source=_source_summary(payload['source']);measured=payload['claim_measured_kernel_active_seconds'];observed=payload['claim_observed_attempt_seconds'];charged=payload['claim_charged_active_seconds']
            require(payload['claim_id']==p['claim_id'] and all(number(x) for x in (measured,observed,charged)) and claim['verified_measured_seconds']<=measured<=observed<=charged and claim['verified_observed_seconds']<=observed and claim['verified_charged_seconds']<=charged and sha(payload['kernel_event_head_sha256']) and sha(payload['external_verification_receipt_sha256']) and type(payload['segment_terminal']) is bool and type(payload['segment_active_target_overshoot']) is bool and type(payload['active_budget_exception']) is bool,'Verified training credit, observed attempts and full charged cost must remain monotonic and separate.')
            require(type(payload['claim_blocked']) is bool and type(payload['unresolved_launch']) is bool and (not (payload['claim_blocked'] or payload['unresolved_launch']) or payload['segment_terminal']),'Blocked or unresolved kernel state requires terminal recovery evidence.')
            segment_overshoot=charged>p['claim_ceiling_seconds'];overrun=state['charged_seconds']+charged>ACTIVE_TRAINING_SECONDS
            require(payload['segment_active_target_overshoot']==segment_overshoot and payload['active_budget_exception']==overrun and (not (segment_overshoot or overrun) or payload['segment_terminal']),'Full in-flight overshoot must be terminal and reported truthfully, never clipped.')
            require(source['updates']>=claim['latest_verified_source']['updates'] and source['source_cfg_sha256']==state['authorization']['configuration_sha256'] and utc(state['last_event_utc'])<=utc(payload['verified_utc']),'Verified checkpoint source/configuration/time rolled back.')
            require(not claim['terminal_verified'] and (p['feasibility']['during_run_recovery_route_verified'] or payload['segment_terminal']),'Batch-end-only retrieval cannot acknowledge a running local snapshot; terminal segment evidence required.')
            claim.update(verified_measured_seconds=measured,verified_observed_seconds=observed,verified_charged_seconds=charged,latest_verified_source=source,kernel_event_head_sha256=payload['kernel_event_head_sha256'],terminal_verified=payload['segment_terminal'],claim_blocked=payload['claim_blocked'],unresolved_launch=payload['unresolved_launch']);state['last_event_utc']=payload['verified_utc']
            state['active_budget_exception']=state['active_budget_exception'] or overrun
        elif kind in ('segment_resolved_healthy','segment_abandoned_unresolved'):
            require(set(payload)=={'claim_id','resolved_utc','external_resolution_receipt_sha256','reason'} and state['open_claim'] is not None and sha(payload['external_resolution_receipt_sha256']) and isinstance(payload['reason'],str) and bool(payload['reason'].strip()),'Externally acknowledged exact segment resolution required.')
            claim=state['open_claim'];p=claim['payload'];require(payload['claim_id']==p['claim_id'] and utc(state['last_event_utc'])<=utc(payload['resolved_utc']),'Segment resolution claim/time differs.')
            measured=claim['verified_measured_seconds'];observed=claim['verified_observed_seconds']
            if kind=='segment_resolved_healthy':
                require(claim['terminal_verified'] and not claim['claim_blocked'] and not claim['unresolved_launch'],'Healthy resolution requires externally verified terminal native and fully resolved unblocked kernel ledger.')
                charge=claim['verified_charged_seconds']
            else:charge=max(p['claim_ceiling_seconds'],claim['verified_charged_seconds'])
            state['charged_seconds']+=charge;state['measured_kernel_active_seconds']+=measured;state['observed_attempt_seconds']+=observed;state['failed_attempt_seconds']+=observed-measured;state['conservative_unmeasured_seconds']+=charge-observed;state['source']=claim['latest_verified_source'];state['previous_segment_externally_verified']=claim['kernel_event_head_sha256'] is not None;state['previous_segment_resolved_healthy']=kind=='segment_resolved_healthy';state['open_claim']=None;state['last_event_utc']=payload['resolved_utc']
            require(state['measured_kernel_active_seconds']<=state['charged_seconds'] and (state['charged_seconds']<=ACTIVE_TRAINING_SECONDS or state['active_budget_exception']),'Active overflow must remain fully charged and explicitly exceptional.')
        else:raise ValueError('Unknown/repeated campaign journal event:'+str(kind))
        prior=event['sha256']
    require(journal['head_sha256']==prior,'Campaign journal head differs.')
    if expected_head_sha256 is not None:require(sha(expected_head_sha256) and prior==expected_head_sha256,'Journal differs from independently retrieved current external head; stale replay rejected.')
    reserved=state['open_claim']['payload']['claim_ceiling_seconds'] if state['open_claim'] else 0.
    target_reached=state['measured_kernel_active_seconds']>=ACTIVE_TRAINING_SECONDS and state['open_claim'] is None
    state.update(head_sha256=prior,reserved_seconds=reserved,remaining_unclaimed_seconds=max(0.,ACTIVE_TRAINING_SECONDS-state['charged_seconds']-reserved),actual_active_target_reached=target_reached,actual_12h_training_complete=target_reached,bounded_completion_without_budget_exception=target_reached and not state['active_budget_exception'],budget_exhausted_inconclusive=(state['charged_seconds']>=ACTIVE_TRAINING_SECONDS or state['segment_count']==state['maximum_segments']) and state['measured_kernel_active_seconds']<ACTIVE_TRAINING_SECONDS and state['open_claim'] is None,external_current_head_independently_attested=False)
    return _copy(state)

def claim_segment(journal,claim,*,expected_head_sha256):
    validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256);out=_append(journal,'segment_claimed',claim);validate_campaign_journal(out);return out
def record_verified_checkpoint(journal,checkpoint,*,expected_head_sha256):
    validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256);out=_append(journal,'checkpoint_externally_verified',checkpoint);validate_campaign_journal(out);return out
def resolve_verified_segment(journal,resolution,*,expected_head_sha256):
    validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256);out=_append(journal,'segment_resolved_healthy',resolution);validate_campaign_journal(out);return out
def abandon_unresolved_segment(journal,resolution,*,expected_head_sha256):
    validate_campaign_journal(journal,expected_head_sha256=expected_head_sha256);out=_append(journal,'segment_abandoned_unresolved',resolution);validate_campaign_journal(out);return out

def audit_training_style_array(array,cfg):
    import numpy as np
    require(array.shape==(cfg['n_env'],) and str(array.dtype)=='int32','Formula native league_style leaf156 shape/dtype.')
    require(bool(np.isin(array,TRAIN_STYLE_IDS).all()),'Formula native league_style contains heldout/reserved/negative/unknown IDs.')

def _native_content(raw,*,expected_cfg=None,source_manifest=None):
    """Pure current-native content checks; source admission is a separate pin gate."""
    import numpy as np
    v2=source_v2_contract();old=v2.native_auditor()
    parts=old.zip_members(raw,('manifest.json','history.json','state.npz'))
    m=read_json_bytes(parts['manifest.json']);history=read_json_bytes(parts['history.json'])
    require(isinstance(m,dict) and m.get('format')==1 and type(m.get('format')) is int and m.get('ruleset')=='modified-spades-500-blind-swap-team-failure-v1','Native format/ruleset differs.')
    require(m.get('training_revision')==NATIVE_REVISION and m.get('training_state_schema')==2 and type(m.get('training_state_schema')) is int and m.get('versions')==old.STACK,'Exact formula native revision/schema/stack required.')
    cfg=m.get('cfg');require(isinstance(cfg,dict) and cfg.get('opponent_training_profile')==TRAINING_PROFILE and type(cfg.get('lr')) is float and cfg['lr']==NORMAL_LR,'Exact formula profile and unchanged normal LR required.')
    require(isinstance(history,list) and all(isinstance(row,dict) for row in history),'Native history schema required.')
    require(canonical(m.get('opponent_diversity'))==canonical(formula_opponent_record()) and canonical(m.get('opponent_training_profile'))==canonical(training_profile_metadata(cfg)) and canonical(m.get('source_opponent_training_profile'))==canonical(v2.training_profile_metadata(cfg)) and canonical(m.get('frozen_opponent_benchmark'))==canonical(old.OPPONENT_DIVERSITY),'Truthful formula profile, retained V2 source profile and separate frozen benchmark required.')
    if expected_cfg is not None:require(canonical(cfg)==canonical(expected_cfg),'Exact full native configuration differs.')
    require(integer(m.get('updates'),SOURCE_UPDATES) and m.get('env_steps')==SOURCE_ENV_STEPS+(m['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and number(m.get('cumulative_seconds')) and m['cumulative_seconds']>=0,'Native lifetime counters/cumulative wall time differ.')
    utc(m.get('created_utc'));require(m.get('sha256')==hashlib.sha256(parts['state.npz']).hexdigest(),'Native NPZ hash differs.')
    specs=m.get('leaves');require(isinstance(specs,list) and len(specs)==214,'Full214 native leaf specifications required.')
    if source_manifest is not None:require(canonical(specs)==canonical(source_manifest['leaves']),'Native214 source leaf specifications changed.')
    old.preflight_npz(parts['state.npz'],specs)
    with np.load(io.BytesIO(parts['state.npz']),allow_pickle=False) as arrays:
        require(sorted(arrays.files)==[f'leaf_{i:05d}' for i in range(214)],'Exactly214 unique native arrays required.')
        for i,spec in enumerate(specs):
            array=arrays[f'leaf_{i:05d}'];require(list(array.shape)==spec['shape'] and str(array.dtype)==spec['dtype'] and not array.dtype.hasobject,'Native leaf shape/dtype differs:'+str(i));require(not np.issubdtype(array.dtype,np.inexact) or bool(np.isfinite(array).all()),'Nonfinite native leaf:'+str(i))
        audit_training_style_array(arrays['leaf_00156'],cfg)
        values=[]
        for i,dtype in zip(range(207,214),('int32','int32','int32','bool','uint32','uint32','int32')):
            array=arrays[f'leaf_{i:05d}'];require(array.shape==() and str(array.dtype)==dtype,'Native lifetime scalar schema differs.');values.append(array.item())
        update,decisions,bootstrap,complete,low,high,base=values
        require(update==m['updates'] and decisions==DECISIONS_PER_UPDATE and bootstrap==0 and complete is True and 0<=base<=update and ((int(high)<<32)|int(low))+(int(update)-int(base))*int(decisions)==m['env_steps'],'Actual native lifetime scalar values differ.')
    return m,history

def validate_training_native(raw,*,source_artifacts,expected_config=None):
    source_cfg=source_artifacts['source_cfg'];cfg=build_training_config(source_cfg)
    if expected_config is not None:require(canonical(expected_config)==canonical(cfg),'Caller full training configuration differs.')
    source=source_artifacts.get('source',source_artifacts);source_raw=source['native_bytes'];source_m=source['native_manifest'];source_h=source['native_history']
    require(source_m['updates']==SOURCE_UPDATES and source_m['env_steps']==SOURCE_ENV_STEPS and source_m['training_revision']==SOURCE_NATIVE_REVISION,'Approved initial control1329 source required.')
    require(source_artifacts['source_health'].get('should_stop') is False,'Original source health forbids formula migration.')
    m,h=_native_content(raw,expected_cfg=cfg,source_manifest=source_m)
    require(canonical(h[:len(source_h)])==canonical(source_h),'Exact immutable approved source history prefix changed.')
    require(m['cumulative_seconds']>=source_m['cumulative_seconds'],'Native cumulative wall time rolled back.')
    source_v2_contract().native_auditor().continuation_arrays(raw,source_raw,m['updates']==SOURCE_UPDATES)
    _audit_native_history(h,source_artifacts,m)
    work=source_v2_contract()._work_audit(h,cfg,SOURCE_UPDATES,m['updates'])
    return m,h,work

ACTIVE_SUMMARY_KEYS={'format','campaign_id','segment_id','initial_charge_seconds','claim_ceiling_seconds','charged_active_seconds','measured_kernel_seconds','observed_attempt_seconds','successful_kernel_seconds','attempted_kernel_seconds','failed_attempt_seconds','conservative_unmeasured_seconds','remaining_active_seconds','maximum_active_seconds','ledger_sha256','ledger_rows','claim_blocked','unresolved_launch','local_ledger_authorizes_new_segment'}
def validate_active_summary(value):
    require(isinstance(value,dict) and set(value)==ACTIVE_SUMMARY_KEYS and value['format']=='spades-active-training-summary-v3' and all(_safe_id(value[k]) for k in ('campaign_id','segment_id')),'Exact active-kernel accounting summary required.')
    for key in ('initial_charge_seconds','claim_ceiling_seconds','charged_active_seconds','measured_kernel_seconds','observed_attempt_seconds','successful_kernel_seconds','attempted_kernel_seconds','failed_attempt_seconds','conservative_unmeasured_seconds','remaining_active_seconds'):
        require(number(value[key]) and value[key]>=0,'Finite nonnegative active accounting required:'+key)
    require(value['maximum_active_seconds']==ACTIVE_TRAINING_SECONDS and type(value['maximum_active_seconds']) is int and value['claim_ceiling_seconds']>0 and value['claim_ceiling_seconds']<=SEGMENT_ACTIVE_SECONDS and value['initial_charge_seconds']+value['claim_ceiling_seconds']<=ACTIVE_TRAINING_SECONDS,'Active clock maximum/claim reserve differs.')
    require(value['measured_kernel_seconds']==value['successful_kernel_seconds']<=value['observed_attempt_seconds']==value['attempted_kernel_seconds'] and abs(value['failed_attempt_seconds']-(value['observed_attempt_seconds']-value['measured_kernel_seconds']))<=1e-6,'Only successful synchronized kernels receive target credit; failed attempts remain observed cost.')
    require(abs(value['charged_active_seconds']-value['initial_charge_seconds']-value['observed_attempt_seconds']-value['conservative_unmeasured_seconds'])<=1e-6 and abs(value['remaining_active_seconds']-max(0.,ACTIVE_TRAINING_SECONDS-value['charged_active_seconds']))<=1e-6,'Observed attempt cost and conservative unknown reserve must remain separate and fully charged.')
    require(sha(value['ledger_sha256']) and integer(value['ledger_rows'],1) and type(value['claim_blocked']) is bool and type(value['unresolved_launch']) is bool and value['local_ledger_authorizes_new_segment'] is False,'Kernel-ledger identity and no-local-launch-authority fields required.')
    return _copy(value)

def _audit_native_history(history,source_artifacts,manifest):
    source=source_artifacts.get('source',source_artifacts);base=source['native_history'];tail=history[len(base):]
    cfg=build_training_config(source_artifacts['source_cfg']);expected=dict(kind='normal_lr_formula_branch_start',source_native_sha256=hashlib.sha256(source['native_bytes']).hexdigest(),migration=migration_record(source_artifacts['source_cfg'],cfg),source_health=source_artifacts['source_health'])
    require(len(tail)>=3 and canonical(tail[0])==canonical(expected),'Exact once-only source restore/health/formula-migration history barrier required.')
    allowed={'normal_lr_segment_start','normal_lr_checkpoint','optimizer_work','aot_raw_metrics','optimizer_health_stop'};segments=[];prior_update=SOURCE_UPDATES;prior_charge=0.
    for row in tail[1:]:
        require(row.get('kind') in allowed,'Unexpected normal-LR native history event.')
        if row['kind']=='normal_lr_segment_start':
            require(set(row)=={'kind','claim_id','journal_head_sha256','source_native_identity','configuration_sha256'} and _safe_id(row['claim_id']) and sha(row['journal_head_sha256']) and row['configuration_sha256']==digest(cfg),'Exact externally claimed segment history required.')
            _identity(row['source_native_identity']);require(row['claim_id'] not in segments and len(segments)<MAX_SEGMENTS,'Duplicate/extra segment history rejected.')
            if not segments:require(row['source_native_identity']==content_identity(source['native_bytes']),'First formula segment must restore exact approved control1329.')
            segments.append(row['claim_id'])
        elif row['kind']=='normal_lr_checkpoint':
            require(set(row)=={'kind','claim_id','update','env_steps','kernel_event_head_sha256','active_clock','terminal','reason','nonce'} and segments and row['claim_id']==segments[-1] and integer(row['update'],prior_update) and row['update']<=manifest['updates'] and row['env_steps']==SOURCE_ENV_STEPS+(row['update']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and sha(row['kernel_event_head_sha256']) and type(row['terminal']) is bool and isinstance(row['reason'],str) and bool(row['reason']) and isinstance(row['nonce'],str) and bool(row['nonce']),'Exact saved native checkpoint counters/claim/nonce required.')
            active=validate_active_summary(row['active_clock']);require(active['ledger_sha256']==row['kernel_event_head_sha256'] and active['charged_active_seconds']>=prior_charge,'Checkpoint kernel-ledger identity or charged accounting rolled back.')
            prior_update=row['update'];prior_charge=active['charged_active_seconds']
    require(tail[-1].get('kind')=='normal_lr_checkpoint' and tail[-1]['update']==manifest['updates'] and tail[-1]['env_steps']==manifest['env_steps'],'Native must end at exact accepted-state checkpoint barrier.')
    return tail[-1]

validate_native_content=_native_content

def validate_continuation_artifacts(native_path,*,source_artifacts,journal_state,expected_head_sha256):
    """Resume exact externally verified bytes against the admitted original source."""
    require(isinstance(journal_state,dict) and journal_state.get('head_sha256')==expected_head_sha256 and sha(expected_head_sha256),'Continuation requires independently retrieved current journal head.')
    open_claim=journal_state.get('open_claim')
    maximum=journal_state.get('maximum_segments',LEGACY_MAX_SEGMENTS);count=journal_state.get('segment_count')
    before_claim=open_claim is None and type(count) is int and 1<=count<maximum
    current_claim=(isinstance(open_claim,dict) and type(count) is int and count in (2,3) and count<=maximum and open_claim.get('payload',{}).get('segment_index')==count and canonical(open_claim['payload'].get('source'))==canonical(journal_state.get('source')))
    require((before_claim or current_claim) and journal_state.get('previous_segment_externally_verified') is True and journal_state.get('active_budget_exception') is False,'Only the latest externally verified prior-segment source can continue within the issued schedule.')
    if journal_state.get('scheduling_amendment') is not None:require(journal_state.get('previous_segment_resolved_healthy') is True,'Amended continuation requires a healthy externally resolved prior segment.')
    require(ACTIVE_TRAINING_SECONDS-journal_state.get('charged_seconds',ACTIVE_TRAINING_SECONDS)>0,'No remaining authorized active allowance.')
    raw=Path(native_path).read_bytes();identity=content_identity(raw);expected=_source_summary(journal_state['source'])
    require(identity==expected['native'],'Resume requires exact latest externally verified native ZIP bytes; no rollback/recompression.')
    m,h,work=validate_training_native(raw,source_artifacts=source_artifacts)
    require(m['updates']==expected['updates'] and m['env_steps']==expected['env_steps'] and digest(m['cfg'])==expected['source_cfg_sha256'],'Externally verified resume counters/configuration differ.')
    require(work['health']['should_stop'] is False and work['health_stop_recorded'] is False,'Persisted optimizer health stop forbids continuation.')
    scope='EXACT_EXTERNALLY_VERIFIED_PRIOR_SEGMENT_NATIVE_CONTINUATION' if journal_state.get('scheduling_amendment') is not None else 'EXACT_EXTERNALLY_VERIFIED_SEGMENT1_NATIVE_CONTINUATION'
    return dict(native_bytes=raw,native_manifest=m,native_history=h,native_identity=identity,source_cfg=m['cfg'],source_health=work['health'],source_journal=work,source_history_sha256=digest(h),source_validation_scope=scope,source_identities=dict(source_native=identity),journal_head_sha256=expected_head_sha256,measured_kernel_active_seconds=journal_state['measured_kernel_active_seconds'],charged_seconds=journal_state['charged_seconds'],prior_external_receipt_sha256=expected['receipt_sha256'])

def active_claim_bounds_ns(initial_charge_seconds,claim_ceiling_seconds,max_seconds=ACTIVE_TRAINING_SECONDS):
    """Conservatively retain carry while avoiding a one-ns residual double ceil.

    Only the exact mathematical remaining claim may lose the representation's
    single extra ns. Measured call durations and accumulated costs are untouched.
    """
    require(all(number(v) and v>=0 for v in (initial_charge_seconds,claim_ceiling_seconds,max_seconds)),'Finite nonnegative active bounds required.')
    ns=1_000_000_000;initial=int(math.ceil(initial_charge_seconds*ns));ceiling=int(math.ceil(claim_ceiling_seconds*ns));maximum=int(math.ceil(max_seconds*ns));excess=initial+ceiling-maximum
    if 0<excess<=1 and claim_ceiling_seconds==max_seconds-initial_charge_seconds:ceiling-=excess
    return initial,ceiling,maximum

def audit_active_kernel_ledger(raw,*,campaign_id,claim,journal_head_sha256,required_heads=()):
    """Independently recompute every attempt, successful credit and reserved cost."""
    ns=1_000_000_000;initial,ceiling,maximum=active_claim_bounds_ns(claim['initial_charge_seconds'],claim['claim_ceiling_seconds'])
    require(ceiling>0 and initial+ceiling<=maximum,'Kernel claim carry/ceiling exceeds fixed campaign allowance.')
    require(isinstance(raw,bytes) and raw and len(raw)<=64*1024**2 and raw.endswith(b'\n'),'Complete bounded newline-terminated kernel ledger required.')
    rows=raw.splitlines();require(len(rows)<=500000 and all(0<len(line)<=16384 for line in rows),'Kernel ledger row/count bound differs.')
    genesis=dict(kind='genesis',format='spades-active-kernel-ledger-v3',campaign_id=campaign_id,segment_id=claim['segment_id'],initial_charge_ns=initial,claim_ceiling_ns=ceiling,maximum_active_ns=maximum,prior_chain_sha256=journal_head_sha256)
    head=journal_head_sha256;charged=initial;observed=0;successful=0;inflight=None;blocked=False;last_ns=None;tokens=set();prefixes={};wanted=set(required_heads)
    def summary(count):
        effective=max(charged,initial+ceiling) if inflight is not None else charged
        return dict(format='spades-active-training-summary-v3',campaign_id=campaign_id,segment_id=claim['segment_id'],initial_charge_seconds=initial/ns,claim_ceiling_seconds=ceiling/ns,charged_active_seconds=effective/ns,measured_kernel_seconds=successful/ns,observed_attempt_seconds=observed/ns,successful_kernel_seconds=successful/ns,attempted_kernel_seconds=observed/ns,failed_attempt_seconds=(observed-successful)/ns,conservative_unmeasured_seconds=max(0,effective-initial-observed)/ns,remaining_active_seconds=max(0,maximum-effective)/ns,maximum_active_seconds=ACTIVE_TRAINING_SECONDS,ledger_sha256=head,ledger_rows=count,claim_blocked=blocked or inflight is not None,unresolved_launch=inflight is not None,local_ledger_authorizes_new_segment=False)
    for index,line in enumerate(rows):
        row=read_json_bytes(line);require(isinstance(row,dict) and set(row)=={'sequence','previous_sha256','event','sha256'} and type(row['sequence']) is int and row['sequence']==index and row['previous_sha256']==head and row['sha256']==digest({k:v for k,v in row.items() if k!='sha256'}),'Kernel ledger exact hash chain/index differs.')
        event=row['event'];require(isinstance(event,dict),'Kernel ledger event required.');kind=event.get('kind')
        if index==0:require(canonical(event)==canonical(genesis),'Kernel ledger source campaign/carry/claim genesis differs.')
        elif kind=='launch_intent':
            require(set(event)=={'kind','operation','token','accepted_state','intent_monotonic_ns'} and inflight is None and not blocked and charged<min(maximum,initial+ceiling) and event['operation'] in ('collect','optimize') and _safe_id(event['token']) and event['token'] not in tokens and integer(event['intent_monotonic_ns']) and (last_ns is None or event['intent_monotonic_ns']>=last_ns),'Nested/reused/after-budget/clock-rollback kernel intent.')
            accepted=event['accepted_state'];require(isinstance(accepted,dict) and set(accepted)=={'updates','env_steps','source_native_sha256'} and integer(accepted['updates'],claim['source']['updates']) and accepted['env_steps']==SOURCE_ENV_STEPS+(accepted['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE and accepted['source_native_sha256']==claim['source']['native']['sha256'],'Kernel intent accepted-state lifetime/source binding differs.')
            inflight=event;tokens.add(event['token']);last_ns=event['intent_monotonic_ns']
        elif kind in ('launch_finished','launch_cancelled'):
            require(inflight is not None and event.get('token')==inflight['token'] and integer(event.get('finished_monotonic_ns')) and event['finished_monotonic_ns']>=inflight['intent_monotonic_ns'],'Kernel intent/completion token or monotonic clock differs.')
            if kind=='launch_finished':
                require(set(event)=={'kind','token','operation','started_monotonic_ns','finished_monotonic_ns','measured_elapsed_ns','failed','synchronized','charged_total_ns','claim_blocked'} and event['operation']==inflight['operation'] and integer(event['started_monotonic_ns'],inflight['intent_monotonic_ns']) and integer(event['measured_elapsed_ns']) and event['finished_monotonic_ns']-event['started_monotonic_ns']==event['measured_elapsed_ns'] and type(event['failed']) is bool and type(event['synchronized']) is bool and type(event['claim_blocked']) is bool and event['claim_blocked'] is (not event['synchronized']),'Exact kernel dispatch-through-sync elapsed/status required.')
                expected=charged+event['measured_elapsed_ns']
                if not event['synchronized']:expected=max(expected,initial+ceiling)
                observed+=event['measured_elapsed_ns']
                if event['synchronized'] and not event['failed']:successful+=event['measured_elapsed_ns']
                blocked=event['claim_blocked']
            else:
                require(set(event)=={'kind','token','reason','charged_total_ns','claim_blocked','finished_monotonic_ns'} and isinstance(event['reason'],str) and event['claim_blocked'] is False,'Exact undispatched cancellation required.')
                expected=charged
            require(type(event.get('charged_total_ns')) is int and event['charged_total_ns']==expected,'Full actual dispatch duration or unknown-reserve charge differs; no clipping.')
            charged=expected;last_ns=event['finished_monotonic_ns'];inflight=None
        else:raise ValueError('Unknown kernel ledger event:'+str(kind))
        head=row['sha256']
        if head in wanted:prefixes[head]=summary(index+1)
    final=summary(len(rows));validate_active_summary(final)
    require(wanted<=set(prefixes),'Native checkpoint references kernel ledger head absent from exact enclosed ledger.')
    return dict(status='PASS',summary=final,prefix_summaries=prefixes,observed_attempt_ns=observed,successful_kernel_ns=successful,charged_active_ns=charged,conservative_charge_if_vm_lost_seconds=max(charged,initial+ceiling)/ns if inflight is not None or blocked else charged/ns,scope='CONTENT_ONLY; clocks and successful calls are recorded evidence, not independent execution attestation.')

PIN_NAMES=dict(operator_sha256='spades_normal_longrun.py',contract_sha256='normal_lr_longrun_contract.py',source_auditor_sha256='normal_lr_source_audit.py',source_v2_contract_sha256='source_v2_contract.py',source_v2_runtime_sha256='source_v2_runtime.py',formula_opponents_sha256='formula_opponents_v3.py',formula_manifest_sha256='formula_manifest_v3.json',plasticity_sha256='plasticity.py',helper_sha256='pinned_fresh_helpers.py',legacy_operator_sha256='spades_legacy_aot.py',original_sha256='original_source.ipynb',production_sha256='production_source.ipynb',frozen_evaluator_sha256='diversity_evaluation.py',sampling_sha256='sampling_policy.py',training_opponents_sha256='training_opponents_v2.py',training_opponent_manifest_sha256='training_opponent_manifest_v2.json')
BUNDLE_CORE={'manifest.json','status.json','provenance.json','authorization.json','campaign_journal.json','external_keeper_permit.json','source_approval_receipt.json','feasibility_receipt.json','external_claim_receipt.json','active_kernel_events.jsonl','checkpoints/original_control1329_native.zip','checkpoints/segment_source_native.zip'}

def _audit_bundle_parts(parts,*,source_artifacts,expected_journal_head_sha256=None,feasibility_report=None):
    """Semantic test seam; production entry always admits the exact source first."""
    historical_journal=read_json_bytes(parts.get('campaign_journal.json',b'null'))
    if isinstance(historical_journal,dict) and isinstance(historical_journal.get('events'),list) and not any(isinstance(event,dict) and event.get('kind')=='scheduling_amended' for event in historical_journal['events']):
        return legacy_contract()._audit_bundle_parts(parts,source_artifacts=source_artifacts,expected_journal_head_sha256=expected_journal_head_sha256,feasibility_report=feasibility_report)
    require(BUNDLE_CORE<=set(parts) and set(parts)<=BUNDLE_CORE|{'checkpoints/current_native.zip'},'Exact portable long-run member set required.')
    manifest=read_json_bytes(parts['manifest.json']);status=read_json_bytes(parts['status.json']);provenance=read_json_bytes(parts['provenance.json']);auth=validate_campaign_authorization(read_json_bytes(parts['authorization.json']));journal=read_json_bytes(parts['campaign_journal.json'])
    require(isinstance(manifest,dict) and set(manifest)=={'format','campaign_id','segment_id','claim_id','journal_head_sha256','reason','nonce','members'} and manifest['format']=='spades-normal-lr-longrun-bundle-v3','Exact normal-LR portable manifest required.')
    require(set(manifest['members'])==set(parts)-{'manifest.json'},'Complete portable member identities required.')
    for name,identity in manifest['members'].items():require(_identity(identity)==content_identity(parts[name]),'Portable member bytes/hash differ:'+name)
    state=validate_campaign_journal(journal,expected_head_sha256=expected_journal_head_sha256)
    require(state['authorization']==auth and state['open_claim'] is not None,'Portable bundle must bind its exact reserved external claim and grant.')
    claim=state['open_claim']['payload'];require(manifest['campaign_id']==auth['campaign_id'] and manifest['segment_id']==claim['segment_id'] and manifest['claim_id']==claim['claim_id'] and manifest['journal_head_sha256']==journal['head_sha256'],'Portable campaign/segment/claim/head differs.')
    require(journal['events'][-1]['kind']=='segment_claimed','Portable launch must bind its final reserved external claim event.')
    acknowledgement=validate_external_claim_receipt(read_json_bytes(parts['external_claim_receipt.json']),source_artifacts=source_artifacts,authorization=auth,claim=claim,prior_journal_head_sha256=journal['events'][-1]['previous_sha256'])
    require(isinstance(status,dict) and status.get('format')=='spades-normal-lr-longrun-status-v3' and isinstance(provenance,dict) and provenance.get('format')=='spades-normal-lr-longrun-provenance-v3','Exact long-run status/provenance required.')
    require(status.get('nonce')==manifest['nonce'] and status.get('reason')==manifest['reason'] and isinstance(manifest['nonce'],str) and bool(manifest['nonce']) and isinstance(manifest['reason'],str) and bool(manifest['reason']),'Portable nonce/reason differs.')
    require(all(status.get(key)==wanted for key,wanted in dict(campaign_id=auth['campaign_id'],segment_id=claim['segment_id'],claim_id=claim['claim_id'],segment_index=claim['segment_index']).items()),'Status campaign/segment binding differs.')
    require(type(status.get('terminal')) is bool and status.get('external_persistence_verified') is False and status.get('hourly_local_checkpoint_is_vm_failure_safe') is False and status.get('external_current_head_independently_attested') is False and status.get('automatic_resume') is False and status.get('automatic_gpu_launch') is False and status.get('automatic_extension') is False and status.get('optimizer_or_rng_reset') is False and status.get('playing_strength_claim') is False,'Local content must not claim external durability, launch authority, resets or strength.')
    require(status.get('prior_source_external_verification_acknowledged') is True and canonical(provenance.get('external_claim_acknowledgement'))==canonical(acknowledgement),'Recorded prior-source parent acknowledgment differs from exact keeper receipt contents.')
    cfg=build_training_config(source_artifacts['source_cfg']);profile=training_profile_metadata(cfg)
    require(canonical(status.get('cfg'))==canonical(cfg)==canonical(provenance.get('cfg')) and auth['configuration_sha256']==digest(cfg) and auth['training_profile_sha256']==digest(profile) and canonical(status.get('training_profile'))==canonical(profile),'Actual normal-LR full configuration/profile differs.')
    require(hashlib.sha256(parts['source_approval_receipt.json']).hexdigest()==SOURCE_APPROVAL_RECEIPT_SHA256 and hashlib.sha256(parts['external_claim_receipt.json']).hexdigest()==claim['external_claim_receipt_sha256'] and hashlib.sha256(parts['feasibility_receipt.json']).hexdigest()==claim['feasibility']['receipt_sha256'],'Exact approved source/feasibility/keeper receipt bytes differ.')
    original=source_artifacts.get('source',source_artifacts);original_raw=original['native_bytes']
    require(parts['checkpoints/original_control1329_native.zip']==original_raw,'Exact original approved control1329 bytes must be portable.')
    segment_raw=parts['checkpoints/segment_source_native.zip'];require(content_identity(segment_raw)==claim['source']['native'],'Exact segment source differs from external claim.')
    require(all(isinstance(record,dict) and all(record.get(k)==v for k,v in content_identity(segment_raw).items()) for record in (status.get('source_native'),provenance.get('source_native'))) and provenance.get('original_source_native_identity')==content_identity(original_raw),'Status/provenance source byte identities must match the exact enclosed originals.')
    if claim['segment_index']==1:
        require(segment_raw==original_raw,'First segment cannot silently select another source.')
        segment_manifest=original['native_manifest'];segment_history=original['native_history'];segment_health=source_artifacts['source_health']
    else:
        segment_manifest,segment_history,segment_work=validate_training_native(segment_raw,source_artifacts=source_artifacts);segment_health=segment_work['health']
    require(segment_manifest['updates']==claim['source']['updates'] and segment_manifest['env_steps']==claim['source']['env_steps'] and digest(segment_manifest['cfg'])==claim['source']['source_cfg_sha256'],'Segment source actual counters/configuration differ from verified receipt.')
    require(canonical(provenance.get('source_cfg'))==canonical(segment_manifest['cfg']) and provenance.get('source_history_sha256')==digest(segment_history) and provenance.get('original_history_sha256')==digest(original['native_history']) and canonical(status.get('source_health'))==canonical(segment_health)==canonical(provenance.get('source_health')),'Actual source config/history/health provenance differs.')
    require(provenance.get('optimizer_or_rng_reset') is False and provenance.get('playing_strength_claim') is False and provenance.get('local_hourly_artifacts_are_external_durability') is False,'Truthful local provenance/no-reset fields required.')
    equality=dict(native_state_leaves=214,all_array_bytes_equal=True,source_native_sha256=hashlib.sha256(segment_raw).hexdigest())
    for key in ('all214_source_restore','all214_after_profile_install'):require(canonical(status.get(key))==canonical(equality)==canonical(provenance.get(key)),'Exact original/full214 restoration attestation differs.')
    records=provenance.get('source_pins');require(isinstance(records,dict),'Actual source file identity closure required.')
    for key,value in state['execution_source_pins'].items():
        require(key in PIN_NAMES and isinstance(records.get(PIN_NAMES[key]),dict) and integer(records[PIN_NAMES[key]].get('bytes'),1) and records[PIN_NAMES[key]].get('sha256')==value,'Source provenance differs from reviewed pin:'+key)
    require(integer(status.get('updates'),segment_manifest['updates']) and status.get('env_steps')==SOURCE_ENV_STEPS+(status['updates']-SOURCE_UPDATES)*DECISIONS_PER_UPDATE,'Latest reported RAM lifetime counters differ.')
    native_report=None;required_heads=set();durable_last=None
    if 'checkpoints/current_native.zip' in parts:
        current=parts['checkpoints/current_native.zip'];info=status.get('current_native')
        require(isinstance(info,dict) and all(info.get(k)==v for k,v in content_identity(current).items()),'Durable native exact identity differs.')
        m,h,work=validate_training_native(current,source_artifacts=source_artifacts);durable_last=h[-1]
        require(durable_last['claim_id']==claim['claim_id'] and m['updates']>=segment_manifest['updates'] and status['updates']>=m['updates'],'Current native claim or RAM/durable counters differ.')
        source_v2_contract().native_auditor().continuation_arrays(current,segment_raw,m['updates']==segment_manifest['updates'])
        current_saved=status['updates']==m['updates'];require(status.get('accepted_state_saved') is current_saved,'Latest accepted RAM-state durability must be explicit and truthful.')
        if current_saved:require(canonical(status.get('last_metrics'))==canonical(work['last_metrics']),'Saved native/status actual metrics differ.')
        required_heads={row['kernel_event_head_sha256'] for row in h if row.get('kind')=='normal_lr_checkpoint' and row.get('claim_id')==claim['claim_id']}
        native_report=dict(identity=content_identity(current),updates=m['updates'],env_steps=m['env_steps'],latest_reported_ram_updates=status['updates'],accepted_state_saved=current_saved,work=work,initial214_array_bytes_independently_verified=m['updates']==segment_manifest['updates'])
    else:
        require(status.get('current_native') is None and status.get('accepted_state_saved') is False,'Never-saved state cannot claim a durable current native.')
    kernel=audit_active_kernel_ledger(parts['active_kernel_events.jsonl'],campaign_id=auth['campaign_id'],claim=claim,journal_head_sha256=journal['head_sha256'],required_heads=required_heads)
    require(canonical(status.get('active_training'))==canonical(kernel['summary'])==canonical(provenance.get('active_training')),'Recorded successful/attempted/conservative active summary differs from independently recomputed ledger.')
    if durable_last is not None:
        require(canonical(durable_last['active_clock'])==canonical(kernel['prefix_summaries'][durable_last['kernel_event_head_sha256']]),'Durable native active clock differs from its exact kernel-ledger prefix.')
    active=kernel['summary'];require(status.get('segment_active_target_overshoot')==(active['charged_active_seconds']-active['initial_charge_seconds']>claim['claim_ceiling_seconds']) and status.get('active_budget_exception')==(active['charged_active_seconds']>ACTIVE_TRAINING_SECONDS),'Terminal overshoot/budget exception must reflect full recorded costs, including failed attempts.')
    permit=read_json_bytes(parts['external_keeper_permit.json']);require(isinstance(permit,dict) and permit.get('format')=='spades-normal-lr-external-claim-permit-v3' and permit.get('journal_head_sha256')==journal['head_sha256'] and permit.get('external_claim_receipt_sha256')==claim['external_claim_receipt_sha256'] and permit.get('segment_id')==claim['segment_id'] and permit.get('source_native_sha256')==claim['source']['native']['sha256'] and utc(permit.get('verified_utc'))<utc(permit.get('valid_until_utc')) and isinstance(permit.get('keeper_reference'),str) and bool(permit['keeper_reference'].strip()),'Recorded parent keeper permit binding differs; this is not independent current-head attestation.')
    if feasibility_report is None:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='normal_lr_feasibility_content_') as directory:
            path=Path(directory)/'receipt.json';path.write_bytes(parts['feasibility_receipt.json'])
            context={} if state['scheduling_amendment'] is None else dict(expected_schedule_amendment_sha256=digest(state['scheduling_amendment']),expected_segment_index=claim['segment_index'],expected_segment_active_seconds=claim['claim_ceiling_seconds'])
            feasibility_report=validate_feasibility_receipt(path,expected_external_receipt_sha256=claim['feasibility']['receipt_sha256'],now_utc=claim['claimed_utc'],**context)
    require(canonical(feasibility_report.get('ledger_feasibility'))==canonical(claim['feasibility']),'Exact admitted feasibility summary differs.')
    if state['scheduling_amendment'] is not None:
        require(feasibility_report.get('schedule_amendment_sha256')==digest(state['scheduling_amendment']) and feasibility_report.get('segment_index')==claim['segment_index'] and feasibility_report.get('maximum_this_segment_active_training_seconds')==claim['claim_ceiling_seconds'],'Audited current feasibility must retain its exact amendment/index/claim binding.')
    successful_total=state['measured_kernel_active_seconds']+active['measured_kernel_seconds'];observed_total=state['observed_attempt_seconds']+active['observed_attempt_seconds'];charged_total=active['charged_active_seconds'];unresolved=active['unresolved_launch'] or active['claim_blocked']
    require(status.get('successful_campaign_kernel_seconds')==successful_total and status.get('recorded_successful_active_target_reached') is (successful_total>=ACTIVE_TRAINING_SECONDS) and status.get('externally_verified_campaign_complete') is False,'Recorded training-target credit or external completion claim differs from successful kernel evidence.')
    actual_target=successful_total>=ACTIVE_TRAINING_SECONDS and status['terminal'] and not unresolved
    return dict(status='PASS',audit_scope='CONTENT_ONLY',verified_content=True,launch_authorized=False,external_persistence_verified=False,external_current_head_independently_attested=False,automatic_resume=False,automatic_extension=False,playing_strength_claim=False,source_native=content_identity(original_raw),segment_source_native=content_identity(segment_raw),campaign_id=auth['campaign_id'],segment_id=claim['segment_id'],claim_id=claim['claim_id'],journal_head_sha256=journal['head_sha256'],configuration_sha256=digest(cfg),terminal=status['terminal'],training_profile=profile,native=native_report,latest_ram_state_independently_verified=bool(native_report and native_report['accepted_state_saved']),active_training=active,cumulative_successful_kernel_seconds=successful_total,cumulative_observed_attempt_seconds=observed_total,cumulative_charged_seconds=charged_total,actual_active_target_reached=actual_target,bounded_completion_without_budget_exception=actual_target and charged_total<=ACTIVE_TRAINING_SECONDS,budget_exhausted_inconclusive=charged_total>=ACTIVE_TRAINING_SECONDS and not actual_target,conservative_charge_if_vm_lost_seconds=kernel['conservative_charge_if_vm_lost_seconds'],feasibility_scope='Recorded parent current facts checked at claim activation; future quota/session availability is not promised.',source_support_dependency='Original source status/authorization/evaluation support must be retained independently; it is not duplicated in this recovery bundle.',metadata_attestation_scope='Content and recorded clocks only. The parent manually verifies external head and persistence; local artifacts implement no external CAS or VM-loss-safe hourly delivery.')

def external_checkpoint_projection(report,*,verified_utc,external_verification_receipt_sha256):
    """Map audited values after the parent independently verifies persistence.

    `claim_charged_active_seconds` subtracts initial carry exactly once. This
    function only prepares journal content; it grants no external authority.
    """
    require(isinstance(report,dict) and report.get('status')=='PASS' and report.get('audit_scope')=='CONTENT_ONLY' and report.get('latest_ram_state_independently_verified') is True and isinstance(report.get('native'),dict),'A fully saved audited current native is required for external verification projection.')
    utc(verified_utc);require(sha(external_verification_receipt_sha256),'Actual external verification receipt digest required.')
    a=validate_active_summary(report['active_training']);n=report['native'];charged=a['charged_active_seconds']-a['initial_charge_seconds']
    return dict(claim_id=report['claim_id'],verified_utc=verified_utc,source=dict(receipt_sha256=external_verification_receipt_sha256,native=_identity(n['identity']),updates=n['updates'],env_steps=n['env_steps'],source_cfg_sha256=report['configuration_sha256']),claim_measured_kernel_active_seconds=a['successful_kernel_seconds'],claim_observed_attempt_seconds=a['attempted_kernel_seconds'],claim_charged_active_seconds=charged,claim_blocked=a['claim_blocked'],unresolved_launch=a['unresolved_launch'],kernel_event_head_sha256=a['ledger_sha256'],segment_terminal=report['terminal'],segment_active_target_overshoot=charged>a['claim_ceiling_seconds'],active_budget_exception=a['charged_active_seconds']>ACTIVE_TRAINING_SECONDS,external_verification_receipt_sha256=external_verification_receipt_sha256)

def audit_normal_lr_longrun_bundle(bundle_path,*,source_native_path,source_approval_receipt_path,expected_journal_head_sha256=None):
    source=validate_source_artifacts(source_native_path,source_approval_receipt_path)
    raw=Path(bundle_path).read_bytes();parts=source_v2_contract()._zip_content(raw,core=BUNDLE_CORE,prefixes=('checkpoints/',))
    report=_audit_bundle_parts(parts,source_artifacts=source,expected_journal_head_sha256=expected_journal_head_sha256);report['bundle']=content_identity(raw);return report
