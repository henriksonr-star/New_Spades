"""One explicitly claimed segment of fixed-LR formula-opponent continuation.

Definitions only. No provider/network/install actions or automatic restart.
The active clock covers actual collect/optimize calls through synchronization.
External serialized claim receipts, rather than local JSON assertions, admit a
new segment. Unknown platform limits or source identity fail closed.
"""
from pathlib import Path
import copy,hashlib,json,math,os,sys,time,uuid,zipfile
from datetime import datetime,timezone

ACTIVE_TRAINING_SECONDS=43200
NS_PER_SECOND=1_000_000_000
FORMAT='spades-normal-lr-longrun-config-v3'
PAYLOAD_FORMAT='spades-normal-lr-longrun-payload-manifest-v3'
PROFILE='normal_lr_formula_longrun_v3'
_RUN_INVOKED=False
_ACTIVE_SESSION=None


class BoundaryStop(RuntimeError):pass


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _sha(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _seconds_ns(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative seconds required.')
    return int(math.ceil(value*NS_PER_SECOND))


def _parse_json(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key:'+key)
            out[key]=value
        return out
    def invalid(value):raise ValueError('Nonfinite JSON token:'+value)
    out=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
    _canonical(out);return out


def _read_json(path):return _parse_json(Path(path).read_bytes())


def _identity(path):
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Absolute regular nonsymlink file required:'+str(path))
    before=path.stat();raw=path.read_bytes();after=path.stat()
    keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    if any(getattr(before,key)!=getattr(after,key) for key in keys):raise ValueError('File changed during read.')
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def _write_new_json(path,value):
    path=Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return _identity(path.absolute())


def _load(path,name,sha):
    import types
    path=Path(path);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Pinned module bytes changed:'+str(path))
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module


class ActiveTrainingClock:
    """Durable kernel ledger inside one externally reserved segment claim.

    A local ledger never authorizes a new segment. Its initial carry and claim
    ceiling must already be bound by the external keeper receipt. An incomplete
    or unsynchronized launch burns the full claim on recovery. The claim is a
    reserve added to initial_charge_seconds, not a new campaign allowance.
    """
    def __init__(self,path,*,campaign_id,segment_id,initial_charge_seconds,
                 claim_ceiling_seconds,prior_chain_sha256,max_seconds=ACTIVE_TRAINING_SECONDS,
                 monotonic_ns=time.monotonic_ns,expected_ledger_head=None):
        if not all(isinstance(x,str) and x for x in (campaign_id,segment_id)):
            raise ValueError('Campaign and approved segment identities required.')
        if (not isinstance(prior_chain_sha256,str) or len(prior_chain_sha256)!=64
                or any(c not in '0123456789abcdef' for c in prior_chain_sha256)):
            raise ValueError('Externally verified prior journal head required.')
        self.path=Path(path);self.monotonic_ns=monotonic_ns
        self.initial_ns=_seconds_ns(initial_charge_seconds);self.claim_ns=_seconds_ns(claim_ceiling_seconds)
        self.max_ns=_seconds_ns(max_seconds)
        if self.max_ns!=43200*NS_PER_SECOND or self.initial_ns>self.max_ns or not 0<self.claim_ns<=self.max_ns-self.initial_ns:
            raise ValueError('Fixed43200-second campaign and bounded positive claim required.')
        self.campaign_id=campaign_id;self.segment_id=segment_id
        self.genesis=dict(kind='genesis',format='spades-active-kernel-ledger-v3',campaign_id=campaign_id,
            segment_id=segment_id,initial_charge_ns=self.initial_ns,claim_ceiling_ns=self.claim_ns,
            maximum_active_ns=self.max_ns,prior_chain_sha256=prior_chain_sha256)
        self.rows=[];self.head=prior_chain_sha256;self.charged_ns=self.initial_ns
        self.inflight=None;self.claim_blocked=False;self.last_ns=None;self.io_failed=False
        if self.path.exists():
            if self.path.is_symlink():raise ValueError('Ledger must not be a symlink.')
            raw=self.path.read_bytes()
            if not raw.endswith(b'\n'):raise ValueError('Truncated kernel ledger; external recovery required.')
            for line in raw.splitlines():self._replay(_parse_json(line))
            if not self.rows or self.rows[0]['event']!=self.genesis:
                raise ValueError('Ledger campaign/carry/claim cannot reset on reopen.')
            if expected_ledger_head is not None and self.head!=expected_ledger_head:
                raise ValueError('Ledger differs from externally verified expected head.')
            if self.inflight is not None:self.claim_blocked=True
        else:
            if expected_ledger_head is not None:raise ValueError('Expected existing ledger is missing.')
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self._append(self.genesis,create=True)
            directory=os.open(str(self.path.parent),os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)

    def _replay(self,row):
        if not isinstance(row,dict) or set(row)!={'sequence','previous_sha256','event','sha256'}:
            raise ValueError('Exact kernel ledger row required.')
        expected=dict(sequence=len(self.rows),previous_sha256=self.head,event=row['event'])
        if row['sequence']!=expected['sequence'] or row['previous_sha256']!=self.head or row['sha256']!=_sha(expected):
            raise ValueError('Kernel ledger hash chain/sequence differs.')
        event=row['event'];kind=event.get('kind')
        if not self.rows:
            if event!=self.genesis:raise ValueError('Ledger campaign/carry/claim cannot reset on reopen.')
        elif kind=='launch_intent':
            if (set(event)!={'kind','operation','token','accepted_state','intent_monotonic_ns'}
                    or self.inflight is not None or event['operation'] not in ('collect','optimize')
                    or self.claim_blocked or self.charged_ns>=min(self.max_ns,self.initial_ns+self.claim_ns)
                    or not isinstance(event['token'],str) or not event['token']
                    or type(event['intent_monotonic_ns']) is not int or event['intent_monotonic_ns']<0
                    or self.last_ns is not None and event['intent_monotonic_ns']<self.last_ns):
                raise ValueError('Nested/invalid kernel ledger intent.')
            _canonical(event['accepted_state'])
            self.inflight=dict(event,started_ns=None)
            self.last_ns=event['intent_monotonic_ns']
        elif kind in ('launch_finished','launch_cancelled'):
            if self.inflight is None or event['token']!=self.inflight['token']:
                raise ValueError('Kernel ledger intent/completion mismatch.')
            charged=event['charged_total_ns']
            end=event.get('finished_monotonic_ns')
            if type(end) is not int or end<self.inflight['intent_monotonic_ns']:
                raise ValueError('Kernel completion clock rolled back.')
            if kind=='launch_finished':
                required={'kind','token','operation','started_monotonic_ns','finished_monotonic_ns',
                    'measured_elapsed_ns','failed','synchronized','charged_total_ns','claim_blocked'}
                start=event.get('started_monotonic_ns');elapsed=event.get('measured_elapsed_ns')
                if (set(event)!=required or event['operation']!=self.inflight['operation']
                        or type(start) is not int or start<self.inflight['intent_monotonic_ns'] or end<start
                        or type(elapsed) is not int or elapsed!=end-start
                        or type(event['failed']) is not bool or type(event['synchronized']) is not bool
                        or type(event['claim_blocked']) is not bool
                        or event['claim_blocked'] is event['synchronized']):
                    raise ValueError('Exact measured kernel duration/status required.')
                expected_charge=self.charged_ns+elapsed
                if not event['synchronized']:expected_charge=max(expected_charge,self.initial_ns+self.claim_ns)
            else:
                if (set(event)!={'kind','token','reason','charged_total_ns','claim_blocked','finished_monotonic_ns'}
                        or not isinstance(event['reason'],str) or event['claim_blocked'] is not False):
                    raise ValueError('Exact undispatched cancellation record required.')
                expected_charge=self.charged_ns
            if type(charged) is not int or charged!=expected_charge:
                raise ValueError('Active charge differs from exact elapsed time/conservative claim.')
            self.charged_ns=charged;self.claim_blocked=bool(event.get('claim_blocked',False));self.inflight=None
            self.last_ns=event.get('finished_monotonic_ns',self.last_ns)
        else:raise ValueError('Unknown kernel ledger event.')
        self.rows.append(row);self.head=row['sha256']

    def _append(self,event,create=False):
        row=dict(sequence=len(self.rows),previous_sha256=self.head,event=event)
        row['sha256']=_sha(row)
        try:
            with self.path.open('x' if create else 'a') as stream:
                stream.write(_canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())
        except BaseException:
            self.io_failed=True;self.claim_blocked=True;raise
        self._replay(row)
        return row

    def _now(self):
        value=self.monotonic_ns()
        if type(value) is not int or value<0 or self.last_ns is not None and value<self.last_ns:
            self.claim_blocked=True;raise RuntimeError('Monotonic kernel clock invalid or rolled back.')
        self.last_ns=value
        return value

    @property
    def charged_seconds(self):
        effective=self.charged_ns
        if self.inflight is not None or self.io_failed:effective=max(effective,self.initial_ns+self.claim_ns)
        return effective/NS_PER_SECOND

    @property
    def remaining_seconds(self):
        effective=self.charged_ns
        if self.inflight is not None or self.io_failed:effective=max(effective,self.initial_ns+self.claim_ns)
        return max(0,self.max_ns-effective)/NS_PER_SECOND

    def before_launch(self,kind,state_identity):
        if self.io_failed or self.claim_blocked or self.inflight is not None:
            raise BoundaryStop('Active claim blocked or launch already unresolved.')
        if kind not in ('collect','optimize'):raise ValueError('Only actual training kernels enter active accounting.')
        if self.charged_ns>=min(self.max_ns,self.initial_ns+self.claim_ns):
            raise BoundaryStop('Active campaign or reserved segment claim exhausted.')
        _canonical(state_identity);now=self._now();token=uuid.uuid4().hex
        self._append(dict(kind='launch_intent',operation=kind,token=token,
            accepted_state=copy.deepcopy(state_identity),intent_monotonic_ns=now))
        return token

    def mark_launch(self,token):
        if self.inflight is None or self.inflight['token']!=token or self.inflight['started_ns'] is not None:
            raise RuntimeError('Exact unstarted kernel token required.')
        if self.io_failed or self.claim_blocked:raise BoundaryStop('Unresolved active claim forbids launch.')
        self.inflight['started_ns']=self._now()

    def cancel_launch(self,token,reason):
        if self.inflight is None or self.inflight['token']!=token or self.inflight['started_ns'] is not None:
            raise RuntimeError('Only a kernel that never started can cancel without charge.')
        return self._append(dict(kind='launch_cancelled',token=token,reason=str(reason),
            charged_total_ns=self.charged_ns,claim_blocked=False,finished_monotonic_ns=self._now()))

    def finish_launch(self,token,failed=False,synchronized=True):
        if self.inflight is None or self.inflight['token']!=token or self.inflight['started_ns'] is None:
            raise RuntimeError('Exact started kernel token required.')
        end=self._now();elapsed=end-self.inflight['started_ns']
        charge=self.charged_ns+elapsed
        # Unknown in-flight work cannot be treated as a completed measured call.
        # Burn the entire externally reserved window, then stop this segment.
        blocked=not synchronized
        if blocked:charge=max(charge,self.initial_ns+self.claim_ns)
        return self._append(dict(kind='launch_finished',token=token,operation=self.inflight['operation'],
            started_monotonic_ns=self.inflight['started_ns'],finished_monotonic_ns=end,
            measured_elapsed_ns=elapsed,failed=bool(failed),synchronized=bool(synchronized),
            charged_total_ns=charge,claim_blocked=blocked))

    def run_kernel(self,kind,function,synchronize,*,state_identity,launch_guard=None):
        token=self.before_launch(kind,state_identity)
        try:
            if launch_guard is not None:launch_guard()
        except BaseException as exc:
            self.cancel_launch(token,str(exc));raise
        self.mark_launch(token);result=None;synchronized=False;failed=True
        try:
            result=function();synchronize(result);synchronized=True;failed=False
            return result
        finally:self.finish_launch(token,failed=failed,synchronized=synchronized)

    def summary(self):
        observed=sum(row['event'].get('measured_elapsed_ns',0) for row in self.rows)
        completed=sum(row['event'].get('measured_elapsed_ns',0) for row in self.rows
            if row['event'].get('synchronized') is True and row['event'].get('failed') is False)
        effective=self.charged_ns
        if self.inflight is not None or self.io_failed:effective=max(effective,self.initial_ns+self.claim_ns)
        return dict(format='spades-active-training-summary-v3',campaign_id=self.campaign_id,
            segment_id=self.segment_id,initial_charge_seconds=self.initial_ns/NS_PER_SECOND,
            claim_ceiling_seconds=self.claim_ns/NS_PER_SECOND,charged_active_seconds=self.charged_seconds,
            # Compatibility names explicitly alias successful synchronization
            # and all observed attempts. Failed/unsynchronized time is charged
            # but can never establish successful twelve-hour training credit.
            measured_kernel_seconds=completed/NS_PER_SECOND,successful_kernel_seconds=completed/NS_PER_SECOND,
            observed_attempt_seconds=observed/NS_PER_SECOND,attempted_kernel_seconds=observed/NS_PER_SECOND,
            failed_attempt_seconds=(observed-completed)/NS_PER_SECOND,
            conservative_unmeasured_seconds=max(0,effective-self.initial_ns-observed)/NS_PER_SECOND,
            remaining_active_seconds=self.remaining_seconds,maximum_active_seconds=ACTIVE_TRAINING_SECONDS,
            ledger_sha256=self.head,ledger_rows=len(self.rows),claim_blocked=self.claim_blocked or self.inflight is not None,
            unresolved_launch=self.inflight is not None,local_ledger_authorizes_new_segment=False)


PIN_NAMES=dict(operator_sha256='spades_normal_longrun.py',contract_sha256='normal_lr_longrun_contract.py',
    source_auditor_sha256='normal_lr_source_audit.py',source_v2_contract_sha256='source_v2_contract.py',
    source_v2_runtime_sha256='source_v2_runtime.py',formula_opponents_sha256='formula_opponents_v3.py',
    formula_manifest_sha256='formula_manifest_v3.json',plasticity_sha256='plasticity.py',
    helper_sha256='pinned_fresh_helpers.py',legacy_operator_sha256='spades_legacy_aot.py',
    original_sha256='original_source.ipynb',production_sha256='production_source.ipynb',
    frozen_evaluator_sha256='diversity_evaluation.py',sampling_sha256='sampling_policy.py',
    training_opponents_sha256='training_opponents_v2.py',
    training_opponent_manifest_sha256='training_opponent_manifest_v2.json')
HELD_V2_RUNTIME_SHA256='7992eb9fc2dcbf42d553051018635eaff1ca7401d216816be1107a5f3ab5d1ce'
PERMIT_KEYS={'format','journal_head_sha256','external_claim_receipt_sha256','segment_id',
    'source_native_sha256','verified_utc','valid_until_utc','keeper_reference'}


def _no_jax():
    if any(name in ('jax','jaxlib','jax_plugins') or name.startswith(('jax.','jaxlib.','jax_plugins.')) for name in sys.modules):
        raise RuntimeError('Fresh kernel before framework import required; no implicit restart.')


def _snapshot_bank_from_native(raw,cfg):
    import io
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        with np.load(io.BytesIO(archive.read('state.npz')),allow_pickle=False) as arrays:
            values=[arrays['leaf_%05d'%i] for i in (203,205)]
            if any(value.shape!=() or str(value.dtype)!='int32' for value in values):
                raise ValueError('Exact snapshot count/retirement scalar schema required.')
            count,retiring=[int(value) for value in values]
    capacity=cfg['snapshot_size']
    if not 0<count<=capacity or not -1<=retiring<capacity:
        raise ValueError('Historical snapshot bank is empty or invalid.')
    available=[slot for slot in range(count) if slot!=retiring]
    if not available:raise ValueError('Historical snapshot bank has no usable slot; silent fallback is forbidden.')
    return dict(count=count,retiring=retiring,capacity=capacity,usable_slots=available)


def _utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None or result.utcoffset().total_seconds()!=0:raise ValueError('Explicit UTC required.')
    return result


def _host_inputs(config_path,payload_dir):
    _no_jax();payload=Path(payload_dir).absolute();config=_read_json(config_path)
    required={'format','source_native_path','original_source_native_path','source_approval_receipt_path',
        'feasibility_receipt_path','campaign_journal_path','external_claim_receipt_path','output_dir',
        'authorization','frontend_preparation'}
    if not isinstance(config,dict) or not required<=set(config) or set(config)-required-{'device_uuid'} or config['format']!=FORMAT:
        raise ValueError('Exact normal-LR long-run configuration required.')
    manifest=_read_json(payload/'normal_longrun_payload_manifest.json')
    if manifest.get('format')!=PAYLOAD_FORMAT or not isinstance(manifest.get('members'),dict):
        raise ValueError('Reviewed full payload manifest required.')
    identities={}
    for name,expected in manifest['members'].items():
        if (not isinstance(name,str) or name.startswith('/') or '\\' in name
                or any(part in ('','..','.') for part in name.split('/'))):raise ValueError('Unsafe payload member.')
        info=_identity(payload/name)
        if expected!=dict(bytes=info['bytes'],sha256=info['sha256']):raise ValueError('Payload source identity differs:'+name)
        identities[name]=info
    if any(name not in identities for name in PIN_NAMES.values()):raise ValueError('Full reviewed source closure required.')
    if identities['source_v2_runtime.py']['sha256']!=HELD_V2_RUNTIME_SHA256:
        raise ValueError('Exact held V2 namespace schema helper required.')
    v2=_load(payload/'source_v2_runtime.py','normal_source_v2_'+uuid.uuid4().hex,HELD_V2_RUNTIME_SHA256)
    for name,sha in v2.CORE_PINS.items():
        if name not in identities or identities[name]['sha256']!=sha:raise ValueError('Immutable base source changed:'+name)
    contract=_load(payload/'normal_lr_longrun_contract.py','normal_contract_'+uuid.uuid4().hex,
        identities['normal_lr_longrun_contract.py']['sha256'])
    pins={key:identities[name]['sha256'] for key,name in PIN_NAMES.items()}
    contract.validate_campaign_authorization(config['authorization'],expected_source_pins=pins)
    if contract.ACTIVE_TRAINING_SECONDS!=ACTIVE_TRAINING_SECONDS:raise ValueError('Fixed active budget differs.')
    journal=_read_json(config['campaign_journal_path']);state=contract.validate_campaign_journal(journal)
    if state['authorization']!=config['authorization'] or state['open_claim'] is None:
        raise ValueError('A pre-existing externally reserved single segment claim is required.')
    claim=state['open_claim']['payload']
    if (claim['segment_index'] not in (1,2)
            or claim['claim_ceiling_seconds']!=min(21600.,ACTIVE_TRAINING_SECONDS-claim['initial_charge_seconds'])):
        raise ValueError('Exactly two sequential six-hour active allocations; final allocation uses remaining budget.')
    original=contract.validate_source_artifacts(config['original_source_native_path'],config['source_approval_receipt_path'])
    original_identity=_identity(Path(config['original_source_native_path']).absolute())
    current_identity=_identity(Path(config['source_native_path']).absolute())
    if claim['segment_index']==1:
        if current_identity['sha256']!=original_identity['sha256']:raise ValueError('First segment must use exact approved Control1329.')
        source=original
    else:
        source=contract.validate_continuation_artifacts(config['source_native_path'],source_artifacts=original,
            journal_state=state,expected_head_sha256=journal['head_sha256'])
    if any(current_identity[key]!=claim['source']['native'][key] for key in ('bytes','sha256')):
        raise ValueError('Current source must equal latest external claim native; no fallback or stale source.')
    feasibility_report=contract.validate_feasibility_receipt(config['feasibility_receipt_path'],
        expected_external_receipt_sha256=claim['feasibility']['receipt_sha256'])
    feasibility=feasibility_report['ledger_feasibility']
    if feasibility!=claim['feasibility']:raise ValueError('Exact currently verified feasibility summary required.')
    if _identity(Path(config['external_claim_receipt_path']).absolute())['sha256']!=claim['external_claim_receipt_sha256']:
        raise ValueError('Exact external keeper claim receipt required.')
    if journal['events'][-1]['kind']!='segment_claimed':
        raise ValueError('Launch requires the newly reserved unused claim as the final external journal event.')
    external_ack=contract.validate_external_claim_receipt(_read_json(config['external_claim_receipt_path']),
        source_artifacts=original,authorization=config['authorization'],claim=claim,
        prior_journal_head_sha256=journal['events'][-1]['previous_sha256'])
    cfg=contract.build_training_config(original['source_cfg'])
    if (cfg['lr']!=.0002 or _sha(cfg)!=config['authorization']['configuration_sha256']
            or _sha(contract.training_profile_metadata(cfg))!=config['authorization']['training_profile_sha256']):
        raise ValueError('Actual fixed normal-LR/full training profile must match the new grant.')
    if source['source_health'].get('should_stop') is not False:
        raise RuntimeError('Verified source optimizer health forbids continuation.')
    snapshot_bank=_snapshot_bank_from_native(source['native_bytes'],source['source_cfg'])
    for key in ('source_native_path','original_source_native_path','source_approval_receipt_path',
            'feasibility_receipt_path','campaign_journal_path','external_claim_receipt_path'):
        identities['@input/'+key]=_identity(Path(config[key]).absolute())
    identities['@input/config']=_identity(Path(config_path).absolute())
    return dict(config=config,payload=payload,identities=identities,v2=v2,contract=contract,
        journal=journal,journal_state=state,claim=claim,original=original,source=source,cfg=cfg,
        source_native=current_identity,feasibility=feasibility,snapshot_bank=snapshot_bank,
        external_claim_acknowledgement=external_ack)


def preflight(config_path,payload_dir):
    """Host-only admission. Local files cannot prove an external head is latest."""
    data=_host_inputs(config_path,payload_dir)
    return dict(format='spades-normal-lr-longrun-preflight-v3',status='PASS_CONTENT_ONLY',
        source_native=data['source_native'],source_cfg=data['source']['source_cfg'],cfg=data['cfg'],
        training_profile=data['contract'].training_profile_metadata(data['cfg']),
        feasibility=data['feasibility'],claim=data['claim'],campaign_journal_head=data['journal']['head_sha256'],
        active_training_seconds=ACTIVE_TRAINING_SECONDS,segment_active_target_seconds=data['claim']['claim_ceiling_seconds'],
        source_health=data['source']['source_health'],framework_imported=False,gpu_work_started=False,
        snapshot_bank=data['snapshot_bank'],
        external_claim_acknowledgement=data['external_claim_acknowledgement'],
        external_current_head_independently_attested=False,launch_authorized_by_preflight=False,
        hourly_local_checkpoint_is_external_durability=False,playing_strength_claim=False)


def _validate_permit(data,permit):
    if not isinstance(permit,dict) or set(permit)!=PERMIT_KEYS or permit['format']!='spades-normal-lr-external-claim-permit-v3':
        raise ValueError('Parent keeper must explicitly supply its independently checked current-head permit.')
    now=datetime.now(timezone.utc);claim=data['claim']
    if (permit['journal_head_sha256']!=data['journal']['head_sha256']
            or permit['external_claim_receipt_sha256']!=claim['external_claim_receipt_sha256']
            or permit['segment_id']!=claim['segment_id']
            or permit['source_native_sha256']!=data['source_native']['sha256']
            or not _utc(permit['verified_utc'])<=now<_utc(permit['valid_until_utc'])
            or now>=_utc(claim['activation_deadline_utc'])
            or now>=_utc(data['feasibility']['valid_until_utc'])
            or not isinstance(permit['keeper_reference'],str) or not permit['keeper_reference'].strip()):
        raise ValueError('External permit source/head/segment or activation window differs.')
    # This validates the supplied keeper evidence. It is deliberately not a
    # synthetic "current=True" assertion or a cross-VM locking implementation.
    data['contract'].validate_campaign_journal(data['journal'],expected_head_sha256=permit['journal_head_sha256'])
    data['claim_admitted_at_entry']=True


def _wall_guard(data,label):
    config=data['config'];preparation=config['frontend_preparation']
    if not isinstance(preparation,dict) or set(preparation)!={'started_utc','started_monotonic_ns','host_boot_id'}:
        raise ValueError('Exact first-code-cell wall origin required.')
    now=datetime.now(timezone.utc);mono=time.monotonic_ns()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (preparation['host_boot_id']!=boot or type(preparation['started_monotonic_ns']) is not int
            or mono<preparation['started_monotonic_ns']):raise BoundaryStop('Session host/monotonic origin changed:'+label)
    elapsed=(mono-preparation['started_monotonic_ns'])/NS_PER_SECOND
    if abs((now-_utc(preparation['started_utc'])).total_seconds()-elapsed)>5:
        raise BoundaryStop('Session UTC/monotonic drift:'+label)
    claim=data['claim'];feasibility=data['feasibility']
    if (now>=_utc(claim['launch_deadline_utc'])
            or not data.get('claim_admitted_at_entry') and now>=_utc(feasibility['valid_until_utc'])
            or elapsed>=feasibility['maximum_segment_wall_seconds']-feasibility['export_reserve_seconds']):
        raise BoundaryStop('Verified wall/quota/export-reserve launch boundary:'+label)
    return elapsed


def _preparation_guard(data,label):
    elapsed=_wall_guard(data,label)
    if elapsed>=600:raise BoundaryStop('Nonrenewable first-cell600-second preparation deadline:'+label)
    return elapsed


def install_source_namespace(ns,runtime,v2,v2_contract,payload,identities,original_cfg,guard):
    """Define the V2 source schema before loading; no source state is migrated."""
    report=v2._install_training_expansion(ns,runtime,v2_contract,payload,identities,original_cfg,guard)
    return dict(namespace_only=True,source_state_migration=False,source_cfg_changed=False,definition=report)


def install_formula_profile(ns,runtime,v2,contract,v2_contract,payload,identities,original_cfg,cfg,guard):
    """Install the new profile after source verification, or before exact resume."""
    if ns['CHECKPOINT_TRAINING_REVISION']!=contract.SOURCE_NATIVE_REVISION:
        raise RuntimeError('Exact V2 source namespace required before formula definitions.')
    guard('formula definitions before installation')
    inherited={name:value for name,value in ns.items() if callable(value)}
    inherited_values={name:copy.deepcopy(value) for name,value in ns.items()
        if name.startswith(('DIVERSITY_','EXPANSION_')) or name=='HEURISTIC_NAMES'}
    frozen=copy.deepcopy(ns['FROZEN_OPPONENT_BENCHMARK_RECORD'])
    source_record=copy.deepcopy(ns['OPPONENT_DIVERSITY_RECORD'])
    source_profile=v2_contract.training_profile_metadata(original_cfg)
    source_validator=ns['_validate_checkpoint_manifest'];source_learning_config=ns['learning_config']
    raw=(Path(payload)/'formula_opponents_v3.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identities['formula_opponents_v3.py']['sha256']:
        raise ValueError('Formula definition bytes changed before installation.')
    exec(compile(raw,str(Path(payload)/'formula_opponents_v3.py'),'exec'),ns)
    allowed={'diverse_heuristic_action','decode_opponent_style_id','is_train_opponent_style_id',
        'is_holdout_opponent_style_id','sample_train_opponent_style_ids'}
    if any(ns.get(name) is not value for name,value in inherited.items() if name not in allowed):
        raise RuntimeError('Formula module modified an inherited rule/teacher/model callable.')
    if any(_canonical(ns.get(name))!=_canonical(value) for name,value in inherited_values.items()):
        raise RuntimeError('Formula module modified inherited IDs/benchmark constants.')
    if (_canonical(ns['FORMULA_OPPONENT_MANIFEST'])!=_canonical(_read_json(Path(payload)/'formula_manifest_v3.json'))
            or tuple(ns['FORMULA_TRAIN_STYLE_IDS'])!=tuple(contract.TRAIN_STYLE_IDS)
            or tuple(ns['FORMULA_HOLDOUT_STYLE_IDS'])!=tuple(contract.HELDOUT_STYLE_IDS)
            or ns.get('FORMULA_PROFILE_VALUE')!=PROFILE):raise RuntimeError('Exact formula237/heldout64 manifest required.')
    record=contract.formula_opponent_record()
    def learning_config(overrides=None):
        if not isinstance(overrides,dict) or overrides.get('opponent_training_profile')!=PROFILE:
            return source_learning_config(overrides)
        old=dict(overrides,opponent_training_profile='training_expansion_v2')
        out=source_learning_config(old);out['opponent_training_profile']=PROFILE;return out
    def validator(metadata):
        if metadata.get('training_revision')==contract.SOURCE_NATIVE_REVISION:return source_validator(metadata)
        saved=metadata.get('cfg')
        if (metadata.get('training_revision')!=contract.NATIVE_REVISION or not isinstance(saved,dict)
                or saved.get('opponent_training_profile')!=PROFILE
                or _canonical(metadata.get('opponent_diversity'))!=_canonical(record)
                or _canonical(metadata.get('opponent_training_profile'))!=_canonical(contract.training_profile_metadata(saved))
                or _canonical(metadata.get('source_opponent_training_profile'))!=_canonical(source_profile)
                or _canonical(metadata.get('frozen_opponent_benchmark'))!=_canonical(frozen)):
            raise ValueError('Exact formula native/current/source/frozen profile metadata required.')
        old=copy.deepcopy(metadata);old['training_revision']=contract.SOURCE_NATIVE_REVISION
        old['cfg']=dict(saved,opponent_training_profile='training_expansion_v2')
        old['opponent_diversity']=source_record;old['opponent_training_profile']=source_profile
        old.pop('source_opponent_training_profile',None);source_validator(old)
        learning_config(saved);return saved
    def styles(array,n_env,legacy=False):
        values=ns['np'].asarray(array);permitted=tuple(range(13)) if legacy else tuple(contract.TRAIN_STYLE_IDS)
        if values.shape!=(int(n_env),) or str(values.dtype)!='int32' or not ns['np'].isin(values,permitted).all():
            raise ValueError('Formula native contains invalid/held-out style IDs.')
    ns['learning_config']=learning_config;ns['_validate_checkpoint_manifest']=validator
    ns['_validate_diversity_native_styles']=styles;ns['CHECKPOINT_TRAINING_REVISION']=contract.NATIVE_REVISION
    ns['TRAINING_REVISION']+='+formula-opponent-v3';ns['OPPONENT_DIVERSITY_RECORD']=copy.deepcopy(record)
    ns['SOURCE_OPPONENT_TRAINING_PROFILE']=copy.deepcopy(source_profile)
    ns['_expansion_training_profile_metadata']=contract.training_profile_metadata
    ns['_FORMULA_SOURCE_NATIVE_REVISION']=contract.SOURCE_NATIVE_REVISION
    save=v2._checkpoint_definition(payload,'save_checkpoint')
    target="        'opponent_diversity': _json_safe(OPPONENT_DIVERSITY_RECORD),\n"
    replacement=target+"        'frozen_opponent_benchmark': _json_safe(FROZEN_OPPONENT_BENCHMARK_RECORD),\n"+"        'opponent_training_profile': _json_safe(_expansion_training_profile_metadata(cfg)),\n"+"        'source_opponent_training_profile': _json_safe(SOURCE_OPPONENT_TRAINING_PROFILE),\n"
    if save.count(target)!=1:raise RuntimeError('Pinned native metadata source differs.')
    save=save.replace(target,replacement,1)
    loader=v2._checkpoint_definition(payload,'load_checkpoint')
    target='revision not in (TRAINING_REVISION, CHECKPOINT_TRAINING_REVISION)'
    if loader.count(target)!=1:raise RuntimeError('Pinned native loader source differs.')
    loader=loader.replace(target,'revision not in (TRAINING_REVISION, CHECKPOINT_TRAINING_REVISION, _FORMULA_SOURCE_NATIVE_REVISION)',1)
    exec(compile(save+loader,'reviewed_formula_native_schema_v3.py','exec'),ns)
    runtime._aot_diversity_contract=lambda:copy.deepcopy(record);runtime._aot_validate_style_array=styles
    guard('formula definitions after installation')
    return dict(format='spades-formula-profile-install-v3',source_native_revision=contract.SOURCE_NATIVE_REVISION,
        native_revision=contract.NATIVE_REVISION,training_profile=contract.training_profile_metadata(cfg),
        state_arrays_modified=False,optimizer_reset=False,rng_reset=False)


def _namespace_binding(ns):
    names=('TRAINING_REVISION','CHECKPOINT_TRAINING_REVISION','OPPONENT_DIVERSITY_RECORD',
        'FROZEN_OPPONENT_BENCHMARK_RECORD','SOURCE_OPPONENT_TRAINING_PROFILE',
        'DIVERSITY_OPPONENT_MANIFEST','EXPANSION_OPPONENT_MANIFEST','FORMULA_OPPONENT_MANIFEST',
        'DIVERSITY_TRAIN_STYLE_IDS','EXPANSION_TRAIN_STYLE_IDS','FORMULA_TRAIN_STYLE_IDS',
        'FORMULA_HOLDOUT_STYLE_IDS')
    return _sha({name:ns[name] for name in names})


def _session_guard(s,label):
    data=s['data'];_wall_guard(data,label)
    if s.get('terminal'):raise BoundaryStop('Terminal segment forbids another workload:'+label)
    clock=s['active_clock']
    if (clock.claim_blocked or clock.io_failed or clock.charged_ns>=
            min(clock.max_ns,clock.initial_ns+clock.claim_ns)):
        raise BoundaryStop('Active allocation exhausted or unresolved:'+label)
    s['runtime']._aot_kaggle_identity_guard(s);s['v2']._memory_options_guard(s['options'])
    ns=s['ns'];runtime=s['runtime']
    if (_canonical(s['cfg'])!=_canonical(s['saved_cfg']) or s['handles']!=s['saved_handles']
            or runtime._aot_eval_namespace_identity(ns)!=s['namespace_callable_identity']
            or runtime._aot_eval_namespace_identity(vars(runtime))!=s['operator_callable_identity']
            or ns['_gpu_runtime_identity']()!=s['runtime_identity']
            or _namespace_binding(ns)!=s['namespace_binding']):
        raise RuntimeError('Training source/configuration/callable identity changed:'+label)
    for name,info in data['identities'].items():
        stat=Path(info['path']).stat()
        if Path(info['path']).is_symlink() or tuple(getattr(stat,key) for key in
                ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))!=s['stat_pins'][name]:
            raise RuntimeError('Pinned payload source changed:'+name)
    runtime._aot_style_identity_guard(s)
    count=int(s['state'].snapshot_count);retiring=int(s['state'].snapshot_retiring)
    if count<=0 or not any(slot!=retiring for slot in range(min(count,s['cfg']['snapshot_size']))):
        raise RuntimeError('Historical snapshot bank has no usable slot; requested snapshot training cannot silently fall back.')
    s['snapshot_bank']=dict(count=count,retiring=retiring,capacity=s['cfg']['snapshot_size'],
        usable_slots=[slot for slot in range(min(count,s['cfg']['snapshot_size'])) if slot!=retiring])
    _wall_guard(data,label+' after validation')


def _active_launch(s,kind,function,arguments):
    """Keep the frozen AOT admission route; replace only launch accounting."""
    runtime=s['runtime'];admission=runtime._aot_admit(s,kind,arguments)
    row=dict(kind=kind,admission=admission,before=s['ns']['_gpu_memory_snapshot'](),started=False,synchronized=False)
    s['recent_calls'].append(row);s['recent_calls']=s['recent_calls'][-8:]
    _write_status(s)
    def invoke():
        row['started']=True;result=function()
        s['active_device_result']=result;s['recovery_sink']['active_device_result']=result
        return result
    result=s['active_clock'].run_kernel(kind,invoke,lambda result:runtime._aot_sync(s,result),
        state_identity=dict(updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
            source_native_sha256=s['data']['source_native']['sha256']),
        launch_guard=lambda:_session_guard(s,kind+' after durable intent before dispatch'))
    last=s['active_clock'].rows[-1]['event']
    row.update(synchronized=True,seconds=last['measured_elapsed_ns']/NS_PER_SECOND,
        active_training_summary=s['active_clock'].summary())
    return result,row


def _prepare_session(data,root):
    payload=data['payload'];ids=data['identities'];v2=data['v2'];contract=data['contract']
    stat_pins=v2._pin_sources(ids)
    runtime=_load(payload/'spades_legacy_aot.py','normal_legacy_'+uuid.uuid4().hex,
        ids['spades_legacy_aot.py']['sha256'])
    _preparation_guard(data,'before device inventory')
    platform=v2._choose_device(runtime,data['config'].get('device_uuid'))['platform']
    for key in ('XLA_FLAGS','CUBLAS_WORKSPACE_CONFIG','TF_DETERMINISTIC_OPS','JAX_DEFAULT_MATMUL_PRECISION'):
        if os.environ.get(key) is not None:raise RuntimeError('Original unset compiler/precision environment required:'+key)
    for key,allowed,default in (('XLA_PYTHON_CLIENT_PREALLOCATE',('false',),'false'),
            ('XLA_PYTHON_CLIENT_MEM_FRACTION',('0.96','.96'),'0.96')):
        if key in os.environ and os.environ[key] not in allowed:raise RuntimeError('Allocator environment differs:'+key)
        os.environ.setdefault(key,default)
    os.environ['CUDA_VISIBLE_DEVICES']=platform['selected_gpu_uuid']
    _no_jax();_preparation_guard(data,'before framework imports')
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax
    from flax import linen as nn
    import optax
    stack=dict(jax='0.11.1',jaxlib='0.11.1',flax='0.12.10',optax='0.2.8')
    if ({'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),'flax':flax.__version__,'optax':optax.__version__}!=stack
            or jax.config.jax_enable_x64 or jax.config.jax_disable_jit or jax.config.jax_default_matmul_precision is not None):
        raise RuntimeError('Exact reviewed stack/config/precision required.')
    _preparation_guard(data,'after framework imports before device initialization')
    devices=jax.devices()
    if len(devices)!=1 or jax.default_backend()!='gpu' or 'T4' not in str(devices[0].device_kind):
        raise RuntimeError('Exactly one selected physical T4 required.')
    original=_read_json(payload/'original_source.ipynb');production=_read_json(payload/'production_source.ipynb')
    ns=dict(__name__='spades_normal_lr_longrun_v3',Path=Path,os=os,time=time,json=json,math=math,np=np,
        jax=jax,jnp=jnp,flax=flax,nn=nn,optax=optax,_SPADES_KAGGLE_PLATFORM=copy.deepcopy(platform),
        _fresh_frozen_evaluation_path=str(payload/'diversity_evaluation.py'))
    guard=lambda label:_preparation_guard(data,label)
    exec(compile(''.join(production['cells'][12]['source']),'pinned_memory_cell12.py','exec'),ns)
    options=dict(ns['GPU_TUNING']);v2._memory_options_guard(options)
    snapshot=ns['_gpu_memory_snapshot']()
    if any(snapshot.get(key)!=platform[expected] for key,expected in (
            ('gpu_uuid','selected_gpu_uuid'),('gpu_pci_bus_id','selected_gpu_pci_bus_id'),('board_total_bytes','selected_gpu_total_bytes'))):
        raise RuntimeError('Actual selected UUID/PCI/capacity mapping required.')
    def admit(estimate,label):
        guard(label+' before admission');v2._memory_options_guard(options)
        record=ns['_gpu_memory_admission'](estimate,options,label)
        detail=record.get('budget_details',{});snap=record.get('snapshot',{})
        if (record.get('status')!='admitted' or record.get('overlap_credit_bytes')!=0
                or snap.get('platform')!='gpu' or detail.get('pool_counter_source')!='pool_bytes'
                or detail.get('pool_counter_status')!='validated_current_total'
                or 'pool_bytes' not in snap.get('jax_memory_stats_raw',{})):
            raise RuntimeError('Current validated pool/board memory admission required.')
        guard(label+' after admission');return record
    admit(1,'definition constants')
    for index in (6,8,10,18):
        guard('definition cell'+str(index));notebook=original if index==6 else production
        exec(compile(''.join(notebook['cells'][index]['source']),'pinned_cell%d.py'%index,'exec'),ns)
    if ns['TRAINING_REVISION']!='strong-teacher-hand-gae-v3' or len(ns['HEURISTIC_NAMES'])!=13:
        raise RuntimeError('Exact thirteen source policies and training definitions required.')
    sampling=_load(payload/'sampling_policy.py','normal_sampling_'+uuid.uuid4().hex,ids['sampling_policy.py']['sha256'])
    ns['sample_train_opponent_style_ids']=sampling.build_train_sampler(jax,jnp,ns['sample_train_opponent_style_ids'],sampling.identity_sampling_policy())
    plasticity=_load(payload/'plasticity.py','normal_plasticity_'+uuid.uuid4().hex,ids['plasticity.py']['sha256'])
    plasticity.install_training(ns,str(payload/'production_source.ipynb'));plasticity.activate_candidate_revision(ns)
    v2_contract=_load(payload/'source_v2_contract.py','normal_v2_contract_'+uuid.uuid4().hex,ids['source_v2_contract.py']['sha256'])
    source_definition=install_source_namespace(ns,runtime,v2,v2_contract,payload,ids,data['original']['source_cfg'],guard)
    current_cfg=data['source']['source_cfg'];cfg=copy.deepcopy(data['cfg']);resumed=current_cfg.get('opponent_training_profile')==PROFILE
    if resumed:
        formula=install_formula_profile(ns,runtime,v2,contract,v2_contract,payload,ids,
            data['original']['source_cfg'],cfg,guard)
    original_load=ns['gpu_admit_checkpoint_load']
    def load_admit(saved,size):
        guard('native before transfer admission');result=original_load(saved,size);guard('native after transfer admission');return result
    ns['gpu_admit_checkpoint_load']=load_admit
    guard('exact source native restore')
    state,loaded_cfg,history,metadata=ns['load_checkpoint'](data['config']['source_native_path'])
    runtime._aot_sync(dict(ns=ns),state)
    if _canonical(loaded_cfg)!=_canonical(current_cfg) or _canonical(history)!=_canonical(data['source']['native_history']):
        raise RuntimeError('Exact source config/history changed on restoration.')
    equality=runtime._aot_assert_state_matches_source906(ns,state,data['config']['source_native_path'],guard)
    journal_rows=ns['_optimizer_resume_journal'](history,current_cfg,int(state.updates))
    health=ns['_optimizer_health'](journal_rows)
    if health.get('should_stop') is not False:raise RuntimeError('Restored original-signature optimizer health forbids continuation.')
    history=copy.deepcopy(history)
    if not resumed:
        formula=install_formula_profile(ns,runtime,v2,contract,v2_contract,payload,ids,
            data['original']['source_cfg'],cfg,guard)
        history.append(dict(kind='normal_lr_formula_branch_start',source_native_sha256=data['source_native']['sha256'],
            migration=contract.migration_record(current_cfg,cfg),source_health=copy.deepcopy(health)))
    post_equality=runtime._aot_assert_state_matches_source906(ns,state,data['config']['source_native_path'],guard)
    history.append(dict(kind='normal_lr_segment_start',claim_id=data['claim']['claim_id'],
        journal_head_sha256=data['journal']['head_sha256'],
        source_native_identity={key:data['source_native'][key] for key in ('bytes','sha256')},configuration_sha256=_sha(cfg)))
    count=int(state.snapshot_count);retiring=int(state.snapshot_retiring)
    if count<=0 or not any(slot!=retiring for slot in range(min(count,cfg['snapshot_size']))):
        raise RuntimeError('Requested historical snapshot bank has no usable slot.')
    trainer=ns['make_train_step'](cfg)
    guard('abstract training shape');abstract=jax.eval_shape(lambda:ns['initialize_training'](cfg))
    after_collect,batch,_=jax.eval_shape(trainer.collect,abstract)
    S,B=ns['_gpu_tree_bytes'](abstract),ns['_gpu_tree_bytes'](batch)
    admit(S,'collect AOT preparation');guard('collect lowering');lowered=trainer.collect.lower(abstract)
    guard('collect compiling');collect=lowered.compile();del lowered
    C=ns['_gpu_compiled_bytes'](collect.memory_analysis())
    admit(C['total_bytes']+S+65536,'optimizer AOT preparation');guard('optimizer lowering');lowered=trainer.optimize.lower(after_collect,batch)
    guard('optimizer compiling');optimize=lowered.compile();del lowered
    O=ns['_gpu_compiled_bytes'](optimize.memory_analysis());estimate=max(C['total_bytes'],O['total_bytes'])+S+65536
    admit(estimate,'sequential AOT selected')
    guard('after final AOT preparation before session publication')
    profile=dict(device=devices[0],optimize_input_specs=runtime._aot_input_specs(jax,np,((after_collect,batch),{})),
        options=options,execution_handles=dict(collect=collect,optimize=optimize),state_bytes=S,batch_bytes=B,
        collect=C,optimize=O,estimated_peak_bytes=estimate,retained_rollback_states=1)
    clock=ActiveTrainingClock(root/'active_kernel_events.jsonl',campaign_id=data['config']['authorization']['campaign_id'],
        segment_id=data['claim']['segment_id'],initial_charge_seconds=data['claim']['initial_charge_seconds'],
        claim_ceiling_seconds=data['claim']['claim_ceiling_seconds'],prior_chain_sha256=data['journal']['head_sha256'])
    helpers=_load(payload/'pinned_fresh_helpers.py','normal_native_helpers_'+uuid.uuid4().hex,ids['pinned_fresh_helpers.py']['sha256'])
    s=dict(data=data,root=str(root),v2=v2,runtime=runtime,ns=ns,state=state,cfg=cfg,saved_cfg=copy.deepcopy(cfg),
        history=history,journal_rows=ns['_optimizer_resume_journal'](history,cfg,int(state.updates)),
        profile=profile,options=options,handles=dict(profile['execution_handles']),saved_handles=dict(profile['execution_handles']),
        platform=copy.deepcopy(platform),saved_platform=copy.deepcopy(platform),runtime_identity=ns['_gpu_runtime_identity'](),
        accepted_update=int(state.updates),accepted_env_steps=state.env_steps,source_leaf_specs=metadata['leaves'],
        candidate_revision=ns['CHECKPOINT_TRAINING_REVISION'],source_health=health,equality=equality,post_migration_equality=post_equality,
        source_definition=source_definition,formula_definition=formula,active_clock=clock,helpers=helpers,
        terminal=False,stop_reason=None,recent_calls=[],recovery_sink={},stat_pins=stat_pins,
        local_natives=[],exports=[],last_metrics=None,last_native=None,last_hourly_native=None,
        next_native_wall=900.,next_archive_wall=3600.,snapshot_bank=dict(count=count,retiring=retiring))
    original_finite=runtime._aot_host_finite
    def finite(session,tree,stage,expected_leaves,payload_guard_bytes):
        if stage=='optimizer':
            if expected_leaves!=238:raise RuntimeError('Frozen optimizer schema changed.')
            expected_leaves=242
        return original_finite(session,tree,stage,expected_leaves,payload_guard_bytes)
    runtime._aot_host_finite=finite;runtime._aot_launch=_active_launch
    runtime._aot_boundary=lambda session,label:_session_guard(session,label)
    runtime._aot_elapsed=lambda session:session['active_clock'].charged_seconds
    runtime._aot_limit_reason=lambda session:('active_allocation_exhausted' if session['active_clock'].charged_ns>=min(
        session['active_clock'].max_ns,session['active_clock'].initial_ns+session['active_clock'].claim_ns) else None)
    s['namespace_callable_identity']=runtime._aot_eval_namespace_identity(ns)
    s['operator_callable_identity']=runtime._aot_eval_namespace_identity(vars(runtime));s['namespace_binding']=_namespace_binding(ns)
    s['recovery_sink'].update(state=state,cfg=copy.deepcopy(cfg),history=history,session=s)
    _session_guard(s,'prepared before first actual collect');return s


def _assignment_diagnostic(s):
    """Count current assignments, never infer action/time exposure from them."""
    ns=s['ns'];np=ns['np'];jax=ns['jax'];state=s['state']
    kinds=np.asarray(jax.device_get(state.league_kind));styles=np.asarray(jax.device_get(state.league_style))
    family=(styles-13)//64;scripted=kinds==2
    count=int(state.snapshot_count);retiring=int(state.snapshot_retiring)
    bank=dict(count=count,retiring=retiring,capacity=s['cfg']['snapshot_size'],
        usable_slots=[slot for slot in range(min(count,s['cfg']['snapshot_size'])) if slot!=retiring])
    return dict(scope='Current accepted-state match assignments; style fields on non-scripted rows are unused.',
        environments=int(kinds.size),selfplay_assignments=int(np.sum(kinds==0)),
        historical_snapshot_assignments=int(np.sum(kinds==1)),scripted_assignments=int(np.sum(scripted)),
        scripted_legacy_assignments=int(np.sum(scripted&(styles<13))),
        scripted_old_family_assignments=int(np.sum(scripted&(styles>=13)&(family<3))),
        scripted_expansion_v2_assignments=int(np.sum(scripted&(family>=5)&(family<=7))),
        scripted_formula_assignments=int(np.sum(scripted&(family==8))),
        scripted_heldout_assignments=int(np.sum(scripted&np.isin(styles,s['data']['contract'].HELDOUT_STYLE_IDS))),
        acted_actions_by_family_measured=False,assignment_counts_are_action_or_time_fractions=False,
        restored_assignments_persist_until_match_completion_or_training_truncation=True,
        historical_bank=bank)


def _status(s):
    clock=s['active_clock'].summary();claim=s['data']['claim']
    successful_campaign=s['data']['journal_state']['measured_kernel_active_seconds']+clock['successful_kernel_seconds']
    return dict(format='spades-normal-lr-longrun-status-v3',campaign_id=s['data']['config']['authorization']['campaign_id'],
        segment_id=claim['segment_id'],claim_id=claim['claim_id'],segment_index=claim['segment_index'],
        source_native=s['data']['source_native'],cfg=s['saved_cfg'],source_health=s['source_health'],
        active_training=clock,updates=s['accepted_update'],env_steps=s['accepted_env_steps'],
        successful_campaign_kernel_seconds=successful_campaign,
        recorded_successful_active_target_reached=successful_campaign>=ACTIVE_TRAINING_SECONDS,
        externally_verified_campaign_complete=False,
        accepted_state_saved=bool(s.get('last_native') and s.get('durable_updates')==s['accepted_update']),
        current_native=s.get('last_native'),terminal=s.get('terminal',False),stop_reason=s.get('stop_reason'),
        error=s.get('error'),last_metrics=s.get('last_metrics'),exports=copy.deepcopy(s.get('exports',[])),
        local_checkpoints=copy.deepcopy(s.get('local_natives',[])),
        training_profile=s['data']['contract'].training_profile_metadata(s['saved_cfg']),
        assignment_diagnostic=s.get('assignment_diagnostic'),
        all214_source_restore=s['equality'],all214_after_profile_install=s['post_migration_equality'],
        segment_active_target_overshoot=(clock['charged_active_seconds']-clock['initial_charge_seconds'])>claim['claim_ceiling_seconds'],
        active_budget_exception=clock['charged_active_seconds']>ACTIVE_TRAINING_SECONDS,
        external_persistence_verified=False,hourly_local_checkpoint_is_vm_failure_safe=False,
        prior_source_external_verification_acknowledged=True,
        external_current_head_independently_attested=False,
        execution_authority='Parent-controlled serialized keeper; artifact validates supplied receipt contents only.',
        automatic_resume=False,automatic_gpu_launch=False,automatic_extension=False,
        historical_ledger_preserved=True,optimizer_or_rng_reset=False,playing_strength_claim=False)


def _write_status(s):
    root=Path(s['root']);temporary=root/('status.'+uuid.uuid4().hex+'.tmp')
    _write_new_json(temporary,_status(s));temporary.replace(root/'status.json')
    return _status(s)


def _remove_owned_pending(s,info,protected):
    """Rotate only this new run's superseded local pending files."""
    path=Path(info['path']);root=Path(s['root']).resolve()
    if str(path) in protected:return
    if root not in path.resolve().parents or path.is_symlink() or _identity(path)!=info:
        raise RuntimeError('Refuse rotation outside exact owned pending file.')
    path.unlink()


def _save_native(s,reason,terminal=False):
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
    s['data']['contract'].validate_training_native(Path(native).read_bytes(),source_artifacts=s['data']['original'],
        expected_config=s['saved_cfg'])
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


def _export_bundle(s,reason):
    """Immutable local recovery bytes; external persistence requires parent ACK."""
    root=Path(s['root']);data=s['data'];nonce=uuid.uuid4().hex
    members={}
    def snapshot(name,path,expected=None):
        info=_identity(Path(path).absolute());raw=Path(path).read_bytes()
        if expected and any(info[k]!=expected[k] for k in ('bytes','sha256')):raise IOError('Snapshot source identity changed.')
        members[name]=raw
    snapshot('checkpoints/original_control1329_native.zip',data['config']['original_source_native_path'])
    snapshot('checkpoints/segment_source_native.zip',data['config']['source_native_path'],data['source_native'])
    if s.get('last_native'):snapshot('checkpoints/current_native.zip',s['last_native']['path'],s['last_native'])
    snapshot('active_kernel_events.jsonl',s['active_clock'].path)
    for name,key in (('source_approval_receipt.json','source_approval_receipt_path'),
            ('feasibility_receipt.json','feasibility_receipt_path'),('external_claim_receipt.json','external_claim_receipt_path')):
        snapshot(name,data['config'][key])
    provenance=dict(format='spades-normal-lr-longrun-provenance-v3',source_native=data['source_native'],
        original_source_native_identity={k:data['original']['native_identity'][k] for k in ('bytes','sha256')},
        source_cfg=data['source']['source_cfg'],cfg=s['saved_cfg'],source_history_sha256=_sha(data['source']['native_history']),
        original_history_sha256=_sha(data['original']['native_history']),source_health=s['source_health'],
        source_pins=data['identities'],platform=s['platform'],all214_source_restore=s['equality'],
        all214_after_profile_install=s['post_migration_equality'],source_namespace=s['source_definition'],
        formula_definition=s['formula_definition'],optimizer_or_rng_reset=False,playing_strength_claim=False,
        external_claim_acknowledgement=data['external_claim_acknowledgement'],
        active_training=s['active_clock'].summary(),local_hourly_artifacts_are_external_durability=False,
        execution_authority='Parent fresh-read/reserve/consume/bind immutable notebook version before one launch; no artifact CAS.')
    status=_status(s);status.update(nonce=nonce,reason=reason)
    for name,value in (('status.json',status),('provenance.json',provenance),
            ('authorization.json',data['config']['authorization']),('campaign_journal.json',data['journal']),
            ('external_keeper_permit.json',data['external_keeper_permit'])):
        members[name]=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    manifest=dict(format='spades-normal-lr-longrun-bundle-v3',campaign_id=status['campaign_id'],
        segment_id=status['segment_id'],claim_id=status['claim_id'],journal_head_sha256=data['journal']['head_sha256'],
        reason=reason,nonce=nonce,members={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in members.items()})
    members['manifest.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    destination=root/('normal_lr_'+reason+'_'+nonce+'.zip');partial=destination.with_suffix('.zip.partial')
    with partial.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for name,raw in members.items():archive.writestr(name,raw)
        stream.flush();os.fsync(stream.fileno())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None or set(archive.namelist())!=set(members):raise IOError('Local bundle CRC/membership failure.')
        for name,expected in manifest['members'].items():
            raw=archive.read(name)
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise IOError('Local bundle exact member readback failed.')
    partial.rename(destination);info=_identity(destination.absolute())
    export=dict(bundle=info,reason=reason,nonce=nonce,external_persistence_verified=False)
    s['exports'].append(export)
    # Retain latest hourly plus new pending/final archive; input/latest external
    # source lives outside this new owned root and is never a rotation target.
    while len(s['exports'])>2:
        old=s['exports'].pop(0);_remove_owned_pending(s,old['bundle'],{info['path']})
    s['last_hourly_native']=s.get('last_native');_write_status(s)
    print(json.dumps(dict(event='local_normal_lr_recovery_bundle',**export),allow_nan=False),flush=True)
    return export


def _train_until_segment_boundary(s):
    """No rollout bound; actual kernel accounting or health/wall limit stops work."""
    try:
        while True:
            _session_guard(s,'before next collect')
            stop=s['runtime']._aot_step(s)
            s['last_health']=s['ns']['_optimizer_health'](s['journal_rows'])
            if stop:raise BoundaryStop(stop)
            elapsed=(time.monotonic_ns()-s['data']['config']['frontend_preparation']['started_monotonic_ns'])/NS_PER_SECOND
            if elapsed>=s['next_archive_wall']:
                _save_native(s,'hourly_local');_export_bundle(s,'hourly_local')
                s['next_archive_wall']=elapsed+3600.;s['next_native_wall']=elapsed+900.
            elif elapsed>=s['next_native_wall']:
                _save_native(s,'supplemental15_local');s['next_native_wall']=elapsed+900.
            _write_status(s)
    except BoundaryStop as exc:
        s['stop_reason']=str(exc)
    except BaseException as exc:
        import traceback
        s['stop_reason']='first_error';s['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        s['terminal']=True
        try:
            _save_native(s,'final',terminal=True)
            s['final_bundle_path']=_export_bundle(s,'final')['bundle']['path']
        except BaseException as exc:
            s['final_bundle_path']=None
            s['preservation_error']=dict(type=type(exc).__name__,message=str(exc))
            s['recovery_sink'].update(state=s['state'],cfg=copy.deepcopy(s['saved_cfg']),history=s['history'])
            _write_status(s)
    return s


def run_segment_once(config_path,payload_dir,*,externally_verified_claim=None):
    """One parent-consumed segment only. No provider launch or automatic resume.

    The parent keeper must independently fresh-read its external journal,
    reserve/consume the exact claim, bind one immutable notebook/version, then
    pass its permit unchanged. Local receipt parsing cannot establish that
    external fact, and a fresh VM must never be used to replay this artifact.
    """
    global _RUN_INVOKED,_ACTIVE_SESSION
    if _RUN_INVOKED:raise RuntimeError('Segment entrypoint already invoked; no automatic retry/re-entry.')
    _RUN_INVOKED=True;data=_host_inputs(config_path,payload_dir);_validate_permit(data,externally_verified_claim)
    data['external_keeper_permit']=copy.deepcopy(externally_verified_claim)
    preparation=data['config']['frontend_preparation'];_wall_guard(data,'entry before framework preparation')
    if (time.monotonic_ns()-preparation['started_monotonic_ns'])/NS_PER_SECOND>600:
        raise BoundaryStop('Frontend staging/setup exceeded its nonrenewable600-second allowance.')
    root=Path(data['config']['output_dir'])
    if not root.is_absolute() or root.is_symlink() or '..' in root.parts or root.exists():
        raise ValueError('Unique new absolute output root required.')
    registry=(Path('/kaggle/working') if Path('/kaggle/working').is_dir() else root.parent)/'.spades_normal_lr_claim_starts_v3'
    registry.mkdir(parents=True,exist_ok=True)
    segment=data['claim']['segment_id']
    if not segment or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-' for c in segment):
        raise ValueError('Safe fixed externally approved segment ID required.')
    _write_new_json(registry/(segment+'.json'),dict(segment_id=segment,claim_id=data['claim']['claim_id'],
        journal_head_sha256=data['journal']['head_sha256'],external_keeper_permit=externally_verified_claim,
        local_claim_is_cross_vm_authority=False,restart_permitted=False))
    root.mkdir(parents=True,exist_ok=False);(root/'checkpoints').mkdir()
    _write_new_json(root/'start_journal.json',dict(format='spades-normal-lr-segment-start-v3',
        claim=data['claim'],frontend_preparation=preparation,external_keeper_permit=externally_verified_claim,
        active_training_target_seconds=data['claim']['claim_ceiling_seconds'],restart_permitted=False,
        external_global_freshness_established_by_artifact=False))
    try:
        s=_prepare_session(data,root);_ACTIVE_SESSION=s
        _train_until_segment_boundary(s)
        result=_status(s);result.update(final_bundle_path=s.get('final_bundle_path'),
            preservation_error=s.get('preservation_error'),output_dir=str(root),active_recovery_retained=True,
            next_segment_requires_parent_external_verification_and_new_claim=True)
    except BaseException as exc:
        import traceback
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if _ACTIVE_SESSION is not None:
            s=_ACTIVE_SESSION;s['terminal']=True;s['error']=error
            result=_status(s);result.update(status='FAILED_ACCEPTED_STATE_RETAINED_NO_RETRY',
                final_bundle_path=s.get('final_bundle_path'),output_dir=str(root),active_recovery_retained=True,
                external_claim_remains_parent_responsibility=True,restart_permitted=False)
        else:
            result=dict(format='spades-normal-lr-preparation-failure-v3',status='PREPARATION_FAILED_NO_RETRY',
                error=error,original_source_preserved=True,final_bundle_path=None,output_dir=str(root),
                active_training_started=False,external_claim_remains_parent_responsibility=True,restart_permitted=False)
    _write_new_json(root/'normal_lr_result.json',result);return result
