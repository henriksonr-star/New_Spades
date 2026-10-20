"""Explicit training-only sampling policy. No imports or execution at import.

The benchmark/opponent behavior/IDs and full native state are unchanged. Host
contract must supply its independently recomputed frozen development audit.
"""
SAMPLING_FORMAT = 'spades-training-sampling-policy-v1'
BASE_MANIFEST_SHA256 = '20e4727fdf4d37d035c9ec1bc5d770eb3510f93c9bd20c680c512ceee5f78f1d'
EVALUATION_SOURCE_SHA256 = '1751557e800d48a596b7a783595aa1bd42ecf6c20b130c97a9a498e583e16ecc'
OPPONENT_SOURCE_SHA256 = '75649c722d2843641322084e157a6c6816bd8a28a32ced416522b9a732d9da55'
HELDOUT_PROTOCOL_SHA256 = '0b5970eb566bfcf3db55ffc5559a80e27c072babffe5e1c64f92d3aa5ed2942d'
TRAIN_STYLE_IDS = tuple(range(13)) + tuple(13+64*f+v for f in range(3) for v in range(32))
DEVELOPMENT_STYLE_IDS = (5,6,7,11) + tuple(13+64*f+v for f in range(3) for v in (0,8,16,24))
PROPOSAL_RULE = 'new-family-upper95-negative-worst-one-4-to5-retain-prior-v1'
POLICY_KEYS = ('format','base_manifest_sha256','evaluation_source_sha256','opponent_semantics_revision',
    'sampler_revision','policy_revision','mode','training_only_override','legacy_probability',
    'new_probability','family_relative_weights','new_family_probabilities','variants_uniform',
    'training_style_ids','parent_policy_sha256','rule','evidence')


def _sampling_require(condition,message):
    if not condition:raise ValueError(message)


def _sampling_canonical(value):
    import json
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sampling_policy_sha256(policy):
    import hashlib
    return hashlib.sha256(_sampling_canonical(policy).encode()).hexdigest()


def _sampling_copy(value):
    import json
    return json.loads(_sampling_canonical(value))


def _sampling_hash(value):
    import re
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _sampling_file(path):
    import hashlib
    from pathlib import Path
    target=Path(path);raw=target.read_bytes()
    return dict(path=str(target),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),raw


def identity_sampling_policy():
    """First unchanged bridge: exact original RNG path, not categorical uniform."""
    return dict(format=SAMPLING_FORMAT,base_manifest_sha256=BASE_MANIFEST_SHA256,
        evaluation_source_sha256=EVALUATION_SOURCE_SHA256,
        opponent_semantics_revision='spades-opponent-diversity-v1',sampler_revision='base_rng_exact_v1',
        policy_revision=0,mode='identity',training_only_override=False,
        legacy_probability=[1,2],new_probability=[1,2],family_relative_weights=[4,4,4],
        new_family_probabilities=[[4,12],[4,12],[4,12]],variants_uniform=32,
        training_style_ids=list(TRAIN_STYLE_IDS),parent_policy_sha256=None,rule=PROPOSAL_RULE,evidence=None)


def _sampling_validate_shape(policy):
    import math
    _sampling_require(isinstance(policy,dict) and set(policy)==set(POLICY_KEYS),'Exact sampling policy schema required.')
    _sampling_canonical(policy)
    base=identity_sampling_policy()
    for key in ('format','base_manifest_sha256','evaluation_source_sha256','opponent_semantics_revision',
                'legacy_probability','new_probability','variants_uniform','training_style_ids','rule'):
        _sampling_require(type(policy[key]) is type(base[key]) and _sampling_canonical(policy[key])==_sampling_canonical(base[key]),'Fixed sampling field changed:'+key)
    weights=policy['family_relative_weights']
    _sampling_require(isinstance(weights,list) and len(weights)==3 and all(type(x) is int and x in (4,5) for x in weights),'Only three bounded relative family weights4or5 are allowed.')
    _sampling_require(_sampling_canonical(policy['new_family_probabilities'])==_sampling_canonical([[x,sum(weights)] for x in weights]), 'Exact integer effective family probability fractions required.')
    revision=policy['policy_revision']
    _sampling_require(type(revision) is int and revision==sum(x==5 for x in weights),'Policy revision must count bounded family increases.')
    if policy['mode']=='identity':
        _sampling_require(_sampling_canonical(policy)==_sampling_canonical(base),'Identity must be exact unchanged base record.')
    else:
        _sampling_require(policy['mode']=='bounded_new_family_tilt' and policy['sampler_revision']=='bounded_family_choice_v1'
            and policy['training_only_override'] is True and 1<=revision<=3
            and _sampling_hash(policy['parent_policy_sha256']) and isinstance(policy['evidence'],dict),
            'Explicit weighted sampler/lineage/evidence required.')
    return _sampling_copy(policy)


def continue_current_policy(prior_policy=None):
    """Keep the exact verified prior sampler, including its RNG path/weights."""
    return _sampling_validate_shape(identity_sampling_policy() if prior_policy is None else prior_policy)


def _sampling_valid_development(report,audit):
    import json,math
    _sampling_require(isinstance(report,dict) and isinstance(audit,dict),'Actual parsed development report and recomputed audit required.')
    _sampling_require(report.get('format')=='spades-diversity-evaluation-report-v1' and report.get('mode')=='development'
        and report.get('status')=='complete' and all(report.get(k) is True for k in ('complete','legal','finite')),
        'Complete finite legal development campaign required.')
    _sampling_require(report.get('evaluation_source_sha256')==EVALUATION_SOURCE_SHA256
        and report.get('opponent_manifest_sha256')==BASE_MANIFEST_SHA256
        and report.get('frozen_opponent_source_sha256')==OPPONENT_SOURCE_SHA256
        and report.get('heldout_protocol_sha256')==HELDOUT_PROTOCOL_SHA256,'Frozen benchmark/opponent/protocol bindings required.')
    expected_plan=dict(mode='development',seed_name='development',seed=2026101921,bootstrap_seed=2026101922,
        style_ids=list(DEVELOPMENT_STYLE_IDS),pairs_per_style=16,games_per_style=32,max_hands=160)
    _sampling_require(report.get('seed_name')=='development' and _sampling_canonical(report.get('plan'))==_sampling_canonical(expected_plan), 'Exact fixed development plan required; heldout evidence is forbidden.')
    _sampling_require(audit.get('status')=='PASS' and audit.get('audit_scope')=='CONTENT_ONLY'
        and audit.get('mode')=='development' and audit.get('cycle_number')==report.get('cycle_number')
        and type(report.get('cycle_number')) is int and 1<=report['cycle_number']<=12
        and audit.get('games')==1024 and audit.get('raw_file_count')==32 and audit.get('transport_file_count')==32
        and audit.get('playing_strength_claim') is False and audit.get('review_authorized') is False,
        'Full frozen development audit required; smoke/partial outcomes cannot propose a policy.')
    record=audit.get('report');_sampling_require(isinstance(record,dict) and set(record)=={'path','bytes','sha256'}
        and type(record['bytes']) is int and record['bytes']>0 and _sampling_hash(record['sha256']),'Exact audited report identity required.')
    actual,raw=_sampling_file(record['path'])
    _sampling_require(actual==record,'Development report bytes changed since the frozen audit.')
    _sampling_require(_sampling_canonical(json.loads(raw))==_sampling_canonical(report),'Parsed report is not the audited exact file.')
    cfg=report.get('cfg');import hashlib
    _sampling_require(isinstance(cfg,dict) and report.get('config_signature')==hashlib.sha256(_sampling_canonical(cfg).encode()).hexdigest(),'Full saved config signature required.')
    checkpoints=report.get('checkpoint_records')
    _sampling_require(isinstance(checkpoints,dict) and set(checkpoints)=={'baseline','candidate'},'Exact baseline/candidate checkpoint identities required.')
    for value in checkpoints.values():
        _sampling_require(isinstance(value,dict) and set(value)=={'path','bytes','sha256'} and isinstance(value['path'],str)
            and type(value['bytes']) is int and value['bytes']>0 and _sampling_hash(value['sha256']),'Exact native checkpoint identity required.')
    analysis=report.get('analysis');_sampling_require(isinstance(analysis,dict) and analysis.get('playing_strength_claim') is False,'Audited per-style/family analysis required, without strength claim.')
    rows=analysis.get('per_style');families=analysis.get('per_family')
    _sampling_require(isinstance(rows,list) and [row.get('style_id') for row in rows]==list(DEVELOPMENT_STYLE_IDS),'Exact16 development style rows required.')
    _sampling_require(isinstance(families,list) and [row.get('family') for row in families]==[-1,0,1,2],'Exact legacy/new training-family summaries required.')
    for row in rows:
        _sampling_require(row.get('complete_pairs')==16,'Every development style must have all16 duplicate pairs.')
        for key in ('baseline_win_credit','candidate_win_credit'):
            value=row.get(key);_sampling_require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'Invalid per-style win credit.')
    for row in families:
        interval=row.get('nominal95_paired_cluster_interval');mean=row.get('mean_paired_difference')
        _sampling_require(row.get('style_count')==4 and type(mean) in (int,float) and math.isfinite(mean) and -1<=mean<=1
            and isinstance(interval,list) and len(interval)==2 and all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in interval)
            and interval[0]<=interval[1] and row.get('clear_harm') is (interval[1]<0),'Valid audited family difference/interval required.')
    return report,audit


def propose_sampling_policy(development_report,development_audit,*,prior_policy=None):
    """Deterministic bounded proposal from a separately recomputed frozen audit.

    Select the most negative new-family mean among upper95<0 families; ties use
    the lower family ID. Increase that one weight4->5. If it is already capped,
    or no family clears that gate, keep the exact prior policy. No reset/decay,
    heldout outcomes, new ranges or unbounded accumulation is implemented.
    """
    prior=continue_current_policy(prior_policy)
    report,audit=_sampling_valid_development(development_report,development_audit)
    candidates=[row for row in report['analysis']['per_family'] if row['family'] in (0,1,2) and row['nominal95_paired_cluster_interval'][1]<0]
    if not candidates:return prior
    selected=min(candidates,key=lambda row:(row['mean_paired_difference'],row['family']))['family']
    weights=list(prior['family_relative_weights'])
    if weights[selected]==5:return prior
    weights[selected]=5
    result=_sampling_copy(prior)
    result.update(mode='bounded_new_family_tilt',sampler_revision='bounded_family_choice_v1',training_only_override=True,
        family_relative_weights=weights,new_family_probabilities=[[x,sum(weights)] for x in weights],
        policy_revision=prior['policy_revision']+1,parent_policy_sha256=sampling_policy_sha256(prior),
        evidence=dict(report=_sampling_copy(audit['report']),audit_result_sha256=sampling_policy_sha256(audit),
            config_signature=report['config_signature'],cycle_number=report['cycle_number'],
            checkpoint_records=_sampling_copy(report['checkpoint_records']),selected_family=selected,
            per_style=_sampling_copy(report['analysis']['per_style']),per_family=_sampling_copy(report['analysis']['per_family']),
            interpretation='Training-only bounded practice allocation; no improved-strength or causal claim.'))
    return _sampling_validate_shape(result)


def validate_sampling_policy(policy,*,development_report=None,development_audit=None,prior_policy=None):
    """Contract passes VERIFIED prior native policy and actual recomputed audit.

    An unchanged transition returns the exact prior record. New weighted policy
    must match the deterministic proposal in full. Never infer that an absent
    prior record means uniform after a weighted checkpoint.
    """
    prior=continue_current_policy(prior_policy);candidate=_sampling_validate_shape(policy)
    if _sampling_canonical(candidate)==_sampling_canonical(prior):return candidate
    _sampling_require(development_report is not None and development_audit is not None,'Changed training sampler requires current audited development evidence.')
    expected=propose_sampling_policy(development_report,development_audit,prior_policy=prior)
    _sampling_require(_sampling_canonical(candidate)==_sampling_canonical(expected),'Sampling policy does not match the exact bounded evidence-derived proposal.')
    return candidate


def build_train_sampler(jax,jnp,base_sampler,validated_policy):
    """Return a pure jittable sampler; call only after host contract validation."""
    policy=_sampling_validate_shape(validated_policy)
    _sampling_require(callable(base_sampler),'Exact original base sampler callable required.')
    if policy['mode']=='identity':
        def identity_sampler(key,n):
            return base_sampler(key,n)
        return identity_sampler
    weights=tuple(policy['family_relative_weights'])
    def weighted_sampler(key,n):
        if not isinstance(n,int) or isinstance(n,bool) or n<0:raise ValueError('Static nonnegative Python integer size required.')
        mix_key,legacy_key,family_key,variant_key=jax.random.split(key,4)
        legacy=jax.random.randint(legacy_key,(n,),0,13,dtype=jnp.int32)
        probabilities=jnp.asarray(weights,jnp.float32)/jnp.float32(sum(weights))
        family=jax.random.categorical(family_key,jnp.log(probabilities),shape=(n,)).astype(jnp.int32)
        variant=jax.random.randint(variant_key,(n,),0,32,dtype=jnp.int32)
        new_style=13+family*64+variant
        return jnp.where(jax.random.bernoulli(mix_key,0.5,(n,)),legacy,new_style).astype(jnp.int32)
    return weighted_sampler
