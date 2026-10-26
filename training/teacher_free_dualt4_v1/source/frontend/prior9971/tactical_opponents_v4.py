"""Public-information tactical opponents; append after exact held formula-v3.

The 301 inherited valid styles delegate the captured dispatcher unchanged.
New tactics read one acting hand and public history only. No search, imports,
I/O, opponent hands, reconstructed hidden deals or future-state information.
"""

TACTICAL_OPPONENT_REVISION = 'spades-tactical-opponent-expansion-v4'
TACTICAL_PROFILE_KEY = 'opponent_training_profile'
TACTICAL_PROFILE_VALUE = 'current_quota_tactical_expansion_v4'
TACTICAL_NEW_TRAIN_FAMILIES = ('nil_cover_contract_defense', 'bag_pressure_exits', 'trump_resource_endplay')
TACTICAL_FRESH_HOLDOUT_FAMILIES = ('public_tempo_disruption', 'late_contract_thresholds')
TACTICAL_NEW_TRAIN_STYLE_IDS = tuple(13 + 64*f + v for f in (9,10,11) for v in range(32))
TACTICAL_TRAIN_STYLE_IDS = FORMULA_TRAIN_STYLE_IDS + TACTICAL_NEW_TRAIN_STYLE_IDS
TACTICAL_OLD_HOLDOUT_STYLE_IDS = FORMULA_HOLDOUT_STYLE_IDS
TACTICAL_FRESH_HOLDOUT_STYLE_IDS = tuple(13 + 64*f + v for f in (12,13) for v in range(32,64))
TACTICAL_CONFIRMATION_STYLE_IDS = TACTICAL_FRESH_HOLDOUT_STYLE_IDS
TACTICAL_HOLDOUT_STYLE_IDS = TACTICAL_OLD_HOLDOUT_STYLE_IDS + TACTICAL_FRESH_HOLDOUT_STYLE_IDS
TACTICAL_VALID_STYLE_IDS = TACTICAL_TRAIN_STYLE_IDS + TACTICAL_HOLDOUT_STYLE_IDS
TACTICAL_MAX_STYLE_ID = 908

_tactical_formula_action = diverse_heuristic_action
_tactical_formula_decode = decode_opponent_style_id
_tactical_formula_is_train = is_train_opponent_style_id
_tactical_formula_is_holdout = is_holdout_opponent_style_id
_tactical_formula_sampler = sample_train_opponent_style_ids

TACTICAL_OPPONENT_MANIFEST = {
    'revision': TACTICAL_OPPONENT_REVISION,
    'base_revision': FORMULA_OPPONENT_REVISION,
    'profile_key': TACTICAL_PROFILE_KEY, 'profile_value': TACTICAL_PROFILE_VALUE,
    'encoding': {'base':13, 'family_stride':64, 'maximum_id':908},
    'legacy_ids': list(range(13)),
    'inherited_train_style_ids': list(FORMULA_TRAIN_STYLE_IDS),
    'new_train_style_ids': list(TACTICAL_NEW_TRAIN_STYLE_IDS),
    'train_style_ids': list(TACTICAL_TRAIN_STYLE_IDS),
    'training_style_ids': list(TACTICAL_TRAIN_STYLE_IDS),
    'old_heldout_style_ids': list(TACTICAL_OLD_HOLDOUT_STYLE_IDS),
    'confirmation_style_ids': list(TACTICAL_FRESH_HOLDOUT_STYLE_IDS),
    'valid_style_ids': list(TACTICAL_VALID_STYLE_IDS),
    'old_holdout_style_ids': list(TACTICAL_OLD_HOLDOUT_STYLE_IDS),
    'fresh_holdout_style_ids': list(TACTICAL_FRESH_HOLDOUT_STYLE_IDS),
    'holdout_style_ids': list(TACTICAL_HOLDOUT_STYLE_IDS),
    'train_families': list(FORMULA_OPPONENT_MANIFEST['train_families']) + list(TACTICAL_NEW_TRAIN_FAMILIES),
    'new_train_families': list(TACTICAL_NEW_TRAIN_FAMILIES),
    'train_family_indices': [0,1,2,5,6,7,8,9,10,11],
    'new_train_family_indices': [9,10,11],
    'old_holdout_families': list(DIVERSITY_HOLDOUT_FAMILIES),
    'fresh_holdout_families': list(TACTICAL_FRESH_HOLDOUT_FAMILIES),
    'holdout_families': list(DIVERSITY_HOLDOUT_FAMILIES) + list(TACTICAL_FRESH_HOLDOUT_FAMILIES),
    'old_holdout_family_indices': [3,4], 'fresh_holdout_family_indices':[12,13],
    'holdout_family_indices': [3,4,12,13],
    'train_variants':list(range(32)), 'holdout_variants':list(range(32,64)),
    'train_style_count':333, 'inherited_train_style_count':237, 'new_train_style_count':96,
    'old_holdout_style_count':64, 'fresh_holdout_style_count':64,
    'holdout_style_count':128, 'valid_style_count':461,
    'scripted_sampling': {
        'group_order':['legacy','old_train','expansion','formula','tactical'],
        'group_integer_weights':[39,27,54,36,52], 'integer_denominator':208,
        'integer_weights':{'legacy':39,'old_train':27,'expansion_v2':54,'formula':36,'tactical':52}, 'denominator':208,
        'legacy_probability':39/208, 'old_train_probability':27/208,
        'expansion_probability':54/208, 'formula_probability':36/208,
        'tactical_probability':52/208, 'legacy_uniform_count':13,
        'old_uniform_family_count':3, 'expansion_uniform_family_count':3,
        'formula_uniform_family_count':1, 'tactical_uniform_family_count':3,
        'uniform_variants_per_family':32, 'assignment_scope':'match',
        'preservation':'Legacy13 retain3/16 of scripted assignments. Prior224 nonlegacy styles retain their relative group weights, collectively scaled9/13. New96 receive1/4 of scripted assignments.',
    },
    'outer_sampling':'Actual saved source selfplay/snapshot/scripted assignment probabilities remain unchanged and positive; verified8351 values are .5/.25/.25. New tactical group is .0625 of all assignments for that source.',
    'tactics': {
        'nil_cover_contract_defense':'Cover a live partner nil with the cheapest provisional winner; let a winning opponent nil take; press public near-complete opponent contracts when own contract is safe.',
        'bag_pressure_exits':'After the own team contract is made, public bag pressure switches to losing high-card exits; before completion choose economical winners. Exit safety uses only public voids.',
        'trump_resource_endplay':'Use short side exits into public opponent voids while partner can follow to draw opposing trumps; cash backed master trumps to strip outstanding resources, then preserve master trump rescue and prefer exhausted late exits.',
        'public_tempo_disruption':'Fresh heldout: change lead tempo toward public opponent voids when partner can ruff, otherwise attack long suits; following combines second-hand low and fourth-hand minimal capture.',
        'late_contract_thresholds':'Fresh heldout: public remaining-trick versus team-need thresholds switch between urgent capture and low-risk contract-satisfied ducking, with asymmetric partner-nil rescue and enemy-contract denial.',
    },
    'parameter':'q=(local_variant+.5)/32; local_variant=variant for train9..11 and variant-32 for fresh12..13. Each family has32 unique coefficient/threshold vectors, including public blind risk, pass danger, nil caution and tactical exchanges.',
    'information':'Acting hand only via _tactical_acting_hand; public table/players, played ownership, voids, bids, tricks, scores, bags and turn. Blind reads public scores/eligible only and no legal mask.',
    'preservation':'All301 inherited valid styles, including13 legacy,237 train and64 development-visible old heldouts, call the captured formula-v3 action with identical state/key/style. All original functions except the five public dispatch/membership/sampler entry points and all inherited constants stay identical. No current-match resampling occurs in this module.',
    'fresh_confirmation':'Families12/13 variants32..63 only; separate contextual handlers, never sampled for training. Seeds2026102191/2192 predeclared in fresh_confirmation_protocol_v4.json;2193/2194 sealed reserve.',
    'invalid_action':-1,
    'search':'none; bounded52-card public rank counts only',
    'strength':'unmeasured; synthetic legality/privacy/causality evidence is not playing-strength evidence',
}


def decode_opponent_style_id(style):
    # Host integer scalars must be checked before JAX's optional int64-to-int32
    # narrowing, so a huge NumPy/Python ID cannot wrap into an admitted style.
    host_scalar = (not hasattr(style, 'aval')) and (isinstance(style, int) or
        (getattr(getattr(style, 'dtype', None), 'kind', '') in ('i','u') and getattr(style, 'size', 1) == 1))
    if host_scalar and (int(style) < 0 or int(style) > TACTICAL_MAX_STYLE_ID):
        return jnp.int32(-2), jnp.int32(-1), jnp.bool_(False)
    host_range = True
    if not hasattr(style, 'aval') and getattr(getattr(style, 'dtype', None), 'kind', '') in ('i','u'):
        host_range = jnp.asarray((style >= 0) & (style <= TACTICAL_MAX_STYLE_ID))
    raw = jnp.asarray(style)
    integer = jnp.issubdtype(raw.dtype, jnp.integer)
    in_range = (raw >= 0) & (raw <= TACTICAL_MAX_STYLE_ID)
    family, variant, old_valid = _tactical_formula_decode(style)
    train_extra = (family >= 9) & (family <= 11) & (variant < 32)
    fresh = ((family == 12) | (family == 13)) & (variant >= 32)
    return family, variant, integer & in_range & host_range & (old_valid | train_extra | fresh)


def is_train_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    old = _tactical_formula_is_train(style)
    return valid & (old | ((family >= 9) & (family <= 11) & (variant < 32)))


def is_holdout_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    return valid & (_tactical_formula_is_holdout(style) | (((family == 12) | (family == 13)) & (variant >= 32)))


def sample_train_opponent_style_ids(key, n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('n must be a nonnegative static integer')
    group_key, legacy_key, family_key, variant_key = jax.random.split(key, 4)
    bucket = jax.random.randint(group_key, (n,), 0, 208)
    legacy = jax.random.randint(legacy_key, (n,), 0, 13)
    family_draw = jax.random.randint(family_key, (n,), 0, 3)
    variant = jax.random.randint(variant_key, (n,), 0, 32)
    family = jnp.where(bucket < 66, family_draw,
        jnp.where(bucket < 120, 5+family_draw, jnp.where(bucket < 156, 8, 9+family_draw)))
    return jnp.where(bucket < 39, legacy, 13+64*family+variant).astype(jnp.int32)


def tactical_style_parameters(family, variant):
    local = jnp.where(family >= 12, variant-32, variant)
    q = (local.astype(jnp.float32)+.5)/32.
    return dict(q=q, cover=5.+3.*q, defense=2.+3.*q,
        bag_exit=2.+4.*q, bag_threshold=3.+5.*q,
        reserve=1.+3.*q, tempo=1.5+3.*q,
        urgent=3.+4.*q, blind_risk=.015+.12*q,
        pass_weight=.5+q, nil_threshold=.25+.7*q)


def _tactical_acting_hand(state):
    return state.hands[actor(state)]


def _tactical_public_features(state):
    hand = _tactical_acting_hand(state)
    me = actor(state)
    partner = (me+2)%4
    ids = jnp.arange(52, dtype=jnp.int32)
    ranks, suits = ids%13, ids//13
    lengths = hand.reshape(4,13).sum(axis=1)
    unseen = ((state.played_by < 0) & ~hand).reshape(4,13)
    higher_unseen = (unseen[:,None,:] & (jnp.arange(13)[None,None,:] > jnp.arange(13)[None,:,None])).sum(axis=2).reshape(52)
    own = hand.reshape(4,13)
    higher_own = (own[:,None,:] & (jnp.arange(13)[None,None,:] > jnp.arange(13)[None,:,None])).sum(axis=2).reshape(52)
    lower_own = (own[:,None,:] & (jnp.arange(13)[None,None,:] < jnp.arange(13)[None,:,None])).sum(axis=2).reshape(52)
    leading = state.trick_pos == 0
    led = jnp.maximum(state.table[0],0)//13
    table = jnp.maximum(state.table,0)
    ts = table//13
    table_strength = jnp.where(ts==3,26+table%13,jnp.where(ts==led,13+table%13,-1))
    table_strength = jnp.where(state.table>=0,table_strength,-100)
    winner_index = jnp.argmax(table_strength)
    winner = jnp.where(leading,-1,state.table_players[winner_index])
    candidate_strength = jnp.where(suits==3,26+ranks,jnp.where(suits==led,13+ranks,-1))
    wins = leading | (candidate_strength > jnp.max(table_strength))
    seats = jnp.arange(4)
    our_team = seats%2 == me%2
    normal = jnp.maximum(state.bids,0)
    normal_tricks = jnp.where(state.bids>0,state.tricks,0)
    failed = jnp.any(our_team & (state.bids<=0) & (state.tricks>0))
    enemy_failed = jnp.any(~our_team & (state.bids<=0) & (state.tricks>0))
    live_nil = (state.bids<=0) & (state.tricks==0)
    our_need = jnp.maximum(jnp.sum(jnp.where(our_team,normal-normal_tricks,0)),0)
    their_need = jnp.maximum(jnp.sum(jnp.where(~our_team,normal-normal_tricks,0)),0)
    remaining = jnp.maximum(13-jnp.maximum(state.turn-12,0)//4,1)
    return dict(hand=hand, ranks=ranks, rank=ranks.astype(jnp.float32)/12., suits=suits,
        lengths=lengths, unseen_count=unseen.sum(axis=1), higher_unseen=higher_unseen,
        higher_own=higher_own, lower_own=lower_own, master=higher_unseen==0,
        leading=leading, trick_pos=state.trick_pos, wins=wins,
        partner_winning=winner==partner,
        partner_nil_winning=(winner==partner)&live_nil[partner]&~failed,
        enemy_nil_winning=jnp.any(~our_team & live_nil & (seats==winner))&~enemy_failed,
        own_nil=live_nil[me]&~failed, partner_nil=live_nil[partner]&~failed,
        our_need=jnp.where(failed,0,our_need), their_need=jnp.where(enemy_failed,0,their_need),
        remaining=remaining, bag_count=state.bags[me%2],
        score_deficit=state.scores[1-me%2]-state.scores[me%2],
        opponent_void_count=jnp.sum(state.voids & (~our_team)[:,None],axis=0),
        partner_void=state.voids[partner],
        enemy_failed=enemy_failed, our_failed=failed)


def _tactical_nil_cover_defense(f,p):
    trump = f['suits']==3
    need = f['our_need']>0
    score = (2.4+1.2*p['q'])*f['wins']*need-.7*f['rank']-.45*trump
    # A partner nil needs an actual overtake, not generic partner-safe ducking.
    score += f['partner_nil_winning']*f['wins']*p['cover']
    score -= f['enemy_nil_winning']*f['wins']*p['cover']
    deny = (f['our_need']==0)&(f['their_need']>0)&(f['their_need']<=2)&(f['remaining']<=5)&~f['enemy_failed']
    score += deny*f['wins']*p['defense']
    safe_cover_lead = f['leading']&f['partner_nil']&~trump&f['master']
    score += safe_cover_lead*(2.+p['q'])
    score -= ~f['leading']*f['partner_winning']*~f['partner_nil']*f['wins']*3.5
    return score


def _tactical_bag_pressure_exits(f,p):
    trump = f['suits']==3
    made = f['our_need']==0
    pressure = made & (f['bag_count'].astype(jnp.float32)>=p['bag_threshold']) & ~f['our_failed']
    score = (2.5+1.5*p['q'])*f['wins']*(~made)-.65*f['rank']-.45*trump
    # High losers dispose of future bag-taking resources; on lead choose a
    # low card in a suit without public enemy voids, avoiding master/trump.
    score += pressure*(~f['leading'])*(~f['wins'])*f['rank']*p['bag_exit']
    score -= pressure*f['wins']*p['bag_exit']
    safe_exit = (~trump)&(f['opponent_void_count'][f['suits']]==0)&~f['master']
    score += pressure*f['leading']*safe_exit*(1.+p['q'])
    score -= pressure*f['leading']*(f['master']+trump.astype(jnp.int32))*(1.+p['q'])
    score -= ~f['leading']*f['partner_winning']*f['wins']*3.
    return score


def _tactical_trump_resource_endplay(f,p):
    trump = f['suits']==3
    need = f['our_need']>0
    score = (2.7+p['q'])*f['wins']*need-.55*f['rank']-.35*trump
    # A public void can force an opponent to spend a trump on our short
    # side-suit exit while partner still follows. No hidden trump owner is
    # inferred: this is a resource-pressure preference, not a forced result.
    force_ruff = f['leading']&need&(f['unseen_count'][3]>0)&(~trump)
    force_ruff &= (f['lengths'][f['suits']]<=2)&(f['opponent_void_count'][f['suits']]>=1)&(~f['partner_void'][f['suits']])
    score += force_ruff*(1.4+3.*p['q'])*(1.+.3*f['rank'])
    backed_master = trump&f['master']&(f['lower_own']>=1)
    draw = f['leading']&need&(f['unseen_count'][3]>0)&(f['remaining']>4)
    score += draw*backed_master*(1.2+2.*p['q'])
    late = f['remaining']<=4
    sufficient = f['lengths'][3]>=f['our_need']
    reserve = f['leading']&late&need&sufficient
    exhausted_side = (~trump)&(f['unseen_count'][f['suits']]==0)
    score += reserve*exhausted_side*p['reserve']
    score -= reserve*trump*f['master']*p['reserve']
    urgent = f['our_need']>=f['remaining']
    score += urgent*f['wins']*(2.+p['q'])
    score -= ~f['leading']*f['partner_winning']*f['wins']*3.
    return score


def _tactical_fresh_public_tempo(f,p):
    """Heldout12: suit tempo, partner ruff opportunity, positional economy."""
    trump = f['suits']==3
    side = ~trump
    need = f['our_need']>0
    score = (1.6+2.*p['q'])*f['wins']*need-.35*f['rank']-.8*trump
    partner_ruff = f['partner_void'][f['suits']] & (f['opponent_void_count'][f['suits']]>=1)
    long_attack = (f['lengths'][f['suits']]>=4)&(f['opponent_void_count'][f['suits']]==0)
    score += f['leading']*side*(partner_ruff*(1.2+p['tempo'])+long_attack*(.6+1.3*p['q']))
    score += (f['trick_pos']==1)*(~f['wins'])*(1.-f['rank'])*(.7+p['q'])
    score += (f['trick_pos']==3)*(~f['partner_winning'])*f['wins']*(1.8+2.2*p['q'])
    score -= ~f['leading']*f['partner_winning']*f['wins']*(2.+p['q'])
    return score


def _tactical_fresh_contract_thresholds(f,p):
    """Heldout13: finite public trick budget, defensive threshold switching."""
    trump = f['suits']==3
    urgent = (f['our_need']>0)&(f['our_need']+(.5+2.*p['q'])>=f['remaining'])
    made = f['our_need']==0
    deny = (f['their_need']>0)&(f['their_need']>=f['remaining']-(.25+1.5*p['q']))
    score = -.9*f['rank']-.2*trump+(1.5+p['q'])*f['wins']*(f['our_need']>0)
    score += urgent*f['wins']*p['urgent']
    score += made*deny*f['wins']*(1.4+2.7*p['q'])
    score += made*(~deny)*(~f['wins'])*f['rank']*(1.7+1.1*p['q'])
    score -= made*(~deny)*f['wins']*(2.8+p['q'])
    score += f['leading']*urgent*f['master']*(1.4+2.*p['q'])
    score += f['partner_nil_winning']*f['wins']*(6.5+1.7*p['q'])
    score -= f['enemy_nil_winning']*f['wins']*(5.5+1.3*p['q'])
    return score


def _tactical_blind_action(state,key,family,variant):
    """Public pre-deal risk only; never inspect any hand or legal mask."""
    me = actor(state)
    p = tactical_style_parameters(family,variant)
    deficit = state.scores[1-me%2]-state.scores[me%2]
    chance = p['blind_risk']*jnp.where(deficit>=150,1.8,.35)
    # Separate fresh policies also have distinct public blind risk scales.
    scale = jnp.where(family==12,.7,jnp.where(family==13,1.25,1.))
    choose = state.eligible[me%2] & (jax.random.uniform(key)<chance*scale)
    return jnp.where(choose,67,66).astype(jnp.int32)


def _tactical_pass_action(state,key,family,variant,legal):
    hand = _tactical_acting_hand(state)
    me = actor(state)
    ids = jnp.arange(52,dtype=jnp.int32)
    ranks,suits = ids%13,ids//13
    lengths = hand.reshape(4,13).sum(axis=1)
    p = tactical_style_parameters(family,variant)
    danger = ranks.astype(jnp.float32)/12.+.75*(suits==3)+.3*(lengths[suits]<=2)
    exporting = state.bids[me]==-1
    preserve_cover = (family==9)&(state.bids[(me+2)%4]==-1)
    score = jnp.where(exporting,danger*p['pass_weight'],-danger)
    score -= preserve_cover*(ranks>=10)*.8
    score += (family==10)*(~exporting)*(lengths[suits]<=2)*.4
    score += (family==11)*(~exporting)*(suits!=3)*.25
    score += (family==12)*(~exporting)*(lengths[suits]>=4)*.45
    score -= (family==13)*(~exporting)*(ranks>=9)*(.3+p['q'])
    score += .01*jax.random.uniform(key,(52,))
    chosen = jnp.argmax(jnp.where(legal[:52],score,-1.e9))
    return jnp.where(jnp.any(legal[:52]),chosen,66).astype(jnp.int32)


def _tactical_bid_action(state,key,family,variant,legal):
    hand = _tactical_acting_hand(state)
    me = actor(state)
    estimate,nil_risk = estimate_bid_and_nil_risk(hand)
    p = tactical_style_parameters(family,variant)
    partner_bid = state.bids[(me+2)%4]
    shift = jnp.where(family==10,-.15,jnp.where(family==11,.2,jnp.where(family==12,-.3,jnp.where(family==13,.35,0.))))
    bid = jnp.clip(jnp.floor(estimate+.2+.65*p['q']+shift),1,13).astype(jnp.int32)
    bid = jnp.minimum(bid,jnp.maximum(1,13-jnp.maximum(partner_bid,0)))
    threshold = p['nil_threshold']*jnp.where(partner_bid<=0,.2,1.)
    chosen = jnp.where(nil_risk<threshold,52,52+bid)
    return jnp.where(legal[66],66,chosen).astype(jnp.int32)


def _tactical_play_action(state,key,family,variant,legal):
    f = _tactical_public_features(state)
    p = tactical_style_parameters(family,variant)
    # Families12/13 have their own handlers, rather than aliasing train tactics.
    score = jax.lax.switch(family-9,(_tactical_nil_cover_defense,
        _tactical_bag_pressure_exits,_tactical_trump_resource_endplay,
        _tactical_fresh_public_tempo,_tactical_fresh_contract_thresholds),f,p)
    score += f['partner_nil_winning']*f['wins']*(4.+p['q'])
    score -= f['enemy_nil_winning']*f['wins']*(4.+p['q'])
    danger = f['rank']+.7*(f['suits']==3)+.25*f['master']
    nil_follow = -10.*f['wins']+(1.1+.4*p['q'])*danger*(~f['wins'])
    nil_lead = -3.*f['rank']-2.*(f['suits']==3)-.5*f['master']
    score = jnp.where(f['own_nil'],jnp.where(f['leading'],nil_lead,nil_follow),score)
    score += .008*jax.random.uniform(key,(52,))
    return jnp.argmax(jnp.where(legal[:52],score,-1.e9)).astype(jnp.int32)


def _tactical_nonblind_action(state,key,family,variant):
    legal = legal_mask(state)
    chosen = jax.lax.switch(jnp.maximum(phase(state)-1,0),
        (_tactical_pass_action,_tactical_bid_action,_tactical_play_action),
        state,key,family,variant,legal)
    return jnp.where(legal[chosen],chosen,jnp.argmax(legal)).astype(jnp.int32)


def diverse_heuristic_action(state,key,style):
    family,variant,valid = decode_opponent_style_id(style)
    # Canonical int32 values also make traced, unselected inherited branches
    # safe for an invalid huge host scalar. Every admitted old ID is identical.
    canonical_style = jnp.where(family < 0, variant, 13+64*family+variant).astype(jnp.int32)
    def admitted(_):
        def tactical(_):
            return jax.lax.cond(phase(state)==0,_tactical_blind_action,
                _tactical_nonblind_action,state,key,family,variant)
        return jax.lax.cond(family>=9,tactical,
            lambda _:_tactical_formula_action(state,key,canonical_style),operand=None)
    return jax.lax.cond(valid,admitted,lambda _:jnp.int32(-1),operand=None).astype(jnp.int32)
