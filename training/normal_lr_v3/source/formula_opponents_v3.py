"""Formula training opponents; append after frozen cell8 and held expansion-v2.

No imports, I/O, hidden hands, simulated deals or lookahead. Probability values
are explicit public-information approximations, not calibrated predictions.
All inherited IDs delegate the captured v2 dispatcher without feature changes.
"""

FORMULA_OPPONENT_REVISION = 'spades-formula-opponent-v3'
FORMULA_PROFILE_KEY = 'opponent_training_profile'
FORMULA_PROFILE_VALUE = 'normal_lr_formula_longrun_v3'
FORMULA_FAMILY_INDEX = 8
FORMULA_NEW_TRAIN_STYLE_IDS = tuple(range(525, 557))
FORMULA_TRAIN_STYLE_IDS = EXPANSION_TRAIN_STYLE_IDS + FORMULA_NEW_TRAIN_STYLE_IDS
FORMULA_HOLDOUT_STYLE_IDS = EXPANSION_HOLDOUT_STYLE_IDS
FORMULA_MAX_STYLE_ID = 556

_formula_v2_action = diverse_heuristic_action
_formula_v2_decode = decode_opponent_style_id
_formula_v2_is_train = is_train_opponent_style_id
_formula_v2_is_holdout = is_holdout_opponent_style_id
_formula_v2_sampler = sample_train_opponent_style_ids

FORMULA_BINOMIAL_COEFFICIENTS = (
    (1,0,0,0,0,0,0,0,0,0,0,0,0),
    (1,1,0,0,0,0,0,0,0,0,0,0,0),
    (1,2,1,0,0,0,0,0,0,0,0,0,0),
    (1,3,3,1,0,0,0,0,0,0,0,0,0),
    (1,4,6,4,1,0,0,0,0,0,0,0,0),
    (1,5,10,10,5,1,0,0,0,0,0,0,0),
    (1,6,15,20,15,6,1,0,0,0,0,0,0),
    (1,7,21,35,35,21,7,1,0,0,0,0,0),
    (1,8,28,56,70,56,28,8,1,0,0,0,0),
    (1,9,36,84,126,126,84,36,9,1,0,0,0),
    (1,10,45,120,210,252,210,120,45,10,1,0,0),
    (1,11,55,165,330,462,462,330,165,55,11,1,0),
    (1,12,66,220,495,792,924,792,495,220,66,12,1),
)

FORMULA_OPPONENT_MANIFEST = {
    'revision': FORMULA_OPPONENT_REVISION,
    'base_revision': EXPANSION_OPPONENT_REVISION,
    'profile_key': FORMULA_PROFILE_KEY, 'profile_value': FORMULA_PROFILE_VALUE,
    'encoding': {'base': 13, 'family_stride': 64, 'maximum_id': 556},
    'legacy_ids': list(range(13)),
    'inherited_train_style_ids': list(EXPANSION_TRAIN_STYLE_IDS),
    'new_train_style_ids': list(FORMULA_NEW_TRAIN_STYLE_IDS),
    'train_style_ids': list(FORMULA_TRAIN_STYLE_IDS),
    'holdout_style_ids': list(FORMULA_HOLDOUT_STYLE_IDS),
    'train_families': list(DIVERSITY_TRAIN_FAMILIES) + list(EXPANSION_TRAIN_FAMILIES) + ['formula_expected_contract'],
    'new_train_families': ['formula_expected_contract'],
    'train_family_indices': [0,1,2,5,6,7,8],
    'new_train_family_indices': [8],
    'holdout_families': list(DIVERSITY_HOLDOUT_FAMILIES),
    'holdout_family_indices': [3,4],
    'train_variants': list(range(32)), 'holdout_variants': list(range(32,64)),
    'train_style_count': 237, 'new_train_style_count': 32,
    'holdout_style_count': 64, 'valid_style_count': 301,
    'scripted_sampling': {'legacy_probability': 0.1875,
        'old_train_probability': 0.1875, 'expansion_probability': 0.375,
        'formula_probability': 0.25, 'legacy_uniform_count': 13,
        'old_uniform_family_count': 3, 'expansion_uniform_family_count': 3,
        'formula_uniform_family_count': 1, 'uniform_variants_per_family': 32,
        'assignment_scope': 'match'},
    'outer_sampling': 'Actual saved source selfplay/snapshot/scripted probabilities are preserved and must all be positive.',
    'formula': {
        'no_danger_probability': 'H(U,D,n)=product(i=0..n-1)((U-D-i)/(U-i)); zero if n>U-D.',
        'seat_beating_probability': '1-H(U,higher,n)*(1-H(U,led_suit,n)*(1-H(U,trumps,n))); public known voids override terms; trump candidates omit ruff term.',
        'team_trick_probability': 'q(c)=own_would_win*product(future_opponents_no_beat)+(1-own_would_win)*partner_winning*product(future_opponents_no_beat_current_card).',
        'future_probability': 'p(c)=clip((sum(own_hand_cash_probabilities)-cash(c)+partner_credit*partner_remaining_bid)/max(R-1,1),.02,.98).',
        'contract_probability': 'Pmake(c)=q(c)*BinTail(R-1,need-1,p(c))+(1-q(c))*BinTail(R-1,need,p(c)).',
        'score': 'urgency*10*B*(2*Pmake-1)-risk_aversion*20*B*sqrt(Pmake*(1-Pmake))-bag_weight*expected_extra+denial_weight*10*Bopp*(1-Popp_make)-resource_weight*cash(c)+public_nil_protection.',
        'variables': {'U':'all unseen cards excluding own hand and public played cards',
            'D':'public count of dangerous unseen ranks', 'n':'seat cards remaining from public play counts',
            'R':'tricks remaining', 'need':'normal team bid still required',
            'B':'normal team bid total', 'cash(c)':'public survival estimate if c leads against both opponents',
            'BinTail':'bounded exact binomial tail for zero through twelve future tricks'},
        'assumptions': [
            'Unknown cards are uniformly allocated from the public unseen pool; known voids override suit hazards.',
            'Higher-card possession and void-with-trump events are combined as independent approximations.',
            'Different future opponent holdings and future trick outcomes are treated as independent.',
            'Future partner rescue is omitted from current-trick probability; public partner bid supplies discounted future credit.',
            'Own cash probabilities are an approximate future resource count; no calibration or playing-strength claim is made.',
        ],
    },
    'parameter': 'q_profile=(variant+0.5)/32; each profile changes variance-risk, bags, partner credit, resource cost and public blind risk.',
    'new_parameter_bands': {'q_profile':[0.015625,0.984375],
        'risk_aversion':[0.0546875,0.3453125], 'partner_credit':[0.453125,0.646875],
        'bag_weight':[0.53125,2.46875], 'denial_weight':[0.1546875,0.4453125],
        'resource_weight':[0.515625,1.484375]},
    'preservation': 'All 269 inherited valid IDs delegate the captured expansion-v2 callable; all original functions, IDs, parameters and heldout behavior remain unchanged.',
    'invalid_action': -1,
    'information': 'Acting hand plus public played ownership, table, voids, bids, tricks, scores and bags only; blind reads no hands or legal mask.',
    'search': 'none; fixed52-card rank counts, bounded13-factor hypergeometric products and bounded13-term binomial sums',
    'strength': 'unmeasured; mathematical approximation is neither an optimal-play nor a calibrated-probability claim',
}


def decode_opponent_style_id(style):
    family, variant, valid = _formula_v2_decode(style)
    raw = jnp.asarray(style)
    extra = jnp.issubdtype(raw.dtype, jnp.integer) & (family == 8) & (variant < 32)
    return family, variant, valid | extra


def is_train_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    return _formula_v2_is_train(style) | (valid & (family == 8) & (variant < 32))


def is_holdout_opponent_style_id(style):
    return _formula_v2_is_holdout(style)


def sample_train_opponent_style_ids(key, n):
    """25% formula; remaining75% preserves the exact v2 group ratios."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('Static nonnegative Python integer population size required.')
    mix_key, legacy_key, family_key, variant_key = jax.random.split(key, 4)
    group = jax.random.uniform(mix_key, (n,))
    legacy = jax.random.randint(legacy_key, (n,), 0, 13, dtype=jnp.int32)
    family = jax.random.randint(family_key, (n,), 0, 3, dtype=jnp.int32)
    variant = jax.random.randint(variant_key, (n,), 0, 32, dtype=jnp.int32)
    family += jnp.where(group >= .375, 5, 0)
    family = jnp.where(group >= .75, 8, family)
    encoded = 13 + 64 * family + variant
    return jnp.where(group < .1875, legacy, encoded).astype(jnp.int32)


def formula_style_parameters(variant):
    q = (variant.astype(jnp.float32) + .5) / 32.
    return dict(q=q, risk_aversion=.05+.3*q, partner_credit=.45+.2*q,
        bag_weight=.5+2.*q, denial_weight=.15+.3*q, resource_weight=.5+q)


def _formula_acting_hand(state):
    return state.hands[actor(state)]


def _formula_hypergeom_none(unseen_total, dangerous, draws):
    """Exact zero-success hypergeometric probability under uniform allocation."""
    i = jnp.arange(13, dtype=jnp.float32)
    total = jnp.asarray(unseen_total, jnp.float32)
    danger = jnp.asarray(dangerous, jnp.float32)
    n = jnp.asarray(draws, jnp.float32)
    ratio = jnp.clip((total[...,None]-danger[...,None]-i) /
                    jnp.maximum(total[...,None]-i, 1.), 0., 1.)
    probability = jnp.prod(jnp.where(i < n[...,None], ratio, 1.), axis=-1)
    return jnp.where(n > total-danger, 0., probability)


def _formula_candidate_survival(state, hand, cards, lead_suits, future_seats):
    """52-card/seat probability table from public unseen counts; no hand read."""
    unseen = ((state.played_by < 0) & ~hand).reshape(4,13)
    unseen_count = unseen.sum(axis=1)
    higher = jnp.flip(jnp.cumsum(jnp.flip(unseen,axis=1),axis=1),axis=1)-unseen
    suits, ranks = cards//13, cards%13
    seats = jnp.arange(4)
    remaining = jnp.clip(13-jnp.sum(state.played_by[:,None]==seats[None,:],axis=0),0,13)
    total = unseen_count.sum().astype(jnp.float32)
    draws = remaining[None,:]
    high_count = higher[suits,ranks][:,None]
    high_count = jnp.where(state.voids[:,suits].T,0,high_count)
    no_higher = _formula_hypergeom_none(total,high_count,draws)
    void_led = _formula_hypergeom_none(total,unseen_count[lead_suits][:,None],draws)
    void_led = jnp.where(state.voids[:,lead_suits].T,1.,void_led)
    has_trump = 1.-_formula_hypergeom_none(total,unseen_count[3],draws)
    has_trump = jnp.where(state.voids[:,3][None,:],0.,has_trump)
    ruff = void_led*has_trump*(suits[:,None]!=3)
    no_beat = no_higher*(1.-ruff)
    return jnp.prod(jnp.where(future_seats[None,:],no_beat,1.),axis=1)


def _formula_binomial_tail(n, need, probability):
    n = jnp.clip(n,0,12).astype(jnp.int32)
    p = jnp.clip(probability,0.,1.)
    successes = jnp.arange(13,dtype=jnp.int32)
    coefficients = jnp.asarray(FORMULA_BINOMIAL_COEFFICIENTS,jnp.float32)[n]
    mass = coefficients*p[...,None]**successes*(1.-p[...,None])**jnp.maximum(n-successes,0)
    tail = jnp.sum(jnp.where((successes>=need)&(successes<=n),mass,0.),axis=-1)
    return jnp.where(need<=0,1.,jnp.where(need>n,0.,tail))


def _formula_public_features(state):
    me = actor(state)
    partner = (me+2)%4
    seats = jnp.arange(4)
    own_team = seats%2 == me%2
    hand = _formula_acting_hand(state)
    ids = jnp.arange(52,dtype=jnp.int32)
    suits, ranks = ids//13, ids%13
    leading = state.trick_pos==0
    led = jnp.maximum(state.table[0],0)//13
    table = jnp.maximum(state.table,0)
    table_strength = jnp.where(table//13==3,26+table%13,
        jnp.where(table//13==led,13+table%13,-1))
    table_strength = jnp.where(state.table>=0,table_strength,-100)
    best = jnp.argmax(table_strength)
    current = table[best]
    winner = jnp.where(leading,-1,state.table_players[best])
    candidate_strength = jnp.where(suits==3,26+ranks,jnp.where(suits==led,13+ranks,-1))
    wins = leading | (candidate_strength>jnp.max(table_strength))
    relative = (seats-me)%4
    future = (relative>0)&(relative<=3-state.trick_pos)
    future_opponents = future&~own_team
    cards = jnp.concatenate((ids,current[None]))
    leads = jnp.concatenate((jnp.where(leading,suits,led),led[None]))
    survival = _formula_candidate_survival(state,hand,cards,leads,future_opponents)
    own_probability = wins.astype(jnp.float32)*survival[:52]
    team_probability = own_probability + (~wins)*(winner==partner)*survival[52]
    own_all_seats = wins.astype(jnp.float32)*_formula_candidate_survival(state,hand,ids,jnp.where(leading,suits,led),future)
    cash = _formula_candidate_survival(state,hand,ids,suits,~own_team)
    normal = jnp.maximum(state.bids,0)
    normal_tricks = jnp.where(state.bids>0,state.tricks,0)
    own_bid = jnp.sum(jnp.where(own_team,normal,0))
    other_bid = jnp.sum(jnp.where(~own_team,normal,0))
    our_need = jnp.maximum(jnp.sum(jnp.where(own_team,normal-normal_tricks,0)),0)
    their_need = jnp.maximum(jnp.sum(jnp.where(~own_team,normal-normal_tricks,0)),0)
    remaining = jnp.maximum(13-jnp.maximum(state.turn-12,0)//4,1)
    failed = jnp.any(own_team&(state.bids<=0)&(state.tricks>0)) | (our_need>remaining)
    other_failed = jnp.any(~own_team&(state.bids<=0)&(state.tricks>0)) | (their_need>remaining)
    live_nil = (state.bids<=0)&(state.tricks==0)
    return dict(team_win=jnp.clip(team_probability,0.,1.),own_win=own_probability,
        own_win_all_seats=own_all_seats,cash=cash,hand_cash=jnp.sum(hand*cash),
        partner_need=jnp.maximum(normal[partner]-normal_tricks[partner],0),
        own_bid=own_bid,other_bid=other_bid,our_need=our_need,their_need=their_need,
        remaining=remaining,own_failed=failed,other_failed=other_failed,
        own_bags=state.bags[me%2],score_deficit=state.scores[1-me%2]-state.scores[me%2],
        own_nil=live_nil[me]&~failed,partner_nil=live_nil[partner]&~failed,
        opponent_nil_winning=jnp.any(~own_team&live_nil&(seats==winner))&~other_failed,
        partner_winning=winner==partner,wins=wins,leading=leading,
        rank=ranks.astype(jnp.float32)/12.,trump=suits==3)


def _formula_card_scores(f,p):
    """Expected contract utility, with probability-dependent resource spending."""
    future_n = jnp.maximum(f['remaining']-1,0)
    future_probability = jnp.clip((f['hand_cash']-f['cash']+
        p['partner_credit']*f['partner_need'])/jnp.maximum(future_n,1),.02,.98)
    take_tail = _formula_binomial_tail(future_n,f['our_need']-1,future_probability)
    lose_tail = _formula_binomial_tail(future_n,f['our_need'],future_probability)
    make = f['team_win']*take_tail+(1.-f['team_win'])*lose_tail
    make = jnp.where(f['own_failed'],0.,make)
    urgency = 1.+.5*p['q']*jnp.clip(f['score_deficit']/500.,-.5,1.)
    expected_contract = 10.*f['own_bid']*(2.*make-1.)
    contract_risk = 20.*f['own_bid']*jnp.sqrt(jnp.maximum(make*(1.-make),0.))
    expected_extra = jnp.maximum(f['team_win']+future_n*future_probability-f['our_need'],0.)
    bags = p['bag_weight']*(1.+f['own_bags']/5.)*expected_extra
    opponent_probability = 1.-future_probability
    other_take = _formula_binomial_tail(future_n,f['their_need']-1,opponent_probability)
    other_lose = _formula_binomial_tail(future_n,f['their_need'],opponent_probability)
    other_make = (1.-f['team_win'])*other_take+f['team_win']*other_lose
    denial = jnp.where(f['other_failed'],0.,p['denial_weight']*10.*f['other_bid']*(1.-other_make))
    score = urgency*expected_contract-p['risk_aversion']*contract_risk-bags+denial-p['resource_weight']*f['cash']
    cover = jnp.where(f['partner_winning'],100.*f['own_win']-100.*(~f['wins']),25.*f['own_win'])
    score += f['partner_nil']*cover
    score += f['opponent_nil_winning']*(~f['wins'])*80.
    danger = f['rank']+.65*f['trump']+.25*f['cash']
    nil_score = -100.*f['own_win_all_seats']+(1.+p['q'])*danger*(1.-f['own_win_all_seats'])
    return jnp.where(f['own_nil'],nil_score,score)


def _formula_blind_action(state,key,family,variant):
    """Public comeback/eligibility only: no hands and no legal_mask call."""
    me = actor(state)
    team = me%2
    q = formula_style_parameters(variant)['q']
    comeback = (state.scores[1-team]-state.scores[team]>=180)&(state.scores[1-team]>=350)
    chance = .002+.01*q+comeback*(.02+.06*q)
    chance *= jnp.where(state.bids[(me+2)%4]==-1,.15,1.)
    return jnp.where(state.eligible[team]&(jax.random.uniform(key)<chance),67,66).astype(jnp.int32)


def _formula_pass_action(state,key,family,variant,legal):
    hand = _formula_acting_hand(state)
    me = actor(state)
    ids = jnp.arange(52)
    suits,ranks = ids//13,ids%13
    lengths = hand.reshape(4,13).sum(axis=1)
    q = formula_style_parameters(variant)['q']
    danger = ranks.astype(jnp.float32)/12.+.7*(suits==3)+.2*(lengths[suits]<=2)
    score = jnp.where(state.bids[me]==-1,(1.+q)*danger,-danger)
    score += .015*jax.random.uniform(key,(52,))
    chosen = jnp.argmax(jnp.where(legal[:52],score,-1.e9))
    return jnp.where(jnp.any(legal[:52]),chosen,66).astype(jnp.int32)


def _formula_bid_action(state,key,family,variant,legal):
    hand = _formula_acting_hand(state)
    me = actor(state)
    ids = jnp.arange(52,dtype=jnp.int32)
    opponents = jnp.arange(4)%2!=me%2
    cash = _formula_candidate_survival(state,hand,ids,ids//13,opponents)
    estimate = jnp.sum(hand*cash)
    _,nil_risk = estimate_bid_and_nil_risk(hand)
    q = formula_style_parameters(variant)['q']
    bid = jnp.clip(jnp.floor(estimate+.25+q),1,13).astype(jnp.int32)
    partner_bid = state.bids[(me+2)%4]
    bid = jnp.minimum(bid,jnp.maximum(1,13-jnp.maximum(partner_bid,0)))
    nil_threshold = (.45+.5*q)*jnp.where(partner_bid<=0,.25,1.)
    chosen = jnp.where(nil_risk<nil_threshold,52,52+bid)
    return jnp.where(legal[66],66,chosen).astype(jnp.int32)


def _formula_play_action(state,key,family,variant,legal):
    features = _formula_public_features(state)
    score = _formula_card_scores(features,formula_style_parameters(variant))
    score += .01*jax.random.uniform(key,(52,))
    return jnp.argmax(jnp.where(legal[:52],score,-1.e9)).astype(jnp.int32)


def _formula_nonblind_action(state,key,family,variant):
    legal = legal_mask(state)
    chosen = jax.lax.switch(jnp.maximum(phase(state)-1,0),
        (_formula_pass_action,_formula_bid_action,_formula_play_action),
        state,key,family,variant,legal)
    return jnp.where(legal[chosen],chosen,jnp.argmax(legal)).astype(jnp.int32)


def diverse_heuristic_action(state,key,style):
    family,variant,valid = decode_opponent_style_id(style)
    def extra(_):
        return jax.lax.cond(phase(state)==0,_formula_blind_action,
            _formula_nonblind_action,state,key,family,variant)
    chosen = jax.lax.cond(family==8,extra,
        lambda _:_formula_v2_action(state,key,style),operand=None)
    return jnp.where(valid,chosen,jnp.int32(-1)).astype(jnp.int32)
