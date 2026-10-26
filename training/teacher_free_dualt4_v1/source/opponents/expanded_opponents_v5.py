"""Append after exact tactical-v4. Opponent policies, never learner targets.

Eight new mechanisms and two separately implemented heldout mechanisms.
No hidden deal completion, search, learning signal, or engine modification.
"""

NOVEL_OPPONENT_REVISION = 'spades-novel-opponent-expansion-v5'
NOVEL_PROFILE_VALUE = 'teacher_free_novel845_dual_t4_v1'
NOVEL_TRAIN_FAMILIES = ('nil_shelter_lanes', 'nil_underlead_pressure',
    'positional_overruff_ladder', 'long_suit_entry_links', 'marginal_loser_disposal',
    'public_partner_suit_return', 'marginal_round_score', 'public_scenario_regret')
NOVEL_EVAL_FAMILIES = ('allocation_lower_bound_pressure', 'lexicographic_seat_safety')
NOVEL_NEW_TRAIN_STYLE_IDS = tuple(range(20000, 20512))
NOVEL_NEW_HOLDOUT_STYLE_IDS = tuple(range(21000, 21128))
NOVEL_TRAIN_STYLE_IDS = TACTICAL_TRAIN_STYLE_IDS + NOVEL_NEW_TRAIN_STYLE_IDS
NOVEL_HOLDOUT_STYLE_IDS = TACTICAL_HOLDOUT_STYLE_IDS + NOVEL_NEW_HOLDOUT_STYLE_IDS
NOVEL_VALID_STYLE_IDS = NOVEL_TRAIN_STYLE_IDS + NOVEL_HOLDOUT_STYLE_IDS

_novel_old_action = diverse_heuristic_action
_novel_old_decode = decode_opponent_style_id
_novel_old_sampler = sample_train_opponent_style_ids

NOVEL_MECHANISMS = (
    'Allocate shelter leads to suits where public partner low-card expenditure and own high cover imply exposure; conserve one cover per exposed lane.',
    'Lead below an opposing live nil public rank floor and replay suits in which its low cards have been spent; duck its current winning card.',
    'Select overruff rungs using outstanding higher trump counts and actual future seat order, distinguishing last-seat economy from exposed intermediate ruffs.',
    'Preserve sole outside entries to established long suits; run the long suit after blockers disappear, and use connected suit exits to retain access.',
    'Discard the card that most reduces residual own-hand exposure, including isolated honors, emptying suits, and retaining the last stopper.',
    'Return suits actually led by partner using public played_at chronology, weighted by recency, repeated leads, and known voids.',
    'Compare exact frozen-rule one-trick score deltas including coupled nil contracts and ten-bag rollover, with discounted future contract urgency.',
    'Minimize maximum regret across three explicit public threat scenarios (cheap capture, delayed overtake, resource preservation), without hidden deals.',
    'HELDOUT: use public suit-capacity lower bounds to pressure opponents forced to retain a suit, plus a distinct interval-based capture rule.',
    'HELDOUT: lexicographic positional rule tree prioritizes nil safety, public secure capture, late-seat economy, and public-information exits.',
)

NOVEL_OPPONENT_MANIFEST = {
    'revision': NOVEL_OPPONENT_REVISION, 'base_revision': TACTICAL_OPPONENT_REVISION,
    'profile_value': NOVEL_PROFILE_VALUE,
    'encoding': {'train_base':20000, 'eval_base':21000, 'family_stride':64,
        'new_train_family_indices':list(range(100,108)), 'new_eval_family_indices':[120,121]},
    'inherited_train_style_ids':list(TACTICAL_TRAIN_STYLE_IDS),
    'new_train_style_ids':list(NOVEL_NEW_TRAIN_STYLE_IDS),
    'train_style_ids':list(NOVEL_TRAIN_STYLE_IDS),
    'inherited_holdout_style_ids':list(TACTICAL_HOLDOUT_STYLE_IDS),
    'new_holdout_style_ids':list(NOVEL_NEW_HOLDOUT_STYLE_IDS),
    'holdout_style_ids':list(NOVEL_HOLDOUT_STYLE_IDS),
    'valid_style_ids':list(NOVEL_VALID_STYLE_IDS),
    'train_style_count':845, 'new_train_style_count':512,
    'holdout_style_count':256, 'new_holdout_style_count':128, 'valid_style_count':1101,
    'new_train_families':list(NOVEL_TRAIN_FAMILIES),
    'new_holdout_families':list(NOVEL_EVAL_FAMILIES),
    'mechanisms':dict(zip(NOVEL_TRAIN_FAMILIES+NOVEL_EVAL_FAMILIES,NOVEL_MECHANISMS)),
    'profiles': [{'id':base+64*f+v, 'family':name, 'variant':v,
        'axes':{'tempo':v//16, 'resource':(v//4)%4, 'risk':v%4},
        'split':split, 'identity':name+'/tempo'+str(v//16)+'/resource'+str((v//4)%4)+'/risk'+str(v%4)}
        for split,base,names in (('train',20000,NOVEL_TRAIN_FAMILIES),('holdout',21000,NOVEL_EVAL_FAMILIES))
        for f,name in enumerate(names) for v in range(64)],
    'variant_design':'4x4x4 factorial: tempo changes positional/remaining-trick thresholds, resource changes suit-length and card-disposal preferences, risk changes bid/cover/blind thresholds. No old-family modulo aliases.',
    'scripted_sampling':{'inherited_v4_probability':.5,'new_probability':.5,
        'new_uniform_families':8,'new_uniform_variants_per_family':64,'assignment_scope':'match'},
    'outer_sampling':{'selfplay':.5,'snapshot':.25,'scripted':.25,'new_absolute_probability':.125},
    'information':'Only actor hand via _novel_hand; public bids/tricks/scores/bags/table/played_by/played_at/voids. Blind only actor/eligible/scores/bags/bids, never any hand or legal_mask.',
    'preservation':'All461 inherited valid IDs delegate captured tactical-v4 with same state/key/style. All128 old exclusions retained. No state, RNG, optimizer, outer mixture or current-match style reset.',
    'evaluation':'All128 fresh IDs and128 inherited excluded IDs forbidden to sampler. Novel holdouts share honest feature extraction, phase handling and nil safeguards; their ordinary play mechanisms are separately implemented.',
    'fresh_evaluation_seeds':{'paired_deals':2026102411,'bootstrap':2026102412,
        'reserve_paired_deals':2026102413,'reserve_bootstrap':2026102414},
    'fresh_evaluation_policy':'Seed labels and128 excluded IDs fixed before any training outcomes; functional fixtures are development-visible; no heldout playing-strength games used to choose opponents or variants.',
    'invalid_action':-1,
    'strength':'Unmeasured. CPU legality, privacy and behavioral diversity tests are not evidence of improved playing strength or blind-nil outcomes.',
}


def decode_opponent_style_id(style):
    host = not hasattr(style, 'aval')
    kind = getattr(getattr(style, 'dtype', None), 'kind', '')
    scalar = isinstance(style, int) or (kind in ('i','u') and getattr(style,'size',1)==1)
    if host and scalar and (int(style)<0 or int(style)>21127):
        return jnp.int32(-2),jnp.int32(-1),jnp.bool_(False)
    host_range = True
    if host and kind in ('i','u'):
        host_range = jnp.asarray((style>=0)&(style<=21127))
    raw = jnp.asarray(style)
    integer = jnp.issubdtype(raw.dtype,jnp.integer)
    value = raw.astype(jnp.int32)
    old_f,old_v,old_ok = _novel_old_decode(style)
    train = (value>=20000)&(value<20512)
    held = (value>=21000)&(value<21128)
    base = jnp.where(held,21000,20000)
    f = jnp.where(train|held,(value-base)//64+jnp.where(held,120,100),old_f)
    v = jnp.where(train|held,(value-base)%64,old_v)
    valid = integer & host_range & (raw>=0)&(raw<=21127)&(old_ok|train|held)
    return f.astype(jnp.int32),v.astype(jnp.int32),valid


def is_train_opponent_style_id(style):
    f,v,valid = decode_opponent_style_id(style)
    inherited = (f==-1)|((f>=0)&(f<=2)&(v<32))|((f>=5)&(f<=11)&(v<32))
    return valid & (inherited|((f>=100)&(f<=107)))


def is_holdout_opponent_style_id(style):
    f,v,valid = decode_opponent_style_id(style)
    inherited = (((f==3)|(f==4)|(f==12)|(f==13))&(v>=32))
    return valid & (inherited|(f==120)|(f==121))


def sample_train_opponent_style_ids(key,n):
    if not isinstance(n,int) or isinstance(n,bool) or n<0:
        raise ValueError('n must be a static nonnegative integer')
    mix,old,new = jax.random.split(key,3)
    inherited = _novel_old_sampler(old,n)
    expanded = jax.random.randint(new,(n,),20000,20512,dtype=jnp.int32)
    return jnp.where(jax.random.bernoulli(mix,.5,(n,)),expanded,inherited)


def novel_style_parameters(variant):
    v = jnp.asarray(variant,jnp.int32)
    tempo,resource,risk = v//16,(v//4)%4,v%4
    return dict(tempo=tempo,resource=resource,risk=risk,
        t=tempo.astype(jnp.float32)/3.,r=resource.astype(jnp.float32)/3.,
        q=risk.astype(jnp.float32)/3.)


def _novel_hand(state):
    return state.hands[actor(state)]


def _novel_features(state):
    hand = _novel_hand(state)
    me = actor(state);partner=(me+2)%4
    ids=jnp.arange(52,dtype=jnp.int32);suits=ids//13;ranks=ids%13
    own=hand.reshape(4,13);lengths=own.sum(1)
    unseen=((state.played_by<0)&~hand).reshape(4,13)
    upper=jnp.arange(13)[None,:]>ranks[:,None]
    lower=jnp.arange(13)[None,:]<ranks[:,None]
    higher=(unseen[suits]&upper).sum(1);lower_own=(own[suits]&lower).sum(1)
    higher_own=(own[suits]&upper).sum(1)
    seats=jnp.arange(4);team=seats%2==me%2
    live=(state.bids<=0)&(state.tricks==0)
    failed=jnp.any(team&(state.bids<=0)&(state.tricks>0))
    other_failed=jnp.any(~team&(state.bids<=0)&(state.tricks>0))
    need=jnp.maximum(jnp.sum(jnp.where(team& (state.bids>0),state.bids-state.tricks,0)),0)
    other_need=jnp.maximum(jnp.sum(jnp.where(~team&(state.bids>0),state.bids-state.tricks,0)),0)
    table=jnp.maximum(state.table,0);led=table[0]//13
    strength=jnp.where(table//13==3,26+table%13,jnp.where(table//13==led,13+table%13,-1))
    strength=jnp.where(state.table>=0,strength,-100)
    wi=jnp.argmax(strength);winner=jnp.where(state.trick_pos==0,-1,state.table_players[wi])
    candidate=jnp.where(suits==3,26+ranks,jnp.where(suits==led,13+ranks,-1))
    wins=(state.trick_pos==0)|(candidate>strength.max())
    future=((seats-me)%4>0)&((seats-me)%4<=3-state.trick_pos)
    future_enemy=future&~team
    enemy_void=(state.voids&(~team)[:,None]).sum(0)
    candidate_led=jnp.where(state.trick_pos==0,suits,led)
    possible_higher=(higher[:,None]>0)&~state.voids[:,suits].T
    possible_ruff=(suits[:,None]!=3)&~state.voids[:,3][None,:]&(unseen[3].sum()>0)
    # Unknown void is a possible threat, not a claim about any actual hand.
    future_threat=jnp.any((possible_higher|possible_ruff)&future_enemy[None,:],1)
    secure=wins&~future_threat
    leads=(state.played_at>=0)&(state.played_at%4==0)&(state.played_by==partner)
    partner_leads=leads.reshape(4,13).sum(1)
    partner_recent=jnp.max(jnp.where(leads,state.played_at,-1).reshape(4,13),1)
    partner_used=(state.played_by==partner).reshape(4,13)
    partner_low=(partner_used&(jnp.arange(13)[None,:]<=5)).sum(1)
    enemy_nil=(~team)&live
    nil_played=jnp.any((state.played_by[:,None]==seats[None,:])&enemy_nil[None,:],1).reshape(4,13)
    nil_floor=jnp.min(jnp.where(nil_played,jnp.arange(13)[None,:],13),1)
    nil_count=nil_played.sum(1)
    # Exact frozen scoring is used only by the score-trade opponent, as an
    # analytic one-trick scenario. No future game states or hidden cards.
    base_delta=score_hand(state.bids,state.tricks,state.bags)[0]
    own_delta=score_hand(state.bids,state.tricks.at[me].add(1),state.bags)[0]
    opp=jnp.where((me+1)%4==winner,winner,(me+1)%4)
    enemy_delta=score_hand(state.bids,state.tricks.at[opp].add(1),state.bags)[0]
    # Utility(we take) - utility(the enemy takes): denied enemy gains add
    # to our own marginal gain, rather than being subtracted from it.
    swing=(own_delta[me%2]-base_delta[me%2])+(enemy_delta[1-me%2]-base_delta[1-me%2])
    remaining_cards=13-jnp.sum(state.played_by[:,None]==seats[None,:],axis=0)
    allowed=(~state.voids)&(seats!=me)[:,None]
    capacity=jnp.where(allowed,remaining_cards[:,None],0)
    # Lower bound on enemy suit count from finite public capacities. The
    # partner is the only other non-enemy unseen holder; no ownership guess.
    enemy_lower=jnp.maximum(unseen.sum(1)-capacity[partner],0)
    master=(higher==0)
    suit_masters=(hand&master).reshape(4,13).sum(1)
    entry=(master&(suits!=3))|(master&(suits==3)&(lengths[3]<=2))
    return dict(hand=hand,ids=ids,suits=suits,ranks=ranks,rank=ranks.astype(jnp.float32)/12.,
        trump=suits==3,lengths=lengths,higher=higher,lower_own=lower_own,higher_own=higher_own,
        unseen=unseen.sum(1),master=master,entry=entry,suit_masters=suit_masters,
        leading=state.trick_pos==0,pos=state.trick_pos,wins=wins,secure=secure,
        future_threat=future_threat,future_enemy_count=future_enemy.sum(),
        partner_winning=winner==partner,partner_nil_winning=(winner==partner)&live[partner]&~failed,
        enemy_nil_winning=jnp.any(enemy_nil&(seats==winner))&~other_failed,
        own_nil=live[me]&~failed,partner_nil=live[partner]&~failed,
        enemy_nil=jnp.any(enemy_nil)&~other_failed,need=jnp.where(failed,0,need),
        other_need=jnp.where(other_failed,0,other_need),remaining=jnp.maximum(13-(state.turn-12)//4,1),
        failed=failed,bags=state.bags[me%2],score_gap=state.scores[me%2]-state.scores[1-me%2],
        enemy_void=enemy_void,partner_void=state.voids[partner],partner_low=partner_low,
        partner_leads=partner_leads,partner_recent=partner_recent,nil_floor=nil_floor,nil_count=nil_count,
        swing=swing,enemy_lower=enemy_lower,legal=legal_mask(state)[:52])


def _novel_base(f,p):
    take=(f['need']>0)&((f['pos']>=p['tempo'])|(f['need']+p['tempo']>=f['remaining']))
    return 2.7*f['wins']*take-(.3+.6*p['r'])*f['rank']-.45*f['trump']-3.*f['partner_winning']*f['wins']


def _novel_shelter(f,p):
    s=f['suits'];base=_novel_base(f,p)
    exposed=f['partner_low'][s]+(f['unseen'][s]<=2+p['tempo'])
    cover=f['master']|(f['ranks']>=10-p['risk'])
    last_cover=cover&(f['higher_own']==0)
    lead=f['leading']&f['partner_nil']&~f['partner_void'][s]&~f['trump']
    base+=lead*cover*(2.+exposed)*(1.+p['q'])
    base-=~f['leading']*f['partner_nil']*~f['partner_nil_winning']*last_cover*(1.+p['r'])
    base+=lead*(f['lengths'][s]>=2+p['resource'])*.8
    return base


def _novel_underlead(f,p):
    s=f['suits'];base=_novel_base(f,p)
    observed=f['nil_count'][s]>0
    below=(f['ranks']+p['risk']<f['nil_floor'][s])&observed
    replay=observed&(f['nil_count'][s]>=1+p['tempo']//2)
    attack=f['leading']&f['enemy_nil']&~f['trump']
    # Use the highest card still below the observed floor, retaining lower
    # probes for another attempt. The floor is evidence, never a guarantee.
    base+=attack*(below*(3.+p['r']+2.*f['rank'])+replay*(1.+p['q'])-.4*f['rank'])
    base+=~f['leading']*f['enemy_nil']*~f['wins']*f['rank']*(1.+p['r'])
    return base


def _novel_overruff(f,p):
    base=_novel_base(f,p)
    exposed=(f['future_enemy_count']>0)&(f['pos']<=1+p['tempo']//2)
    rung=f['trump']&f['wins']&(f['higher']<=p['risk'])
    base+=~f['leading']*exposed*rung*(3.+2.*p['q'])
    base-=~f['leading']*exposed*f['trump']*f['wins']*jnp.minimum(f['higher'],5)*(.4+.4*p['r'])
    base+=~exposed*f['wins']*(1.-f['rank'])*(.5+p['r'])
    return base


def _novel_entries(f,p):
    s=f['suits'];base=_novel_base(f,p)
    run=(f['lengths']>=3+p['resource'])&(f['suit_masters']>=1)
    established=run[s]&(f['unseen'][s]<=p['risk']+1)
    other_run=jnp.any(run)&~run[s]
    sole_entry=f['entry']&other_run&(jnp.sum(f['entry']&f['hand'])<=1+p['tempo']//2)
    base-=sole_entry*(2.+2.*p['r'])*~(f['need']>=f['remaining'])
    base+=f['leading']*established*f['master']*(3.+p['q'])
    base+=~f['leading']*f['partner_winning']*~f['wins']*run[s]*(f['higher_own']>0)*(1.+p['t'])
    return base


def _novel_disposal(f,p):
    s=f['suits'];base=_novel_base(f,p)
    isolated=(f['lower_own']<=p['resource'])&(f['ranks']>=7+p['risk'])
    creates_void=(f['lengths'][s]==1)&~f['trump']&(f['lengths'][3]>0)
    last_stopper=f['master']&(f['higher_own']==0)&(f['lengths'][s]>=2+p['tempo'])
    gain=(1.+p['r'])*isolated+(1.2+p['t'])*creates_void+f['rank']-last_stopper*(1.+p['q'])
    base+=~f['leading']*~f['wins']*gain*2.
    base+=f['leading']*creates_void*(f['remaining']>4+p['tempo'])*1.6
    return base


def _novel_return(f,p):
    s=f['suits'];base=_novel_base(f,p)
    signal=f['partner_leads'][s]*(1.+p['r'])+jnp.maximum(f['partner_recent'][s],0)/52.*(1.+p['t'])
    usable=(f['partner_leads'][s]>0)&~f['partner_void'][s]&~f['trump']
    base+=f['leading']*usable*signal*(2.+p['q'])*(1.-.5*f['rank'])
    base-=f['leading']*usable*f['master']*(f['lengths'][s]>1+p['resource'])*(.5+p['t'])
    return base


def _novel_score_trade(f,p):
    base=_novel_base(f,p)
    immediate=jnp.clip(f['swing']/20.,-8.,8.)
    urgency=jnp.where(f['need']>0,3.*f['need']/f['remaining'],0.)
    bag_cost=jnp.where((f['need']==0)&(f['bags']>=7+p['risk']),3.+p['r'],0.)
    base+=f['wins']*(immediate*(.5+.5*p['t'])+urgency-bag_cost)
    base+=~f['wins']*f['rank']*(f['need']==0)*(1.+p['r'])
    return base


def _novel_regret(f,p):
    capture=f['wins']*(f['need']>0)*4.-f['rank']-.5*f['trump']
    delayed=f['secure']*(3.+p['q'])-f['future_threat']*f['wins']*(1.+p['t'])-.2*f['rank']
    resource=~f['wins']*(f['rank']+f['lengths'][f['suits']]/13.)-f['entry']*(1.+p['r'])
    values=jnp.stack((capture,delayed,resource))
    maxima=jnp.max(jnp.where(f['legal'][None,:],values,-1.e6),axis=1)
    regret=maxima[:,None]-values
    weights=jnp.array([1.+p['t'],.6+.8*p['q'],.4+.9*p['r']])
    return -jnp.max(weights[:,None]*regret,axis=0)+.08*_novel_base(f,p)


def _novel_eval_bounds(f,p):
    """Independent holdout: finite-capacity lower bounds, not train tactics."""
    s=f['suits'];forced=f['enemy_lower'][s]>=1+p['resource']
    capture=f['wins']&(f['higher']<=p['risk'])
    lead=forced*~f['trump']*(f['lengths'][s]>=2+p['tempo']//2)
    score=f['leading']*lead*(4.+f['enemy_lower'][s]/3.)
    score+=~f['leading']*capture*(f['need']>0)*(3.+p['t'])
    score-=f['rank']*(.7+p['q'])+f['entry']*.7
    score-=f['partner_winning']*f['wins']*3.
    return score


def _novel_eval_lexicographic(f,p):
    """Independent holdout: priority classes dominate rank economy."""
    urgent=f['need']+p['tempo']>=f['remaining']
    last=f['pos']>=3-p['risk']//2
    take=(f['need']>0)&(last|urgent)
    secure=f['secure']&~f['partner_winning']
    safe_exit=~f['wins']&~f['entry']
    priority=jnp.where(secure,4,jnp.where(take&f['wins'],3,jnp.where(safe_exit,2,0)))
    lead_class=jnp.where(f['master']&(f['need']>0),3,
        jnp.where((f['lengths'][f['suits']]<=1+p['resource'])&~f['trump'],2,1))
    priority=jnp.where(f['leading'],lead_class,priority)
    return 10.*priority-f['rank']*(1.+p['r'])-.25*f['trump']


def _novel_blind_action(state,key,family,variant):
    """Strictly predeal public information; never hand or legal-mask access."""
    me=actor(state);team=me%2;p=novel_style_parameters(variant)
    deficit=state.scores[1-team]-state.scores[team]
    chase=(deficit>=180+30*p['tempo'])&(state.scores[1-team]>=300+30*p['resource'])
    base=.004+.006*p['q']
    chance=base+chase*(.025+.06*p['q'])+(state.bags[team]>=8)*.004
    chance*=jnp.where(state.bids[(me+2)%4]==-1,.1,1.)
    chance*=jnp.where(family==101,1.15,jnp.where(family==120,.7,jnp.where(family==121,.85,1.)))
    return jnp.where(state.eligible[team]&(jax.random.uniform(key)<chance),67,66).astype(jnp.int32)


def _novel_pass_action(state,key,family,variant,legal):
    hand=_novel_hand(state);me=actor(state);p=novel_style_parameters(variant)
    ids=jnp.arange(52);s=ids//13;r=ids%13;own=hand.reshape(4,13);lengths=own.sum(1)
    lower=(own[s]&(jnp.arange(13)[None,:]<r[:,None])).sum(1)
    danger=(r/12.)**2*(1.+1./(1.+lower))+.8*(s==3)+.35*(lengths[s]<=2)
    export=danger+(lengths[s]==1)*(.2+.5*p['r'])
    gift=-danger-(r>=9)*(lengths[s]<=2)*(.5+p['q'])
    gift+=(lengths[s]>=3+p['resource'])*(r<=3+p['risk'])*.4
    score=jnp.where(state.bids[me]==-1,export,gift)+.0001*jax.random.uniform(key,(52,))
    chosen=jnp.argmax(jnp.where(legal[:52],score,-1.e9))
    return jnp.where(jnp.any(legal[:52]),chosen,66).astype(jnp.int32)


def _novel_bid_action(state,key,family,variant,legal):
    """Own-hand structural count; no other hand or public played-card inference."""
    hand=_novel_hand(state);p=novel_style_parameters(variant)
    own=hand.reshape(4,13);lengths=own.sum(1)
    aces=own[:,12];kings=own[:,11];queens=own[:,10]
    top=aces+kings*(lengths>=2)*.65+queens*(lengths>=3)*.35
    trump=own[3,12]+own[3,11]*.8+own[3,10]*.6+jnp.maximum(lengths[3]-3,0)*(.55+.2*p['r'])
    ruffs=jnp.sum((lengths[:3]<=1)&(lengths[3]>=3))*(.25+.2*p['t'])
    estimate=top[:3].sum()+trump+ruffs
    bid=jnp.clip(jnp.floor(estimate+.15+.75*p['q']),1,13).astype(jnp.int32)
    danger=jnp.sum(own[:,10:])+jnp.sum(own[3,7:])+jnp.maximum(lengths[3]-3,0)
    nil=(danger==0)&(estimate<.35+.2*p['q'])&(jnp.max(lengths)<=5+p['resource']//2)
    chosen=jnp.where(nil,52,52+bid)
    return jnp.where(legal[66],66,chosen).astype(jnp.int32)


def _novel_play_action(state,key,family,variant,legal):
    f=_novel_features(state);p=novel_style_parameters(variant)
    index=jnp.where(family>=120,family-112,family-100)
    score=jax.lax.switch(index,(_novel_shelter,_novel_underlead,_novel_overruff,
        _novel_entries,_novel_disposal,_novel_return,_novel_score_trade,_novel_regret,
        _novel_eval_bounds,_novel_eval_lexicographic),f,p)
    # Actual overtaking preserves a live nil; never assume success from a lead.
    score+=f['partner_nil_winning']*f['wins']*(40.+5.*p['q'])
    score-=f['enemy_nil_winning']*f['wins']*(40.+5.*p['q'])
    danger=f['rank']**2*(1.+1./(1.+f['lower_own']))+.9*f['trump']+.3*f['master']
    nil_follow=-60.*f['wins']+(1.+p['r'])*danger*~f['wins']
    nil_lead=-3.*danger-.5/(1.+f['lengths'][f['suits']])-.3*f['partner_void'][f['suits']]
    score=jnp.where(f['own_nil'],jnp.where(f['leading'],nil_lead,nil_follow),score)
    score+=.0001*jax.random.uniform(key,(52,))
    return jnp.argmax(jnp.where(legal[:52],score,-1.e9)).astype(jnp.int32)


def _novel_nonblind_action(state,key,family,variant):
    legal=legal_mask(state)
    chosen=jax.lax.switch(jnp.maximum(phase(state)-1,0),
        (_novel_pass_action,_novel_bid_action,_novel_play_action),state,key,family,variant,legal)
    return jnp.where(legal[chosen],chosen,jnp.argmax(legal)).astype(jnp.int32)


def diverse_heuristic_action(state,key,style):
    family,variant,valid=decode_opponent_style_id(style)
    old_style=jnp.where((family>=0)&(family<100),13+64*family+variant,
        jnp.where(family==-1,variant,0)).astype(jnp.int32)
    def admitted(_):
        def novel(_):
            return jax.lax.cond(phase(state)==0,_novel_blind_action,_novel_nonblind_action,
                state,key,family,variant)
        return jax.lax.cond(family>=100,novel,
            lambda _:_novel_old_action(state,key,old_style),operand=None)
    return jax.lax.cond(valid,admitted,lambda _:jnp.int32(-1),operand=None)
