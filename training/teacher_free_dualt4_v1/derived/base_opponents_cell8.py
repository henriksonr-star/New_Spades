# Observation-limited development opponents. All styles share one vectorized
# feature computation so a heterogeneous JAX batch remains efficient.
HEURISTIC_NAMES = (
    "random", "conservative", "aggressive", "nil_specialist",
    "partner_support", "bag_averse", "nil_hunter", "adaptive",
    "blind_gambler", "low_card", "high_card", "card_counter",
    "contract_disruptor",
)


def estimate_bid_and_nil_risk(hand):
    """Estimate tricks and nil danger using ONLY the acting player's hand.

    The estimate is a baseline, not a claim of optimal bidding. Nil thresholds
    are deliberately stricter than in ordinary Spades because a failed nil
    also loses the partner's entire contract in this ruleset.
    """
    cards = hand.reshape(4, 13).astype(jnp.float32)
    lengths = cards.sum(axis=1)
    side = cards[:3]
    side_len = lengths[:3]
    honors = (
        .94 * side[:, 12]
        + jnp.where(side_len >= 2, .61, .26) * side[:, 11]
        + jnp.where(side_len >= 3, .29, .06) * side[:, 10]
        + jnp.where(side_len >= 4, .11, .01) * side[:, 9]
    )
    # Very long side suits are likely to be ruffed before all their honors cash.
    side_tricks = jnp.sum(honors * jnp.clip(1.05 - .045 * side_len, .65, 1.))
    trump_values = jnp.array([.03, .05, .08, .12, .17, .23, .31,
                             .42, .56, .70, .83, .95, 1.])
    trump_tricks = jnp.sum(cards[3] * trump_values)
    trump_tricks += .36 * jnp.maximum(lengths[3] - 3., 0.)
    shortness = jnp.sum(jnp.where(side_len == 0, .65,
                                 jnp.where(side_len == 1, .38, .0)))
    ruffs = jnp.minimum(shortness, .40 * jnp.maximum(lengths[3] - 1., 0.))
    estimate = jnp.clip(side_tricks + trump_tricks + ruffs, 0., 13.)

    ranks = jnp.arange(13, dtype=jnp.float32)
    # Higher cards are less exposed in a long suit with several small exits.
    exposure = ((ranks + 1.) / 13.) ** 4
    low_exits = jnp.sum(cards * (ranks <= 5), axis=1)
    protection = jnp.sqrt(jnp.maximum(lengths, 1.)) + .30 * low_exits
    nil_risk = jnp.sum(cards * exposure / protection[:, None])
    nil_risk += .9 * jnp.sum(cards[3] * ((ranks + 1.) / 13.) ** 3)
    nil_risk += jnp.sum(cards * (ranks >= 10) * (lengths[:, None] <= 2)) * .45
    # The ace of trump can never be ducked. Long trump holdings also force wins.
    nil_risk += 10. * cards[3, 12] + .18 * jnp.maximum(lengths[3] - 3., 0.)
    return estimate, nil_risk


def _heuristic_blind(state, key, style, legal):
    # IMPORTANT: this phase never inspects any hand, including our own.
    me = actor(state)
    team = me % 2
    deficit = state.scores[1 - team] - state.scores[team]
    chance = jnp.array([.5, .0, .06, .12, .01, .01, .03,
                        .04, .60, .0, .0, .025, .08])[style]
    late_comeback = (state.scores[1 - team] >= 400) & (deficit >= 200)
    chance = jnp.where((style == 7) & late_comeback, .22, chance)
    already_blind = state.bids[(me + 2) % 4] == -1
    chance *= jnp.where(already_blind, jnp.where(style == 8, .6, .08), 1.)
    declare = legal[67] & (jax.random.uniform(key) < chance)
    return jnp.where(declare, 67, 66).astype(jnp.int32)


def _heuristic_pass(state, key, style, legal):
    me = actor(state)
    hand = state.hands[me]
    rank = jnp.arange(52) % 13
    suit = jnp.arange(52) // 13
    lengths = hand.reshape(4, 13).sum(axis=1)
    # A blind nil bidder exports the card most likely to force a trick.
    danger = rank.astype(jnp.float32) + 5. * (suit == 3)
    danger += 2. * ((lengths[suit] <= 2) & (rank >= 9))
    # A supporting partner gives a small card, keeping high covering cards.
    gift = -rank.astype(jnp.float32) - .15 * (suit == 3)
    gift += .15 * (lengths[suit] <= 2)
    score = jnp.where(state.bids[me] == -1, danger, gift)
    score += .02 * jax.random.uniform(key, (52,))
    chosen = jnp.argmax(jnp.where(legal[:52], score, -1.e9))
    return jnp.where(jnp.any(legal[:52]), chosen, 66).astype(jnp.int32)


def _heuristic_bid(state, key, style, legal):
    me = actor(state)
    partner = (me + 2) % 4
    estimate, nil_risk = estimate_bid_and_nil_risk(state.hands[me])
    bias = jnp.array([0., -.35, .75, -.25, -.15, .40, .20,
                     .0, .0, -.80, .80, .20, .10])[style]
    bid = jnp.clip(jnp.floor(estimate + .35 + bias), 1, 13).astype(jnp.int32)
    partner_bid = state.bids[partner]
    # Avoid creating a normal contract that is mathematically impossible.
    bid = jnp.minimum(bid, jnp.maximum(1, 13 - jnp.maximum(partner_bid, 0)))
    threshold = jnp.array([0., .72, .50, 1.32, .78, .55, .82,
                          .95, .90, -.1, -.1, .90, .85])[style]
    partner_special = (partner_bid == 0) | (partner_bid == -1)
    threshold *= jnp.where(partner_special, .25, 1.)
    nil = nil_risk < threshold
    action = jnp.where(nil, 52, 52 + bid)
    # Double-nil players have already committed and have a forced skip.
    return jnp.where(legal[66], 66, action).astype(jnp.int32)


def _heuristic_play(state, key, style, legal):
    me = actor(state)
    partner = (me + 2) % 4
    team = me % 2
    ids = jnp.arange(52)
    ranks = ids % 13
    suits = ids // 13
    hand = state.hands[me]
    lengths = hand.reshape(4, 13).sum(axis=1)
    leading = state.trick_pos == 0
    led = jnp.maximum(state.table[0], 0) // 13
    table_cards = jnp.maximum(state.table, 0)
    table_suits = table_cards // 13
    table_rank = table_cards % 13
    strength = jnp.where(table_suits == 3, 26 + table_rank,
                         jnp.where(table_suits == led, 13 + table_rank, -1))
    strength = jnp.where(state.table >= 0, strength, -100)
    table_index = jnp.argmax(strength)
    current_card = table_cards[table_index]
    current_winner = jnp.where(leading, -1, state.table_players[table_index])
    candidate_strength = jnp.where(suits == 3, 26 + ranks,
                                   jnp.where(suits == led, 13 + ranks, -1))
    would_win = leading | (candidate_strength > jnp.max(strength))

    # Card counting uses played-card ownership and our own cards only.
    unseen = ((state.played_by < 0) & ~hand).reshape(4, 13)
    unseen_count = unseen.sum(axis=1)
    higher = jnp.flip(jnp.cumsum(jnp.flip(unseen, axis=1), axis=1), axis=1) - unseen
    unseen_total = jnp.maximum(unseen_count.sum(), 1).astype(jnp.float32)
    seats = jnp.arange(4)
    remaining_cards = 13 - jnp.sum(state.played_by[:, None] == seats[None, :], axis=0)
    relative = (seats - me) % 4
    future = (relative > 0) & (relative <= 3 - state.trick_pos)

    def survival(cards, lead_suits):
        cs, cr = cards // 13, cards % 13
        high_fraction = higher[cs, cr].astype(jnp.float32) / unseen_total
        p_high = 1. - jnp.exp(-high_fraction[:, None] * remaining_cards[None, :])
        known_void = state.voids[:, cs].T
        p_high = jnp.where(known_void, 0., p_high)
        lead_fraction = unseen_count[lead_suits].astype(jnp.float32) / unseen_total
        p_void = jnp.exp(-lead_fraction[:, None] * remaining_cards[None, :])
        p_void = jnp.where(state.voids[:, lead_suits].T, 1., p_void)
        trump_fraction = unseen_count[3].astype(jnp.float32) / unseen_total
        p_trump = 1. - jnp.exp(-trump_fraction * remaining_cards)
        p_trump = jnp.where(state.voids[:, 3], 0., p_trump)
        p_ruff = p_void * p_trump[None, :] * (cs[:, None] != 3)
        p_beat = 1. - (1. - p_high) * (1. - p_ruff)
        return jnp.prod(jnp.where(future[None, :], 1. - p_beat, 1.), axis=1)

    all_cards = jnp.concatenate((ids, current_card[None]))
    lead_suits = jnp.concatenate((jnp.where(leading, suits, led), led[None]))
    survives = survival(all_cards, lead_suits)
    own_win_probability = would_win.astype(jnp.float32) * survives[:52]
    existing_win_probability = survives[52]
    partner_winning = current_winner == partner
    team_win_probability = own_win_probability + (
        (~would_win) * partner_winning * existing_win_probability
    )

    own_team = seats % 2 == team
    special = state.bids <= 0
    live_nil = special & (state.tricks == 0)
    failed_nil = special & (state.tricks > 0)
    remaining_tricks = 13 - jnp.maximum(state.turn - 12, 0) // 4
    normal_bid = jnp.where(state.bids > 0, state.bids, 0)
    normal_tricks = jnp.where(state.bids > 0, state.tricks, 0)
    our_need = jnp.maximum(jnp.sum(jnp.where(own_team, normal_bid - normal_tricks, 0)), 0)
    their_need = jnp.maximum(jnp.sum(jnp.where(~own_team, normal_bid - normal_tricks, 0)), 0)
    our_failed = jnp.any(own_team & failed_nil) | (our_need > remaining_tricks)
    their_failed = jnp.any(~own_team & failed_nil) | (their_need > remaining_tricks)
    own_nil = live_nil[me] & ~our_failed
    partner_nil = live_nil[partner] & ~our_failed
    opponents_nil = (~own_team) & live_nil & ~their_failed
    current_opp_nil = jnp.any(opponents_nil & (seats == current_winner))

    bid_drive = jnp.array([1., 1.20, 1.45, 1., 1.10, 1.0, .95,
                          1.20, 1.20, 1., 1., 1.20, .85])[style]
    bag_weight = jnp.array([.4, .55, .18, .45, .5, 1.6, .35,
                           .85, .45, .4, .2, .95, .65])[style]
    hunt_weight = jnp.array([0., .7, .65, 1., .7, .75, 2.3,
                            1.4, 1., .0, .0, 1.35, 2.0])[style]
    support_weight = jnp.array([0., 1., .8, 1.2, 1.8, 1., .9,
                               1.4, 1.1, 0., 0., 1.4, .9])[style]
    denial_weight = jnp.array([0., .18, .45, .20, .20, .20, .35,
                              .35, .30, .0, .0, .40, 1.0])[style]
    urgency = 1. + 2. * our_need / jnp.maximum(remaining_tricks, 1)
    our_drive = jnp.where(our_need > 0, bid_drive * urgency,
                         -bag_weight * (1. + state.bags[team] / 4.))
    # In this variant a failed special contract is already irrecoverable.
    # Stop chasing our own bid; only disruption of opponents has value then.
    our_drive = jnp.where(our_failed, 0., our_drive)
    deny = jnp.where(their_need > 0, denial_weight,
                     -denial_weight * (.5 + state.bags[1 - team] / 4.))
    deny = jnp.where(their_failed, 0., deny)
    score = (our_drive + deny) * team_win_probability

    master = higher.reshape(52) == 0
    rank_value = ranks.astype(jnp.float32) / 12.
    card_cost = .22 * rank_value + .22 * (suits == 3) + .08 * master
    score -= card_cost
    score += leading * (.025 * lengths[suits] +
                        .08 * (lengths[suits] == 1) * (lengths[3] >= 2))
    score += leading * (style == 11) * .25 * master * (suits != 3) * (our_need > 0)

    # Protect a live partner nil, particularly if they currently hold the trick.
    partner_future = ((partner - me) % 4) <= 3 - state.trick_pos
    cover = jnp.where(partner_winning, 9. * would_win,
                      jnp.where(leading | partner_future, 2.8 * own_win_probability, 0.))
    score += partner_nil * support_weight * cover

    # If an opposing nil currently holds the trick, leave it with them.
    nil_trap = current_opp_nil * (~would_win) * existing_win_probability * 7.
    # Lead a small card in a suit an opposing nil can still follow; a low
    # percentile card leaves that opponent fewer smaller cards to duck with.
    above_fraction = higher.reshape(52) / jnp.maximum(unseen_count[suits], 1)
    can_follow = ~state.voids[:, suits].T
    target_follow = jnp.sum(can_follow * opponents_nil[None, :], axis=1)
    lead_trap = leading * target_follow * above_fraction * .85
    score += hunt_weight * (nil_trap + lead_trap)

    # A nil bidder dumps dangerous cards when safely below the winning card.
    danger = rank_value + .55 * (suits == 3) + .30 * master
    danger += .18 * (lengths[suits] <= 2)
    nil_score = -9. * own_win_probability + 1.45 * danger * (1. - own_win_probability)
    nil_score -= .08 * rank_value
    nil_score += .25 * hunt_weight * (nil_trap + lead_trap)
    score = jnp.where(own_nil, nil_score, score)

    # Deliberately simple endpoints expose policies overfitted to clever bots.
    score = jnp.where(style == 9, -ranks.astype(jnp.float32) - .05 * (suits == 3), score)
    score = jnp.where(style == 10, ranks.astype(jnp.float32) + .05 * (suits == 3), score)
    score += .035 * jax.random.uniform(key, (52,))
    return jnp.argmax(jnp.where(legal[:52], score, -1.e9)).astype(jnp.int32)


def heuristic_action(state, key, style):
    """Legal action for one observation-limited opponent; supports jit/vmap.

    Private information is restricted to state.hands[actor(state)]. No opponent
    hand, pending pass, centralized observation, or undealt future is consulted.
    During blind polling even the acting player's hand is ignored. Actions can
    depend on public seat, bids, scores, bags, table, voids, and played cards.
    """
    style = jnp.clip(jnp.asarray(style, jnp.int32), 0, len(HEURISTIC_NAMES) - 1)
    legal = legal_mask(state)
    chosen = jax.lax.switch(
        phase(state),
        (_heuristic_blind, _heuristic_pass, _heuristic_bid, _heuristic_play),
        state, key, style, legal,
    )
    random_choice = jnp.argmax(jnp.where(legal, jax.random.uniform(key, (68,)), -1.)).astype(jnp.int32)
    chosen = jnp.where(style == 0, random_choice, chosen)
    # A common final mask protects even deliberately weak stress-test styles.
    return jnp.where(legal[chosen], chosen, jnp.argmax(legal)).astype(jnp.int32)


# Explicit opponent-diversity-v1 module; original thirteen heuristics above remain unchanged.
"""Additional observation-limited opponents; append after original cell8.

Uses the supplied JAX/rules/original heuristic namespace. No game-rule, learning,
state-schema, original-name or original-function replacement occurs here.
Encoded IDs persist in the existing scalar league_style for an entire match.
"""

DIVERSITY_OPPONENT_REVISION = 'spades-opponent-diversity-v1'
DIVERSITY_STYLE_BASE = 13
DIVERSITY_FAMILY_STRIDE = 64
DIVERSITY_TRAIN_FAMILIES = ('trump_management', 'void_building', 'tempo_ducking')
DIVERSITY_HOLDOUT_FAMILIES = ('crossruff_pressure', 'score_threshold_defense')
DIVERSITY_TRAIN_VARIANTS = tuple(range(32))
DIVERSITY_HOLDOUT_VARIANTS = tuple(range(32, 64))
DIVERSITY_TRAIN_STYLE_IDS = tuple(range(13)) + tuple(
    13 + family * 64 + variant for family in range(3) for variant in range(32))
DIVERSITY_HOLDOUT_STYLE_IDS = tuple(
    13 + family * 64 + variant for family in range(3, 5) for variant in range(32, 64))
DIVERSITY_MAX_STYLE_ID = 332
DIVERSITY_OPPONENT_MANIFEST = {
    'revision': DIVERSITY_OPPONENT_REVISION,
    'legacy_ids': list(range(13)),
    'encoding': {'base': 13, 'family_stride': 64, 'maximum_id': 332},
    'train_families': list(DIVERSITY_TRAIN_FAMILIES),
    'holdout_families': list(DIVERSITY_HOLDOUT_FAMILIES),
    'train_family_indices': [0, 1, 2], 'holdout_family_indices': [3, 4],
    'train_variants': list(range(32)), 'holdout_variants': list(range(32, 64)),
    'train_style_ids': list(DIVERSITY_TRAIN_STYLE_IDS),
    'holdout_style_ids': list(DIVERSITY_HOLDOUT_STYLE_IDS),
    'parameter': 'p=(variant+0.5)/64',
    'parameter_quantization': '32 midpoint bins per split; q=(variant%32+0.5)/32',
    'used_parameter_bands': {
        'train': {'bid_bias_interval': [-0.4, 0.4], 'bid_bias_attainable': [-0.3875, 0.3875],
                  'risk_scale_interval': [0.8, 1.2], 'risk_scale_attainable': [0.80625, 1.19375],
                  'urgency_scale_interval': [0.8, 1.2], 'urgency_scale_attainable': [0.80625, 1.19375]},
        'holdout_crossruff_pressure': {'bid_bias_interval': [-1.0, -0.7], 'bid_bias_attainable': [-0.9953125, -0.7046875],
                  'risk_scale_interval': [1.35, 1.7], 'risk_scale_attainable': [1.35546875, 1.69453125],
                  'urgency_scale_interval': [1.35, 1.7], 'urgency_scale_attainable': [1.35546875, 1.69453125]},
        'holdout_score_threshold_defense': {'bid_bias_interval': [0.7, 1.0], 'bid_bias_attainable': [0.7046875, 0.9953125],
                  'risk_scale_interval': [1.35, 1.7], 'risk_scale_attainable': [1.35546875, 1.69453125],
                  'urgency_scale_interval': [1.35, 1.7], 'urgency_scale_attainable': [1.35546875, 1.69453125]},
    },
    'family_rules': {
        'trump_management': 'lead trump from long holdings when contract needs tricks; conserve high trump; cheap ruffs',
        'void_building': 'clear short side suits while holding trump; use low long-suit entries',
        'tempo_ducking': 'duck early seats; cheapest last-seat win when needed; urgency threshold switches intent',
        'crossruff_pressure': 'HELDOUT partner-known-void leads interact with opponent-void avoidance and master cashing',
        'score_threshold_defense': 'HELDOUT public score and bag thresholds switch denial versus avoiding bags',
    },
    'train_parameter_interval': [0.0078125, 0.4921875],
    'holdout_parameter_interval': [0.5078125, 0.9921875],
    'scripted_sampling': {'legacy_probability': 0.5, 'new_probability': 0.5,
                          'legacy_uniform_count': 13, 'new_uniform_family_count': 3,
                          'new_uniform_variants_per_family': 32, 'assignment_scope': 'match'},
    'invalid_action': -1,
    'information': 'acting hand and public bids/tricks/scores/bags/table/played ownership/voids only; blind uses no hands',
    'search': 'none; fixed52-card feature scores; no lookahead/neural inference',
    'strength': 'unmeasured; diversity does not imply stronger opponents',
}


def decode_opponent_style_id(style):
    """JAX scalar decode: legacy family=-1, variant=legacy ID; invalid=False.

    Caller state stores int32 IDs. No clipping converts invalid/reserved IDs
    into an admitted training style. Host/native boundaries validate the dtype.
    """
    raw = jnp.asarray(style)
    integer_dtype = jnp.issubdtype(raw.dtype, jnp.integer)
    value = raw.astype(jnp.int32)
    offset = value - DIVERSITY_STYLE_BASE
    family = offset // DIVERSITY_FAMILY_STRIDE
    variant = offset % DIVERSITY_FAMILY_STRIDE
    legacy = (value >= 0) & (value < DIVERSITY_STYLE_BASE)
    train = (family >= 0) & (family < 3) & (variant < 32)
    holdout = (family >= 3) & (family < 5) & (variant >= 32)
    return (jnp.where(legacy, -1, family).astype(jnp.int32),
            jnp.where(legacy, value, variant).astype(jnp.int32),
            (legacy | train | holdout) & integer_dtype)


def is_train_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    return valid & ((family == -1) | ((family >= 0) & (family < 3) & (variant < 32)))


def is_holdout_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    return valid & (family >= 3) & (family < 5) & (variant >= 32)


def sample_train_opponent_style_ids(key, n):
    """Half original13, half balanced3x32 train variants; never holdout IDs.

    The caller supplies its existing per-match style key and keeps the encoded
    scalar until that match finishes. This does not add a persistent RNG stream.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('Static nonnegative Python integer population size required.')
    mix_key, legacy_key, family_key, variant_key = jax.random.split(key, 4)
    legacy = jax.random.randint(legacy_key, (n,), 0, 13, dtype=jnp.int32)
    family = jax.random.randint(family_key, (n,), 0, 3, dtype=jnp.int32)
    variant = jax.random.randint(variant_key, (n,), 0, 32, dtype=jnp.int32)
    new_style = 13 + family * 64 + variant
    return jnp.where(jax.random.bernoulli(mix_key, 0.5, (n,)), legacy,
                     new_style).astype(jnp.int32)


def diversity_style_parameters(family, variant):
    """Actual finite profile parameters; train and heldout bands are disjoint."""
    q = ((variant % 32).astype(jnp.float32) + 0.5) / 32.
    train = family < 3
    bid_bias = jnp.where(train, -0.4 + 0.8 * q,
                        jnp.where(family == 3, -1.0 + 0.3 * q, 0.7 + 0.3 * q))
    scale = jnp.where(train, 0.8 + 0.4 * q, 1.35 + 0.35 * q)
    return dict(p=(variant.astype(jnp.float32) + 0.5) / 64.,
                bid_bias=bid_bias, risk_scale=scale, urgency_scale=scale)


def _diversity_acting_hand(state):
    """The only private-card read in the new play/pass/bid implementation."""
    return state.hands[actor(state)]


def _diversity_public_play_features(state):
    """Shared cheap52-card features from own hand and public game state."""
    me = actor(state)
    partner = (me + 2) % 4
    team = me % 2
    ids = jnp.arange(52, dtype=jnp.int32)
    suits, ranks = ids // 13, ids % 13
    rank = ranks.astype(jnp.float32) / 12.
    hand = _diversity_acting_hand(state)
    lengths = hand.reshape(4, 13).sum(axis=1)
    leading = state.trick_pos == 0
    led = jnp.maximum(state.table[0], 0) // 13
    table = jnp.maximum(state.table, 0)
    table_strength = jnp.where(table // 13 == 3, 26 + table % 13,
                               jnp.where(table // 13 == led, 13 + table % 13, -1))
    table_strength = jnp.where(state.table >= 0, table_strength, -100)
    current_winner = jnp.where(leading, -1, state.table_players[jnp.argmax(table_strength)])
    candidate_strength = jnp.where(suits == 3, 26 + ranks,
                                   jnp.where(suits == led, 13 + ranks, -1))
    wins = leading | (candidate_strength > jnp.max(table_strength))
    partner_winning = current_winner == partner
    unseen = ((state.played_by < 0) & ~hand).reshape(4, 13)
    higher = jnp.flip(jnp.cumsum(jnp.flip(unseen, axis=1), axis=1), axis=1) - unseen
    master = higher.reshape(52) == 0
    seats = jnp.arange(4)
    own_team = seats % 2 == team
    normal_bid = jnp.maximum(state.bids, 0)
    normal_tricks = jnp.where(state.bids > 0, state.tricks, 0)
    our_need = jnp.maximum(jnp.sum(jnp.where(own_team, normal_bid - normal_tricks, 0)), 0)
    their_need = jnp.maximum(jnp.sum(jnp.where(~own_team, normal_bid - normal_tricks, 0)), 0)
    remaining = jnp.maximum(13 - jnp.maximum(state.turn - 12, 0) // 4, 1)
    live_nil = (state.bids <= 0) & (state.tricks == 0)
    own_failed = jnp.any(own_team & (state.bids <= 0) & (state.tricks > 0))
    their_failed = jnp.any(~own_team & (state.bids <= 0) & (state.tricks > 0))
    our_need = jnp.where(own_failed, 0, our_need)
    their_need = jnp.where(their_failed, 0, their_need)
    partner_nil = live_nil[partner] & ~own_failed
    own_nil = live_nil[me] & ~own_failed
    opposing_nil_winning = jnp.any((~own_team) & live_nil & (seats == current_winner)) & ~their_failed
    opponent_void_count = jnp.sum(state.voids & (~own_team)[:, None], axis=0)
    partner_void = state.voids[partner]
    return dict(rank=rank, suits=suits, lengths=lengths, leading=leading,
        wins=wins, partner_winning=partner_winning, master=master,
        our_need=our_need, their_need=their_need, remaining=remaining,
        own_nil=own_nil, partner_nil=partner_nil, opposing_nil_winning=opposing_nil_winning,
        opponent_void_count=opponent_void_count, partner_void=partner_void,
        score_gap=state.scores[team] - state.scores[1 - team],
        own_score=state.scores[team], opponent_score=state.scores[1 - team],
        own_bags=state.bags[team],
        me=me, partner=partner, team=team, trick_pos=state.trick_pos)


def _diversity_trump_management(f, p):
    """Draw trump from long holdings; conserve high trump; cheap ruffing."""
    trump = f['suits'] == 3
    need = f['our_need'] > 0
    draw = f['leading'] & (f['lengths'][3] >= 4) & need
    # Targeted trump leads differ from a universal high/low-card endpoint.
    score = (2.2 + 1.4 * p) * f['wins'] * need
    score -= (0.35 + 0.9 * p) * f['rank'] + (1.1 + p) * trump
    score += draw * trump * (1.4 + 1.8 * f['master'])
    score -= (~f['leading']) * f['partner_winning'] * f['wins'] * (2.5 + p)
    score += (~f['leading']) * trump * f['wins'] * need * (1.0 - f['rank'])
    return score


def _diversity_void_building(f, p):
    """Clear short side suits to create ruffs; prefer low entries to long suits."""
    side = f['suits'] != 3
    singleton = (f['lengths'][f['suits']] == 1) & side
    short = jnp.maximum(4 - f['lengths'][f['suits']], 0).astype(jnp.float32)
    can_ruff = f['lengths'][3] >= 2
    score = (1.4 + p) * f['wins'] * (f['our_need'] > 0) - 0.3 * f['rank']
    score += can_ruff * side * (0.35 + 0.7 * p) * short
    score += singleton * can_ruff * (1.2 + p)
    score += f['leading'] * side * (~singleton) * (0.12 * f['lengths'][f['suits']] - 0.8 * f['rank'])
    score -= (f['suits'] == 3) * (0.8 + 0.6 * p)
    score -= (~f['leading']) * f['partner_winning'] * f['wins'] * 2.4
    return score


def _diversity_tempo_ducking(f, p):
    """Duck early seats; take a cheap last-seat trick when the contract needs it."""
    last = f['trick_pos'] == 3
    pressure = f['our_need'] / f['remaining']
    take = (f['our_need'] > 0) & (last | (pressure > 0.55 + 0.35 * p))
    score = f['wins'] * jnp.where(take, 3.0 + p, -1.5 - 2.0 * p)
    score -= (0.4 + 0.5 * p) * f['rank'] + 0.5 * (f['suits'] == 3)
    score += f['leading'] * (0.24 * f['lengths'][f['suits']] - (0.9 + p) * f['rank'])
    score -= f['partner_winning'] * f['wins'] * (2.0 + p)
    return score


def _diversity_crossruff_pressure(f, p):
    """HELDOUT: public partner-void leads and opponent-void avoidance interact."""
    side = f['suits'] != 3
    partner_ruff_entry = side & f['partner_void'][f['suits']]
    safe_side = side & (f['opponent_void_count'][f['suits']] == 0)
    score = (2.0 + p) * f['wins'] * (f['our_need'] > 0) - 0.45 * f['rank']
    score += f['leading'] * partner_ruff_entry * (2.8 + 2.2 * p) * (1.0 - 0.5 * f['rank'])
    score += f['leading'] * safe_side * f['master'] * (1.4 + p)
    score -= f['leading'] * side * f['opponent_void_count'][f['suits']] * (0.6 + p)
    score += (~f['leading']) * (f['suits'] == 3) * f['wins'] * (1.0 - f['rank']) * (1.8 + p)
    score -= f['partner_winning'] * f['wins'] * 3.0
    return score


def _diversity_score_threshold_defense(f, p):
    """HELDOUT: score/bag threshold switches between denial and bag avoidance."""
    opponent_close = f['opponent_score'] >= 300.0 + 100.0 * p
    own_close = f['own_score'] >= 330.0 + 80.0 * p
    losing = f['score_gap'] < -(20.0 + 80.0 * p)
    bag_danger = f['own_bags'] >= 5.0 + 3.0 * p
    deny = (f['their_need'] > 0) & (opponent_close | losing)
    take = (f['our_need'] > 0) | deny
    avoid = (~take) & (bag_danger | own_close)
    score = f['wins'] * jnp.where(take, 2.5 + 2.0 * p, jnp.where(avoid, -4.0, -0.5))
    # Cash public masters only under score pressure; otherwise save high cards.
    score += f['leading'] * f['master'] * deny * (2.0 + p)
    score -= (0.3 + p) * f['rank'] + 0.45 * (f['suits'] == 3)
    score -= f['partner_winning'] * f['wins'] * (1.5 + 2.0 * p)
    return score


def _diversity_blind_action(state, key, family, variant):
    """No legal_mask/hand/private-card read in blind polling."""
    me = actor(state)
    team = me % 2
    p = (variant.astype(jnp.float32) + 0.5) / 64.
    deficit = state.scores[1 - team] - state.scores[team]
    base = jnp.array([0.015, 0.025, 0.008, 0.04, 0.01], jnp.float32)[family]
    comeback = (deficit >= 180) & (state.scores[1 - team] >= 350)
    chance = base + (0.04 + 0.08 * p) * comeback
    chance *= jnp.where(state.bids[(me + 2) % 4] == -1, 0.15, 1.)
    declare = state.eligible[team] & (jax.random.uniform(key) < chance)
    return jnp.where(declare, 67, 66).astype(jnp.int32)


def _diversity_pass_action(state, key, family, variant, legal):
    hand = _diversity_acting_hand(state)
    me = actor(state)
    ids = jnp.arange(52)
    ranks, suits = ids % 13, ids // 13
    lengths = hand.reshape(4, 13).sum(axis=1)
    p = (variant.astype(jnp.float32) + 0.5) / 64.
    export_danger = ranks.astype(jnp.float32) + (5.0 + 2.0 * p) * (suits == 3)
    export_danger += (lengths[suits] <= 2) * (ranks >= 9) * (1.0 + p)
    support_gift = -ranks.astype(jnp.float32) - (0.2 + p) * (suits == 3)
    support_gift += (family == 1) * (lengths[suits] == 1) * 0.2
    score = jnp.where(state.bids[me] == -1, export_danger, support_gift)
    score += 0.015 * jax.random.uniform(key, (52,))
    chosen = jnp.argmax(jnp.where(legal[:52], score, -1.e9))
    return jnp.where(jnp.any(legal[:52]), chosen, 66).astype(jnp.int32)


def _diversity_bid_action(state, key, family, variant, legal):
    me = actor(state)
    estimate, nil_risk = estimate_bid_and_nil_risk(_diversity_acting_hand(state))
    parameters = diversity_style_parameters(family, variant)
    bias = parameters['bid_bias']
    bid = jnp.clip(jnp.floor(estimate + 0.35 + bias), 1, 13).astype(jnp.int32)
    partner_bid = state.bids[(me + 2) % 4]
    bid = jnp.minimum(bid, jnp.maximum(1, 13 - jnp.maximum(partner_bid, 0)))
    threshold = 0.75 * parameters['risk_scale']
    threshold *= jnp.where((partner_bid == 0) | (partner_bid == -1), 0.25, 1.)
    chosen = jnp.where(nil_risk < threshold, 52, 52 + bid)
    return jnp.where(legal[66], 66, chosen).astype(jnp.int32)


def _diversity_play_action(state, key, family, variant, legal):
    f = _diversity_public_play_features(state)
    p = (variant.astype(jnp.float32) + 0.5) / 64.
    score = jax.lax.switch(family, (_diversity_trump_management, _diversity_void_building,
        _diversity_tempo_ducking, _diversity_crossruff_pressure,
        _diversity_score_threshold_defense), f, p)
    parameters = diversity_style_parameters(family, variant)
    score += (parameters['urgency_scale'] - 1.0) * f['wins'] * (f['our_need'] > 0)
    # Shared contract safeguards use only public bids/tricks/table and our hand.
    score += f['partner_nil'] * f['wins'] * (5.5 + p)
    score += f['opposing_nil_winning'] * (~f['wins']) * 5.0
    danger = f['rank'] + 0.65 * (f['suits'] == 3) + 0.25 * f['master']
    nil_score = -9.0 * f['wins'] + (1.1 + 0.5 * p) * danger * (~f['wins'])
    # A lead provisionally wins regardless of rank. Score its exposure directly
    # so live nil leads do not collapse to a constant followed by tie noise.
    nil_lead_score = -3.0 * f['rank'] - 2.0 * (f['suits'] == 3) - 0.5 * f['master']
    nil_score = jnp.where(f['leading'], nil_lead_score, nil_score)
    score = jnp.where(f['own_nil'], nil_score, score)
    score += 0.025 * jax.random.uniform(key, (52,))
    return jnp.argmax(jnp.where(legal[:52], score, -1.e9)).astype(jnp.int32)


def _diversity_nonblind_action(state, key, family, variant):
    legal = legal_mask(state)
    chosen = jax.lax.switch(jnp.maximum(phase(state) - 1, 0),
        (_diversity_pass_action, _diversity_bid_action, _diversity_play_action),
        state, key, family, variant, legal)
    return jnp.where(legal[chosen], chosen, jnp.argmax(legal)).astype(jnp.int32)


def diverse_heuristic_action(state, key, style):
    """Exact legacy delegation or legal new action; invalid IDs yield-1.

    Invalid IDs are a fail-fast sentinel, never silently clamped to a valid
    opponent. Native/runtime boundaries must reject them before workloads.
    """
    family, variant, valid = decode_opponent_style_id(style)
    def new_action(_):
        return jax.lax.cond(phase(state) == 0, _diversity_blind_action,
                           _diversity_nonblind_action, state, key, family, variant)
    chosen = jax.lax.cond(family == -1,
        lambda _: heuristic_action(state, key, style), new_action, operand=None)
    return jnp.where(valid, chosen, jnp.int32(-1)).astype(jnp.int32)
