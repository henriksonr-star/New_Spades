"""Append-only training policies; execute after the exact frozen cell8 source.

The original thirteen policies and all five diversity-v1 families remain held
by their original function objects. New features are isolated here; old style
IDs call the captured dispatcher. This module has no I/O, imports or search.
"""

EXPANSION_OPPONENT_REVISION = 'spades-training-opponent-expansion-v2'
EXPANSION_PROFILE_KEY = 'opponent_training_profile'
EXPANSION_PROFILE_VALUE = 'training_expansion_v2'
EXPANSION_TRAIN_FAMILIES = ('sequence_unblocking', 'trump_promotion', 'gap_finessing')
EXPANSION_TRAIN_FAMILY_INDICES = (5, 6, 7)
EXPANSION_NEW_TRAIN_STYLE_IDS = tuple(
    13 + family * 64 + variant for family in (5, 6, 7) for variant in range(32))
EXPANSION_TRAIN_STYLE_IDS = DIVERSITY_TRAIN_STYLE_IDS + EXPANSION_NEW_TRAIN_STYLE_IDS
EXPANSION_HOLDOUT_STYLE_IDS = DIVERSITY_HOLDOUT_STYLE_IDS
EXPANSION_MAX_STYLE_ID = 492

_expansion_v1_action = diverse_heuristic_action
_expansion_v1_decode = decode_opponent_style_id
_expansion_v1_is_train = is_train_opponent_style_id
_expansion_v1_is_holdout = is_holdout_opponent_style_id
_expansion_v1_sampler = sample_train_opponent_style_ids

EXPANSION_OPPONENT_MANIFEST = {
    'revision': EXPANSION_OPPONENT_REVISION,
    'base_revision': DIVERSITY_OPPONENT_REVISION,
    'profile_key': EXPANSION_PROFILE_KEY, 'profile_value': EXPANSION_PROFILE_VALUE,
    'encoding': {'base': 13, 'family_stride': 64, 'maximum_id': 492},
    'legacy_ids': list(range(13)),
    'inherited_train_style_ids': list(DIVERSITY_TRAIN_STYLE_IDS),
    'new_train_style_ids': list(EXPANSION_NEW_TRAIN_STYLE_IDS),
    'train_style_ids': list(EXPANSION_TRAIN_STYLE_IDS),
    'holdout_style_ids': list(EXPANSION_HOLDOUT_STYLE_IDS),
    'inherited_train_families': list(DIVERSITY_TRAIN_FAMILIES),
    'new_train_families': list(EXPANSION_TRAIN_FAMILIES),
    'train_families': list(DIVERSITY_TRAIN_FAMILIES) + list(EXPANSION_TRAIN_FAMILIES),
    'train_family_indices': [0, 1, 2, 5, 6, 7],
    'new_train_family_indices': [5, 6, 7],
    'holdout_families': list(DIVERSITY_HOLDOUT_FAMILIES),
    'holdout_family_indices': [3, 4],
    'train_variants': list(range(32)), 'holdout_variants': list(range(32, 64)),
    'train_style_count': 205, 'new_train_style_count': 96,
    'holdout_style_count': 64, 'valid_style_count': 269,
    'parameter': 'q=(variant+0.5)/32 for new families; inherited parameters unchanged',
    'new_parameter_bands': {'q': [0.015625, 0.984375],
        'bid_bias': [-0.484375, 0.484375],
        'sequence_exchange': [0.0625, 3.9375],
        'promotion_exchange': [0.078125, 4.921875],
        'finesse_exchange': [0.0625, 3.9375]},
    'family_rules': {
        'sequence_unblocking': 'Lead connected side-suit honor sequences; release a short honor behind a winning partner to unblock that suit.',
        'trump_promotion': 'Force the sole outstanding higher trump with a backed high trump; retain master trump while exiting an exhausted side suit late.',
        'gap_finessing': 'Lead below gapped side-suit honors; choose an interior honor in third seat when one unseen higher card remains.',
    },
    'scripted_sampling': {'legacy_probability': 0.25, 'old_train_probability': 0.25,
        'new_probability': 0.5, 'legacy_uniform_count': 13,
        'old_uniform_family_count': 3, 'new_uniform_family_count': 3,
        'uniform_variants_per_family': 32, 'assignment_scope': 'match'},
    'outer_sampling': 'Use the actual saved source cfg selfplay/snapshot probabilities unchanged.',
    'preservation': 'All 173 inherited valid IDs delegate captured v1 callable; functions, IDs, parameters and heldout memberships unchanged.',
    'invalid_action': -1,
    'information': 'Acting hand plus public bids/tricks/scores/bags/table/played ownership/voids; blind uses no hands or legal mask.',
    'search': 'none; bounded 52x13 feature comparisons, no lookahead or neural inference',
    'strength': 'unmeasured; additional tactical policies do not imply stronger opponents',
}


def decode_opponent_style_id(style):
    """Extend valid IDs only with family5..7, variants0..31."""
    family, variant, inherited_valid = _expansion_v1_decode(style)
    raw = jnp.asarray(style)
    new_valid = (jnp.issubdtype(raw.dtype, jnp.integer) &
                 (family >= 5) & (family <= 7) & (variant < 32))
    return family, variant, inherited_valid | new_valid


def is_train_opponent_style_id(style):
    family, variant, valid = decode_opponent_style_id(style)
    return valid & ((family == -1) | (((family >= 0) & (family < 3) |
                                     (family >= 5) & (family <= 7)) & (variant < 32)))


def is_holdout_opponent_style_id(style):
    # The captured helper resolves this extended decoder, whose old IDs retain
    # their exact tuples; its fixed family3/4 predicate excludes every new ID.
    return _expansion_v1_is_holdout(style)


def sample_train_opponent_style_ids(key, n):
    """Within SCRIPTED: 25% original13, 25% old3x32, 50% new3x32."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('Static nonnegative Python integer population size required.')
    mix_key, legacy_key, family_key, variant_key = jax.random.split(key, 4)
    group = jax.random.uniform(mix_key, (n,))
    legacy = jax.random.randint(legacy_key, (n,), 0, 13, dtype=jnp.int32)
    family = jax.random.randint(family_key, (n,), 0, 3, dtype=jnp.int32)
    variant = jax.random.randint(variant_key, (n,), 0, 32, dtype=jnp.int32)
    family = family + jnp.where(group >= 0.5, 5, 0)
    encoded = 13 + 64 * family + variant
    return jnp.where(group < 0.25, legacy, encoded).astype(jnp.int32)


def expansion_style_parameters(variant):
    """Thirty-two finite profiles vary tactical exchanges, bids and nil risk."""
    q = (variant.astype(jnp.float32) + 0.5) / 32.
    return dict(q=q, bid_bias=q - 0.5, nil_risk_scale=0.7 + 0.6 * q,
                sequence_exchange=4.0 * q, promotion_exchange=5.0 * q,
                finesse_exchange=4.0 * q)


def _expansion_acting_hand(state):
    """The single private-card read for the additional policies."""
    return state.hands[actor(state)]


def _expansion_public_features(state):
    """Independent feature path: own sequences and public outstanding ranks.

    Every candidate is compared with the thirteen ranks in its suit. No deal
    completion, hidden-hand inference oracle, or simulated action is evaluated.
    """
    me = actor(state)
    partner = (me + 2) % 4
    ids = jnp.arange(52, dtype=jnp.int32)
    suits, ranks = ids // 13, ids % 13
    hand = _expansion_acting_hand(state)
    own = hand.reshape(4, 13)
    lengths = own.sum(axis=1)
    unseen = ((state.played_by < 0) & ~hand).reshape(4, 13)
    rank_grid = jnp.arange(13, dtype=jnp.int32)[None, :]
    above = rank_grid > ranks[:, None]
    below = rank_grid < ranks[:, None]
    higher_unseen = jnp.sum(unseen[suits] & above, axis=1)
    lower_own = jnp.sum(own[suits] & below, axis=1)
    higher_own = jnp.sum(own[suits] & above, axis=1)
    below_neighbor = hand[suits * 13 + jnp.maximum(ranks - 1, 0)] & (ranks > 0)
    above_neighbor = hand[suits * 13 + jnp.minimum(ranks + 1, 12)] & (ranks < 12)
    # An interior honor has another own honor two ranks above and a missing
    # middle rank that is still publicly outstanding.
    upper_two = hand[suits * 13 + jnp.minimum(ranks + 2, 12)] & (ranks <= 10)
    missing_middle = unseen[suits, jnp.minimum(ranks + 1, 12)] & (ranks <= 10)
    gap_honor = upper_two & missing_middle & (ranks >= 8)
    suit_has_gap = jnp.any((gap_honor & hand).reshape(4, 13), axis=1)
    leading = state.trick_pos == 0
    led = jnp.maximum(state.table[0], 0) // 13
    table = jnp.maximum(state.table, 0)
    table_strength = jnp.where(table // 13 == 3, 26 + table % 13,
        jnp.where(table // 13 == led, 13 + table % 13, -1))
    table_strength = jnp.where(state.table >= 0, table_strength, -100)
    winner = jnp.where(leading, -1, state.table_players[jnp.argmax(table_strength)])
    strength = jnp.where(suits == 3, 26 + ranks,
        jnp.where(suits == led, 13 + ranks, -1))
    wins = leading | (strength > table_strength.max())
    seats = jnp.arange(4)
    own_team = seats % 2 == me % 2
    normal = jnp.maximum(state.bids, 0)
    tricks = jnp.where(state.bids > 0, state.tricks, 0)
    our_need = jnp.maximum(jnp.sum(jnp.where(own_team, normal - tricks, 0)), 0)
    failed = jnp.any(own_team & (state.bids <= 0) & (state.tricks > 0))
    live_nil = (state.bids <= 0) & (state.tricks == 0)
    opponent_failed = jnp.any(~own_team & (state.bids <= 0) & (state.tricks > 0))
    return dict(rank=ranks.astype(jnp.float32) / 12., ranks=ranks, suits=suits,
        lengths=lengths, leading=leading, trick_pos=state.trick_pos,
        higher_unseen=higher_unseen, unseen_count=unseen.sum(axis=1),
        lower_own=lower_own, higher_own=higher_own,
        sequence_top=below_neighbor & ~above_neighbor & (ranks >= 9),
        gap_honor=gap_honor, suit_has_gap=suit_has_gap,
        master=higher_unseen == 0, wins=wins, partner_winning=winner == partner,
        own_nil=live_nil[me] & ~failed, partner_nil=live_nil[partner] & ~failed,
        opposing_nil_winning=jnp.any(~own_team & live_nil & (seats == winner)) & ~opponent_failed,
        our_need=jnp.where(failed, 0, our_need),
        remaining=jnp.maximum(13 - jnp.maximum(state.turn - 12, 0) // 4, 1),
        opponent_void_count=jnp.sum(state.voids & (~own_team)[:, None], axis=0))


def _expansion_sequence_unblocking(f, p):
    """Top-of-sequence development and partner-safe short-honor release."""
    side = f['suits'] != 3
    need = f['our_need'] > 0
    score = 2.0 * f['wins'] * need - 0.55 * f['rank'] - 0.45 * ~side
    develop = f['leading'] & side & f['sequence_top'] & (f['lengths'][f['suits']] >= 3)
    score += develop * (1.0 + p['sequence_exchange'])
    # When partner already wins, dropping a short high card removes a blockage
    # without taking the trick; this cannot be expressed as generic low/high.
    unblock = ~f['leading'] & f['partner_winning'] & ~f['wins'] & side
    unblock &= (f['lengths'][f['suits']] <= 2) & (f['ranks'] >= 9)
    score += unblock * (0.4 + p['sequence_exchange']) * f['rank']
    score -= ~f['leading'] * f['partner_winning'] * f['wins'] * 4.0
    return score


def _expansion_trump_promotion(f, p):
    """Attack a sole blocking trump with backing, then keep promoted winners."""
    trump = f['suits'] == 3
    need = f['our_need'] > 0
    score = 2.0 * f['wins'] * need - 0.6 * f['rank'] - 0.6 * trump
    # The highest held trump below exactly one public outstanding rank is a
    # promotion sacrifice only when another held trump can benefit afterward.
    promote = trump & (f['higher_unseen'] == 1) & (f['higher_own'] == 0) & (f['lower_own'] >= 1)
    attack = f['leading'] | ((f['trick_pos'] <= 2) & ~f['partner_winning'])
    score += attack * promote * need * (0.8 + p['promotion_exchange'])
    late = f['remaining'] <= 4
    exhausted_exit = ~trump & (f['unseen_count'][f['suits']] == 0)
    preserve = f['leading'] & late & (f['lengths'][3] >= 1)
    score += preserve * exhausted_exit * (0.5 + p['q'])
    score -= preserve * trump * f['master'] * (0.5 + p['q'])
    score -= ~f['leading'] * f['partner_winning'] * f['wins'] * 4.0
    return score


def _expansion_gap_finessing(f, p):
    """Lead below gapped honors; finesse with an interior honor in third seat."""
    side = f['suits'] != 3
    need = f['our_need'] > 0
    score = 1.9 * f['wins'] * need - 0.5 * f['rank'] - 0.5 * ~side
    low_entry = side & f['suit_has_gap'][f['suits']] & (f['ranks'] <= 7)
    finesse_safe = f['opponent_void_count'][f['suits']] == 0
    score += f['leading'] * low_entry * finesse_safe * (0.5 + p['finesse_exchange'])
    # The gapped lower honor wins provisionally while retaining the covering
    # higher honor. Its public higher count, not any opponent hand, gates play.
    interior = side & f['gap_honor'] & (f['higher_unseen'] == 1) & f['wins']
    score += (f['trick_pos'] == 2) * ~f['partner_winning'] * interior * (0.5 + p['finesse_exchange'])
    score -= ~f['leading'] * f['partner_winning'] * f['wins'] * 4.0
    return score


def _expansion_blind_action(state, key, family, variant):
    """Use public eligibility/scores/bids only; never call legal_mask."""
    me = actor(state)
    team = me % 2
    q = (variant.astype(jnp.float32) + 0.5) / 32.
    deficit = state.scores[1 - team] - state.scores[team]
    comeback = (deficit >= 180) & (state.scores[1 - team] >= 350)
    chance = (0.008 + 0.012 * (family - 5)) + comeback * (0.03 + 0.08 * q)
    chance *= jnp.where(state.bids[(me + 2) % 4] == -1, 0.15, 1.)
    return jnp.where(state.eligible[team] & (jax.random.uniform(key) < chance), 67, 66).astype(jnp.int32)


def _expansion_pass_action(state, key, family, variant, legal):
    me = actor(state)
    hand = _expansion_acting_hand(state)
    ids = jnp.arange(52)
    suits, ranks = ids // 13, ids % 13
    lengths = hand.reshape(4, 13).sum(axis=1)
    q = expansion_style_parameters(variant)['q']
    export = ranks.astype(jnp.float32) + (5.0 + 2.0 * q) * (suits == 3)
    gift = -ranks.astype(jnp.float32) - (0.5 + q) * (suits == 3)
    # A sequence developer preserves connected honors; a promotion player
    # preserves backed trump; a finesse player preserves gapped side honors.
    below = hand[suits * 13 + jnp.maximum(ranks - 1, 0)] & (ranks > 0)
    export -= (family == 5) * below * (ranks >= 9) * (1.0 + q)
    export -= (family == 6) * (suits == 3) * (lengths[3] >= 2) * q
    export -= (family == 7) * (suits != 3) * (ranks >= 8) * (lengths[suits] >= 3) * q
    score = jnp.where(state.bids[me] == -1, export, gift)
    score += 0.015 * jax.random.uniform(key, (52,))
    choice = jnp.argmax(jnp.where(legal[:52], score, -1.e9))
    return jnp.where(jnp.any(legal[:52]), choice, 66).astype(jnp.int32)


def _expansion_bid_action(state, key, family, variant, legal):
    me = actor(state)
    estimate, nil_risk = estimate_bid_and_nil_risk(_expansion_acting_hand(state))
    p = expansion_style_parameters(variant)
    bid = jnp.clip(jnp.floor(estimate + 0.35 + p['bid_bias']), 1, 13).astype(jnp.int32)
    partner_bid = state.bids[(me + 2) % 4]
    bid = jnp.minimum(bid, jnp.maximum(1, 13 - jnp.maximum(partner_bid, 0)))
    threshold = 0.75 * p['nil_risk_scale']
    threshold *= jnp.where(partner_bid <= 0, 0.25, 1.)
    choice = jnp.where(nil_risk < threshold, 52, 52 + bid)
    return jnp.where(legal[66], 66, choice).astype(jnp.int32)


def _expansion_play_action(state, key, family, variant, legal):
    f = _expansion_public_features(state)
    p = expansion_style_parameters(variant)
    score = jax.lax.switch(family - 5, (_expansion_sequence_unblocking,
        _expansion_trump_promotion, _expansion_gap_finessing), f, p)
    score += f['partner_nil'] * f['wins'] * 6.0
    score += f['opposing_nil_winning'] * ~f['wins'] * 5.0
    danger = f['rank'] + 0.65 * (f['suits'] == 3) + 0.25 * f['master']
    nil_score = -9.0 * f['wins'] + (1.1 + 0.5 * p['q']) * danger * ~f['wins']
    nil_lead = -3.0 * f['rank'] - 2.0 * (f['suits'] == 3) - 0.5 * f['master']
    score = jnp.where(f['own_nil'], jnp.where(f['leading'], nil_lead, nil_score), score)
    score += 0.025 * jax.random.uniform(key, (52,))
    return jnp.argmax(jnp.where(legal[:52], score, -1.e9)).astype(jnp.int32)


def _expansion_nonblind_action(state, key, family, variant):
    legal = legal_mask(state)
    choice = jax.lax.switch(jnp.maximum(phase(state) - 1, 0),
        (_expansion_pass_action, _expansion_bid_action, _expansion_play_action),
        state, key, family, variant, legal)
    return jnp.where(legal[choice], choice, jnp.argmax(legal)).astype(jnp.int32)


def diverse_heuristic_action(state, key, style):
    """Delegate every inherited ID exactly; dispatch only new family5..7."""
    family, variant, valid = decode_opponent_style_id(style)
    def extra(_):
        return jax.lax.cond(phase(state) == 0, _expansion_blind_action,
            _expansion_nonblind_action, state, key, family, variant)
    choice = jax.lax.cond((family >= 5) & (family <= 7), extra,
        lambda _: _expansion_v1_action(state, key, style), operand=None)
    return jnp.where(valid, choice, jnp.int32(-1)).astype(jnp.int32)
