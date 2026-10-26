# Pure JAX rules engine. Card IDs: clubs, diamonds, hearts, spades; 2 through ace.
from typing import NamedTuple

N_ACTIONS = 68
NOOP, DOUBLE_NIL = 66, 67
BLIND, PASS, BID, PLAY = 0, 1, 2, 3
GAMMA_HAND = 0.997
CARD_IDS = jnp.arange(52, dtype=jnp.int32)
CARD_SUITS, CARD_RANKS = CARD_IDS // 13, CARD_IDS % 13
TEAM_PLAYERS = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)


class State(NamedTuple):
    hands: jax.Array
    bids: jax.Array
    tricks: jax.Array
    scores: jax.Array
    bags: jax.Array
    dealer: jax.Array
    turn: jax.Array
    current_player: jax.Array
    table: jax.Array
    table_players: jax.Array
    trick_pos: jax.Array
    leader: jax.Array
    played_by: jax.Array
    played_at: jax.Array
    voids: jax.Array
    pending_pass: jax.Array
    passed: jax.Array
    received: jax.Array
    eligible: jax.Array
    tiebreak: jax.Array
    hand_number: jax.Array


def _new_hand(key, scores, bags, dealer, tiebreak, hand_number):
    # Interleave the shuffled deck clockwise, beginning left of the dealer.
    cards = jax.random.permutation(key, 52)
    owners = (dealer + 1 + jnp.arange(52, dtype=jnp.int32)) % 4
    hands = jnp.zeros((4, 52), dtype=jnp.bool_).at[owners, cards].set(True)
    first = (dealer + 1) % 4
    return State(
        hands=hands, bids=jnp.full(4, -2, jnp.int32),
        tricks=jnp.zeros(4, jnp.int32), scores=scores, bags=bags,
        dealer=dealer, turn=jnp.int32(0), current_player=first,
        table=jnp.full(4, -1, jnp.int32),
        table_players=jnp.full(4, -1, jnp.int32), trick_pos=jnp.int32(0),
        leader=first, played_by=jnp.full(52, -1, jnp.int32),
        played_at=jnp.full(52, -1, jnp.int32),
        voids=jnp.zeros((4, 4), jnp.bool_),
        pending_pass=jnp.full(4, -1, jnp.int32),
        passed=jnp.full(4, -1, jnp.int32),
        received=jnp.full(4, -1, jnp.int32),
        eligible=scores[::-1] - scores >= 150,
        tiebreak=tiebreak, hand_number=hand_number,
    )


def reset(key):
    """Start a fresh match, selecting a dealer uniformly at random."""
    dealer_key, deal_key = jax.random.split(key)
    return _new_hand(deal_key, jnp.zeros(2, jnp.int32),
                     jnp.zeros(2, jnp.int32),
                     jax.random.randint(dealer_key, (), 0, 4, dtype=jnp.int32),
                     jnp.int32(0), jnp.int32(0))


def deal_next(state, key):
    """Deal the next hand; carry scores/bags and rotate the dealer clockwise."""
    return _new_hand(key, state.scores, state.bags, (state.dealer + 1) % 4,
                     state.tiebreak, state.hand_number + 1)


def phase(state):
    return jnp.minimum(state.turn // 4, PLAY)


def actor(state):
    return jnp.where(state.turn < 12,
                     (state.dealer + 1 + state.turn % 4) % 4,
                     state.current_player)


def legal_mask(state):
    """Exact legal actions, including forced no-ops in the fixed 64-turn schedule."""
    p, ph = actor(state), phase(state)
    hand = state.hands[p]
    passing = (state.bids[p] == -1) | (state.bids[(p + 2) % 4] == -1)
    led_suit = jnp.maximum(state.table[0], 0) // 13
    follows = CARD_SUITS == led_suit
    must_follow = (state.trick_pos > 0) & jnp.any(hand & follows)
    play_cards = hand & (~must_follow | follows)
    card_actions = ((ph == PASS) & passing & hand) | ((ph == PLAY) & play_cards)
    bid_actions = jnp.full(14, (ph == BID) & (state.bids[p] != -1))
    decline = ((ph == BLIND) | ((ph == PASS) & ~passing)
               | ((ph == BID) & (state.bids[p] == -1)))
    blind = (ph == BLIND) & state.eligible[p % 2]
    return jnp.concatenate((card_actions, bid_actions,
                            jnp.array([decline, blind], dtype=jnp.bool_)))


def score_hand(bids, tricks, bags):
    """Score both complete contracts, then apply each cumulative ten-bag penalty.

    A failed nil invalidates the ENTIRE partnership contract. Its partner's
    extra tricks neither rescue the contract nor add bags.
    """
    b, t = bids[TEAM_PLAYERS], tricks[TEAM_PLAYERS]
    normal = b > 0
    both_normal = jnp.all(normal, axis=1)
    bid_sum, trick_sum = jnp.sum(b, axis=1), jnp.sum(t, axis=1)
    ordinary_success = trick_sum >= bid_sum
    individual_success = jnp.all(jnp.where(normal, t >= b, t == 0), axis=1)
    success = jnp.where(both_normal, ordinary_success, individual_success)
    value = jnp.sum(jnp.where(normal, 10 * b, jnp.where(b == -1, 200, 100)), axis=1)
    excess = jnp.where(both_normal, trick_sum - bid_sum,
                       jnp.sum(jnp.where(normal, t - b, 0), axis=1))
    bags_added = jnp.where(success, jnp.maximum(excess, 0), 0)
    total_bags = bags + bags_added
    delta = jnp.where(success, value + bags_added, -value) - 100 * (total_bags // 10)
    return delta, total_bags % 10, success, bags_added


def resolve_match(scores, tiebreak):
    """Return (done, winner, next_tiebreak); winner 2 denotes the ten-hand draw."""
    tied = scores[0] == scores[1]
    target_reached = jnp.any(scores >= 500)
    active = (tiebreak > 0) | target_reached
    draw = (tiebreak >= 10) & tied
    done = active & (~tied | draw)
    winner = jnp.where(done, jnp.where(tied, 2, jnp.argmax(scores)), -1).astype(jnp.int32)
    next_tiebreak = jnp.where(active & tied & ~done, tiebreak + 1, tiebreak)
    return done, winner, next_tiebreak


def _blind_turn(state, action):
    p = actor(state)
    bids = state.bids.at[p].set(jnp.where(action == DOUBLE_NIL, -1, state.bids[p]))
    return state._replace(bids=bids, turn=state.turn + 1)


def _pass_turn(state, action):
    p = actor(state)
    pending = state.pending_pass.at[p].set(jnp.where(action < 52, action, -1))
    state = state._replace(pending_pass=pending, turn=state.turn + 1)

    def exchange(s):
        received = s.pending_pass[jnp.array([2, 3, 0, 1])]
        outgoing = jax.nn.one_hot(s.pending_pass, 52, dtype=jnp.bool_)
        incoming = jax.nn.one_hot(received, 52, dtype=jnp.bool_)
        return s._replace(hands=(s.hands & ~outgoing) | incoming,
                          passed=s.pending_pass, received=received,
                          pending_pass=jnp.full(4, -1, jnp.int32))

    # Complete all exchanges only after all selections have been collected.
    return jax.lax.cond(state.turn == 8, exchange, lambda s: s, state)


def _bid_turn(state, action):
    p = actor(state)
    bids = state.bids.at[p].set(jnp.where(action == NOOP, state.bids[p], action - 52))
    return state._replace(bids=bids, turn=state.turn + 1)


def trick_winner(table, table_players):
    """Winner of a complete trick; only led-suit cards or trump can win."""
    suits, ranks = table // 13, table % 13
    strength = jnp.where(suits == 3, 26 + ranks,
                         jnp.where(suits == suits[0], 13 + ranks, -1))
    return table_players[jnp.argmax(strength)]


def _play_turn(state, action):
    p, pos = state.current_player, state.trick_pos
    table = state.table.at[pos].set(action)
    players = state.table_players.at[pos].set(p)
    led_suit = jnp.maximum(table[0], 0) // 13
    is_void = (pos > 0) & (action // 13 != led_suit)
    voids = state.voids.at[p, led_suit].set(state.voids[p, led_suit] | is_void)
    end_trick = pos == 3
    winner = trick_winner(table, players)
    tricks = state.tricks.at[jnp.maximum(winner, 0)].add(end_trick.astype(jnp.int32))
    next_player = jnp.where(end_trick, winner, (p + 1) % 4)
    return state._replace(
        hands=state.hands.at[p, action].set(False), tricks=tricks,
        turn=state.turn + 1, current_player=next_player,
        table=jnp.where(end_trick, -1, table),
        table_players=jnp.where(end_trick, -1, players),
        trick_pos=jnp.where(end_trick, 0, pos + 1),
        leader=jnp.where(end_trick, winner, state.leader),
        played_by=state.played_by.at[action].set(p),
        played_at=state.played_at.at[action].set(state.turn - 12), voids=voids,
    )


def _potential(scores):
    return 0.5 * jnp.tanh((scores[0] - scores[1]).astype(jnp.float32) / 500.0)


def _complete_hand(before, progressed, valid, key):
    delta, bags, success, added = score_hand(progressed.bids, progressed.tricks,
                                            progressed.bags)
    scores = progressed.scores + delta
    done, winner, tiebreak = resolve_match(scores, progressed.tiebreak)
    scored = progressed._replace(scores=scores, bags=bags, tiebreak=tiebreak)
    result_reward = jnp.where(winner == 0, 1.0, jnp.where(winner == 1, -1.0, 0.0))
    reward = (result_reward + jnp.where(done, 0.0, GAMMA_HAND * _potential(scores))
              - _potential(before.scores))
    next_state = jax.lax.cond(done, lambda _: reset(key),
                              lambda _: deal_next(scored, key), operand=None)
    info = dict(
        discount=jnp.where(done, 0.0, GAMMA_HAND).astype(jnp.float32),
        hand_done=jnp.bool_(True), match_done=done, winner=winner,
        score_delta=delta, contract_success=success,
        nil_attempts=jnp.sum(scored.bids == 0),
        nil_successes=jnp.sum((scored.bids == 0) & (scored.tricks == 0)),
        dn_attempts=jnp.sum(scored.bids == -1),
        dn_successes=jnp.sum((scored.bids == -1) & (scored.tricks == 0)),
        bags_added=added, illegal=~valid, final_scores=scores,
    )
    return next_state, reward.astype(jnp.float32), done, info


def _advance_result(progressed, valid):
    info = dict(
        discount=jnp.float32(1.0), hand_done=jnp.bool_(False),
        match_done=jnp.bool_(False), winner=jnp.int32(-1),
        score_delta=jnp.zeros(2, jnp.int32),
        contract_success=jnp.zeros(2, jnp.bool_),
        nil_attempts=jnp.int32(0), nil_successes=jnp.int32(0),
        dn_attempts=jnp.int32(0), dn_successes=jnp.int32(0),
        bags_added=jnp.zeros(2, jnp.int32), illegal=~valid,
        final_scores=progressed.scores,
    )
    return progressed, jnp.float32(0.0), jnp.bool_(False), info


def step(state, action, key):
    """One turn, automatically resetting hands and completed matches.

    Returns (next_state, team0_reward, match_done, info). A supplied illegal
    action falls back to the first legal action and sets info['illegal']; use
    checked_step in interactive code to raise instead. Training always masks
    logits, so it should never exercise this fallback.
    """
    action = jnp.asarray(action, dtype=jnp.int32)
    mask = legal_mask(state)
    in_range = (action >= 0) & (action < N_ACTIONS)
    valid = in_range & mask[jnp.clip(action, 0, N_ACTIONS - 1)]
    safe_action = jnp.where(valid, action, jnp.argmax(mask)).astype(jnp.int32)
    progressed = jax.lax.switch(phase(state),
                               (_blind_turn, _pass_turn, _bid_turn, _play_turn),
                               state, safe_action)
    return jax.lax.cond(state.turn == 63,
                        lambda _: _complete_hand(state, progressed, valid, key),
                        lambda _: _advance_result(progressed, valid), operand=None)


def batch_step(states, actions, keys):
    """Efficient synchronized batch step for training.

    All environments MUST share their scheduled turn (reset them together).
    Every hand is exactly 64 turns, including terminal hands, so normal rollout
    collection preserves this invariant. The whole-batch branch avoids
    shuffling future hands on every ordinary move, which plain vmap(step)
    would do because JAX converts batched conditionals into selections.
    Use vmap(step) for asynchronous states, such as search rollouts.
    """
    actions = jnp.asarray(actions, dtype=jnp.int32)
    masks = jax.vmap(legal_mask)(states)
    clipped = jnp.clip(actions, 0, N_ACTIONS - 1)
    valid = ((actions >= 0) & (actions < N_ACTIONS)
              & jnp.take_along_axis(masks, clipped[:, None], axis=1)[:, 0])
    safe = jnp.where(valid, actions, jnp.argmax(masks, axis=1)).astype(jnp.int32)
    progressed = jax.lax.switch(
        jnp.minimum(states.turn[0] // 4, PLAY),
        tuple(jax.vmap(fn) for fn in (_blind_turn, _pass_turn, _bid_turn, _play_turn)),
        states, safe,
    )
    return jax.lax.cond(
        states.turn[0] == 63,
        lambda _: jax.vmap(_complete_hand)(states, progressed, valid, keys),
        lambda _: jax.vmap(_advance_result)(progressed, valid), operand=None,
    )


def checked_step(state, action, key):
    """Host-side strict action validation for human play/debugging."""
    a = int(action)
    if not 0 <= a < N_ACTIONS or not bool(np.asarray(legal_mask(state))[a]):
        raise ValueError(f'Illegal action {a}; legal actions: '
                         f'{np.flatnonzero(np.asarray(legal_mask(state))).tolist()}')
    return step(state, a, key)


def _observe_for(state, viewer):
    """Actor-relative information set; no other hand or sealed selection is read."""
    players = (viewer + jnp.arange(4, dtype=jnp.int32)) % 4
    teams = (viewer % 2 + jnp.arange(2, dtype=jnp.int32)) % 2
    visible = state.turn >= 4
    exchanged = state.turn >= 8
    hand = (state.hands[viewer] & visible).astype(jnp.float32)
    owner_relative = (state.played_by - viewer) % 4
    owners = jax.nn.one_hot(owner_relative, 4) * (state.played_by >= 0)[:, None]
    table_relative = (state.table_players - viewer) % 4
    table_owners = (jax.nn.one_hot(table_relative, 4)
                    * (state.table_players >= 0)[:, None])
    pieces = [
        hand,
        jax.nn.one_hot(phase(state), 4),
        jnp.array([state.turn / 63.0]),
        jax.nn.one_hot((state.dealer - viewer) % 4, 4),
        jax.nn.one_hot((state.leader - viewer) % 4, 4),
        jax.nn.one_hot(state.bids[players] + 2, 16).reshape(-1),
        state.tricks[players] / 13.0,
        state.scores[teams] / 500.0,
        state.bags[teams] / 10.0,
        state.eligible[teams].astype(jnp.float32),
        jnp.array([state.tiebreak / 10.0, state.tiebreak > 0], dtype=jnp.float32),
        jax.nn.one_hot(state.table, 52).reshape(-1),
        table_owners.reshape(-1),
        jax.nn.one_hot(state.trick_pos, 4),
        owners.reshape(-1),
        (state.played_at + 1) / 52.0,
        state.voids[players].astype(jnp.float32).reshape(-1),
        jax.nn.one_hot(state.passed[viewer], 52) * exchanged,
        jax.nn.one_hot(state.received[viewer], 52) * exchanged,
        hand.reshape(4, 13).sum(axis=1) / 13.0,
    ]
    return jnp.concatenate(pieces).astype(jnp.float32)


def observe(state):
    """Policy observation: current player's hand and public information only.

    During BLIND the entire hand is hidden. During PASS, teammates' sealed
    selections remain hidden; own sent/received cards appear after exchange.
    Exact public card ownership and chronological order preserve the complete
    card-play history for inference about the remaining hidden hands.
    """
    return _observe_for(state, actor(state))


def critic_observe(state):
    """Privileged, fixed team-0 view for the separate training-only value net."""
    return jnp.concatenate((
        _observe_for(state, jnp.int32(0)),
        state.hands.astype(jnp.float32).reshape(-1),
        jax.nn.one_hot(state.pending_pass, 52).reshape(-1),
        jax.nn.one_hot(actor(state), 4),
    )).astype(jnp.float32)


OBS_DIM = int(jax.eval_shape(observe, reset(jax.random.PRNGKey(0))).shape[0])
CRITIC_DIM = int(jax.eval_shape(critic_observe, reset(jax.random.PRNGKey(0))).shape[0])
CARD_RANK_NAMES = tuple('23456789TJQKA')
CARD_SUIT_NAMES = ('♣', '♦', '♥', '♠')


def card_name(card):
    card = int(card)
    return '—' if card < 0 else CARD_RANK_NAMES[card % 13] + CARD_SUIT_NAMES[card // 13]


def action_name(action):
    action = int(action)
    if action < 52:
        return card_name(action)
    if action < 66:
        return 'Nil' if action == 52 else f'Bid {action - 52}'
    return 'Decline / skip' if action == NOOP else 'Double Nil'

print(f'Rules engine ready: {OBS_DIM} policy features, {CRITIC_DIM} critic features, '
      f'{N_ACTIONS} actions, 64 scheduled turns/hand.')


# Sequential public blind-nil responses, version1.
# No new State fields or policy features. Declines are derived from polling order.
BN_PENDING, BN_DECLINED, BN_DECLARED = 0, 1, 2


def blind_decision_status(state):
    """Absolute seats: pending, declined BN, or announced BN; no hands read.

    Every seat is polled once clockwise, left of the dealer first. A response
    becomes public only after that seat's turn. Ordinary nil is not blind nil.
    """
    seats = jnp.arange(4, dtype=jnp.int32)
    polling_position = (seats - state.dealer - 1) % 4
    polled = polling_position < jnp.minimum(state.turn, 4)
    return jnp.where(polled,
        jnp.where(state.bids == -1, BN_DECLARED, BN_DECLINED),
        BN_PENDING).astype(jnp.int32)


def blind_decisions_from_observation(observation):
    """Decode explicit actor-relative BN status from the existing753 features.

    Supports a single observation or batches. This accessor is not an extra
    neural-network input: turn/dealer already distinguish pending from decline.
    """
    if observation.shape[-1] != 753:
        raise ValueError("Expected the unchanged753-feature policy observation")
    turn = jnp.rint(observation[..., 56] * 63.).astype(jnp.int32)
    relative_dealer = jnp.argmax(observation[..., 57:61], axis=-1)
    polling_position = (jnp.arange(4, dtype=jnp.int32) - relative_dealer[..., None] - 1) % 4
    polled = polling_position < jnp.minimum(turn[..., None], 4)
    encoded_bids = observation[..., 65:129].reshape(observation.shape[:-1] + (4, 16))
    return jnp.where(polled,
        jnp.where(encoded_bids[..., 1] > .5, BN_DECLARED, BN_DECLINED),
        BN_PENDING).astype(jnp.int32)


def _blind_turn(state, action):
    # Only the current seat is committed; an earlier response cannot be revised.
    # Explicitly clear the current pending bid on decline, rather than retaining
    # any accidentally prefilled future declaration in malformed input state.
    player = actor(state)
    bid = jnp.where(action == DOUBLE_NIL, -1, -2)
    can_respond = (state.turn >= 0) & (state.turn < 4)
    bids = jnp.where(can_respond, state.bids.at[player].set(bid), state.bids)
    return state._replace(bids=bids, turn=state.turn + can_respond.astype(jnp.int32))


_BN_BASE_OBSERVE_FOR = _observe_for


def _observe_for(state, viewer):
    # Unpolled declarations must not be visible, even if an invalid caller
    # accidentally prefilled future bids. Reachable game observations are
    # byte-identical to the original policy/critic observations in every phase.
    public_blind_bids = jnp.where(blind_decision_status(state) == BN_DECLARED, -1, -2)
    public_bids = jnp.where(state.turn < 4, public_blind_bids, state.bids)
    return _BN_BASE_OBSERVE_FOR(state._replace(bids=public_bids), viewer)


_observe_for._sequential_bn_visibility_v1 = True


def observe(state):
    """Current actor's public information, with blind hands still masked."""
    # Fresh callable identity prevents reusing the original eval_shape trace.
    return _observe_for(state, actor(state))


def critic_observe(state):
    """Unchanged privileged value input, separate from the policy tower."""
    return jnp.concatenate((
        _observe_for(state, jnp.int32(0)),
        state.hands.astype(jnp.float32).reshape(-1),
        jax.nn.one_hot(state.pending_pass, 52).reshape(-1),
        jax.nn.one_hot(actor(state), 4),
    )).astype(jnp.float32)
