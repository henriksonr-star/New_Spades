"""Passive, learner-only blind-nil diagnostics for the frozen teacher-free PPO.

No import-time JAX or checkpoint operations. The adapter only adds collection
metrics. Persistent game state already carries unfinished declarations between
rollouts; no new TrainingState leaves, cohort resets, RNG calls or loss terms.
"""
from __future__ import annotations

import hashlib
import math

FORMAT = 'spades-passive-training-blind-diagnostics-v1'
FROZEN_TEACHER_FREE_DEFINITION_SHA256 = '82d12df92969e503d8cda02660fd0f34dfe0cc058c307f7a2fa8368bd2d23cc0'
PREFIX = 'blind_diag_'
COUNTER_NAMES = (
    'eligible_opportunities', 'declarations', 'completed_declarations',
    'zero_trick_successes', 'joint_zero_and_team_successes',
    'zero_trick_team_failures', 'completed_learner_team_hands',
    'learner_team_contract_successes', 'learner_actual_team_score_sum',
    'dn_team_hands', 'dn_team_contract_successes', 'dn_actual_team_score_sum',
    'dn_signed_contract_points_sum', 'dn_signed_partner_contract_points_sum',
    'dn_overtrick_points_sum', 'dn_bag_penalty_points_sum',
    'non_dn_actual_team_score_sum', 'dn_bidder_tricks_sum',
    'mixed_completed_matches', 'mixed_wins', 'mixed_losses', 'mixed_draws',
    'training_truncations', 'illegal_actions', 'scoring_coherence_failures',
    'dn_dn_team_hands', 'dn_nil_team_hands', 'dn_positive_team_hands',
    'dn_dn_team_contract_successes', 'dn_nil_team_contract_successes', 'dn_positive_team_contract_successes',
    'dn_dn_actual_team_score_sum', 'dn_nil_actual_team_score_sum', 'dn_positive_actual_team_score_sum',
    'dn_positive_partner_bid_sum', 'dn_positive_partner_tricks_sum',
    'dn_nil_partner_tricks_sum', 'dn_dn_tricks_sum',
)
METRIC_NAMES = tuple(PREFIX + name for name in COUNTER_NAMES)
ENDPOINT_NAMES = tuple(PREFIX + name for name in ('pending_start', 'pending_end', 'declaration_flow_error'))
OPPONENT_FAMILIES = ('inherited', 'nil_shelter_lanes', 'nil_underlead_pressure',
    'positional_overruff_ladder', 'long_suit_entry_links', 'marginal_loser_disposal',
    'public_partner_suit_return', 'marginal_round_score', 'public_scenario_regret', 'invalid')
EXPOSURE_METRIC_NAMES = tuple('opponent_diag_' + family + '_' + event
    for event in ('opponent_decisions', 'learner_decisions', 'accepted_match_assignments')
    for family in OPPONENT_FAMILIES)
EXPOSURE_START_NAMES = tuple('opponent_diag_' + family + '_envs_start' for family in OPPONENT_FAMILIES)
ALL_METRIC_NAMES = METRIC_NAMES + ENDPOINT_NAMES + EXPOSURE_METRIC_NAMES + EXPOSURE_START_NAMES
RATE_SPECS = {
    'eligible_usage': ('declarations', 'eligible_opportunities'),
    'zero_trick_success': ('zero_trick_successes', 'completed_declarations'),
    'joint_zero_and_team_success': ('joint_zero_and_team_successes', 'completed_declarations'),
    'mean_actual_points_per_dn_team_hand': ('dn_actual_team_score_sum', 'dn_team_hands'),
    'mean_actual_points_per_learner_team_hand': ('learner_actual_team_score_sum', 'completed_learner_team_hands'),
    'learner_team_contract_success': ('learner_team_contract_successes', 'completed_learner_team_hands'),
    'mixed_completed_match_win_rate': ('mixed_wins', 'mixed_completed_matches'),
    'dn_dn_team_contract_success': ('dn_dn_team_contract_successes', 'dn_dn_team_hands'),
    'dn_nil_team_contract_success': ('dn_nil_team_contract_successes', 'dn_nil_team_hands'),
    'dn_positive_team_contract_success': ('dn_positive_team_contract_successes', 'dn_positive_team_hands'),
    'mean_points_per_dn_dn_team_hand': ('dn_dn_actual_team_score_sum', 'dn_dn_team_hands'),
    'mean_points_per_dn_nil_team_hand': ('dn_nil_actual_team_score_sum', 'dn_nil_team_hands'),
    'mean_points_per_dn_positive_team_hand': ('dn_positive_actual_team_score_sum', 'dn_positive_team_hands'),
    'mean_positive_partner_bid': ('dn_positive_partner_bid_sum', 'dn_positive_team_hands'),
    'mean_positive_partner_tricks': ('dn_positive_partner_tricks_sum', 'dn_positive_team_hands'),
    'mean_nil_partner_tricks': ('dn_nil_partner_tricks_sum', 'dn_nil_team_hands'),
    'mean_dn_dn_total_tricks': ('dn_dn_tricks_sum', 'dn_dn_team_hands'),
}


def learner_team_mask(league_kind, learner_team, *, xp):
    """Self-play counts both current-policy teams; mixed games count one."""
    return ((league_kind[:, None] == 0)
            | (xp.arange(2)[None, :] == learner_team[:, None]))


def pending_declarations(state, *, xp):
    own = learner_team_mask(state.league_kind, state.learner_team, xp=xp)
    return xp.sum(own[:, xp.arange(4) % 2] & (state.envs.bids == -1))


def opponent_categories(kinds, styles, valid_styles, *, xp):
    """Exact style validity is supplied by installed training-roster validator."""
    new = (styles >= 20000) & (styles < 20512)
    family = xp.where(new, 1+(styles-20000)//64, 0)
    family = xp.where(valid_styles, family, len(OPPONENT_FAMILIES)-1)
    return (kinds[:, None] == 2) & (family[:, None] == xp.arange(len(OPPONENT_FAMILIES))[None, :])


def opponent_exposure_counts(kinds, styles, valid, learner_turn, next_kinds,
                             next_styles, next_valid, reassigned, *, xp):
    current = opponent_categories(kinds, styles, valid, xp=xp)
    accepted = opponent_categories(next_kinds, next_styles, next_valid, xp=xp) & reassigned[:, None]
    return xp.concatenate((xp.sum(current & ~learner_turn[:, None], axis=0),
                           xp.sum(current & learner_turn[:, None], axis=0),
                           xp.sum(accepted, axis=0))).astype(xp.float32)


def turn_counters(envs, actions, legal, players, league_kind, learner_team,
                  final_tricks, info, truncated, *, xp):
    """Scalar sums from batched arrays; arbitrary/asynchronous per-env turns.

    final_tricks is the frozen collector's existing scored_tricks (last trick
    included). It is read only where info.hand_done, before automatic reset.
    Score components are an exact algebraic partition, not a counterfactual
    causal attribution. Teams with two DN seats contribute their delta once.
    """
    n = actions.shape[0]
    expected = {'bids': (n, 4), 'tricks': (n, 4), 'legal': (n, 68),
                'scores': (n, 2), 'eligible': (n, 2)}
    for key, array in [('bids', envs.bids), ('tricks', final_tricks), ('legal', legal),
                       ('scores', info['score_delta']), ('eligible', envs.eligible)]:
        if array.shape != expected[key]:
            raise ValueError('Batched diagnostic shape differs: ' + key)
    own_team = learner_team_mask(league_kind, learner_team, xp=xp)
    own_seat = own_team[:, xp.arange(4) % 2]
    row = xp.arange(n)
    valid = ((actions >= 0) & (actions < 68)
             & legal[row, xp.clip(actions, 0, 67)])
    safe = xp.where(valid, actions, xp.argmax(legal, axis=1))
    own_turn = own_team[row, players % 2]
    opportunity = (envs.turn < 4) & own_turn & legal[:, 67]
    declared = opportunity & (safe == 67)
    hand_end = info['hand_done']
    teams = own_team & hand_end[:, None]
    seats = own_seat & hand_end[:, None]
    dn = envs.bids == -1
    completed = seats & dn
    zero = final_tricks == 0
    success = info['contract_success']
    seat_success = success[:, xp.arange(4) % 2]
    team_players = xp.asarray([[0, 2], [1, 3]])
    b = envs.bids[:, team_players]
    t = final_tricks[:, team_players]
    normal = b > 0
    both_normal = xp.all(normal, axis=2)
    expected_success = xp.where(both_normal, xp.sum(t, axis=2) >= xp.sum(b, axis=2),
                                xp.all(xp.where(normal, t >= b, t == 0), axis=2))
    value = xp.sum(xp.where(normal, 10*b, xp.where(b == -1, 200, 100)), axis=2)
    excess = xp.where(both_normal, xp.sum(t-b, axis=2),
                      xp.sum(xp.where(normal, t-b, 0), axis=2))
    expected_bags = xp.where(expected_success, xp.maximum(excess, 0), 0)
    penalties = 100*((envs.bags + expected_bags)//10)
    expected_delta = xp.where(expected_success, value + expected_bags, -value) - penalties
    dn_count = xp.sum(b == -1, axis=2)
    dn_teams = teams & (dn_count > 0)
    dn_dn = dn_teams & (dn_count == 2)
    dn_nil = dn_teams & xp.any(b == 0, axis=2)
    dn_positive = dn_teams & xp.any(normal, axis=2)
    sign = xp.where(success, 1, -1)
    mixed_end = (league_kind != 0) & info['match_done']
    winner = info['winner']
    coherent_team = ((expected_success == success)
                     & (expected_bags == info['bags_added'])
                     & (expected_delta == info['score_delta']))
    coherent_hand = ((xp.sum(final_tricks, axis=1) == 13)
                     & xp.all(final_tricks >= 0, axis=1)
                     & xp.all((envs.bids >= -1) & (envs.bids <= 13), axis=1))
    s = lambda mask: xp.sum(mask, dtype=xp.int32)
    points = lambda mask, value: xp.sum(xp.where(mask, value, 0), dtype=xp.int32)
    values = (
        s(opportunity), s(declared), s(completed), s(completed & zero),
        s(completed & zero & seat_success), s(completed & zero & ~seat_success),
        s(teams), s(teams & success), points(teams, info['score_delta']),
        s(dn_teams), s(dn_teams & success), points(dn_teams, info['score_delta']),
        points(dn_teams, sign*200*dn_count), points(dn_teams, sign*(value-200*dn_count)),
        points(dn_teams, info['bags_added']), points(dn_teams, penalties),
        points(teams & (dn_count == 0), info['score_delta']), points(completed, final_tricks),
        s(mixed_end), s(mixed_end & (winner == learner_team)),
        s(mixed_end & (winner == 1-learner_team)), s(mixed_end & (winner == 2)),
        s(truncated), s(info['illegal']),
        s(teams & (~coherent_team | ~coherent_hand[:, None])),
        s(dn_dn), s(dn_nil), s(dn_positive),
        s(dn_dn & success), s(dn_nil & success), s(dn_positive & success),
        points(dn_dn, info['score_delta']), points(dn_nil, info['score_delta']),
        points(dn_positive, info['score_delta']),
        points(dn_positive, xp.sum(xp.where(normal, b, 0), axis=2)),
        points(dn_positive, xp.sum(xp.where(normal, t, 0), axis=2)),
        points(dn_nil, xp.sum(xp.where(b == 0, t, 0), axis=2)),
        points(dn_dn, xp.sum(t, axis=2)),
    )
    return xp.stack(values).astype(xp.float32)


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Exact frozen teacher-free diagnostic anchor required: ' + before[:100])
    return source.replace(before, after, 1)


def instrumented_training_definition(source):
    """Append metrics to the exact teacher-free definition, with no train edits."""
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_TEACHER_FREE_DEFINITION_SHA256:
        raise ValueError('Exact frozen teacher-free definition required.')
    source = _replace_once(source, '    def collect(state):\n',
        '    def collect(state):\n        _blind_pending_start = _blind_pending(state)\n'
        '        _blind_opponent_start = _blind_opponent_snapshot(state)\n')
    source = _replace_once(source, '            metric = jnp.stack([\n',
        '            _blind_diagnostics = _blind_turn_counters(\n'
        '                s.envs, actions, legal, who, s.league_kind, s.learner_team,\n'
        '                scored_tricks, info, truncated)\n'
        '            _blind_exposure = _blind_opponent_exposure(\n'
        '                s, learner_turn, kind, style, done | truncated)\n'
        '            metric = jnp.stack([\n')
    source = _replace_once(source, '                jnp.sum(truncated), *eligible_blind_metrics,\n',
        '                jnp.sum(truncated), *eligible_blind_metrics, *_blind_diagnostics, *_blind_exposure,\n')
    source = _replace_once(source, '_ROLLOUT_METRIC_NAMES + _TEACHER_FREE_BLIND_METRIC_NAMES,\n',
        '_ROLLOUT_METRIC_NAMES + _TEACHER_FREE_BLIND_METRIC_NAMES + _PASSIVE_BLIND_NAMES,\n')
    source = _replace_once(source, '        return state, batch, metrics\n',
        "        metrics['blind_diag_pending_start'] = _blind_pending_start.astype(jnp.float32)\n"
        "        metrics['blind_diag_pending_end'] = _blind_pending(state).astype(jnp.float32)\n"
        "        metrics['blind_diag_declaration_flow_error'] = (\n"
        "            metrics['blind_diag_pending_start'] + metrics['blind_diag_declarations']\n"
        "            - metrics['blind_diag_completed_declarations'] - metrics['blind_diag_pending_end'])\n"
        '        metrics.update(zip(_PASSIVE_OPPONENT_START_NAMES, _blind_opponent_start))\n'
        '        return state, batch, metrics\n')
    return source


def install_training_diagnostics(ns, production_notebook_path, objective):
    """Call after teacher-free install, before make_train_step / compilation.

    The callable has no state argument. Existing trainer closures are untouched;
    create the admitted trainer only after installing this adapter.
    """
    if ns.get('_TEACHER_FREE_OBJECTIVE') != objective.OBJECTIVE:
        raise ValueError('Teacher-free objective must already be installed.')
    if ns.get('_PASSIVE_BLIND_NAMES') is not None:
        raise ValueError('Passive diagnostics install exactly once.')
    source = objective.patched_training_definition(production_notebook_path)
    patched = instrumented_training_definition(source)
    xp = ns['jnp']
    valid_style = ns['jax'].vmap(ns['is_train_opponent_style_id'])
    ns['_blind_pending'] = lambda state: pending_declarations(state, xp=xp)
    ns['_blind_turn_counters'] = lambda *args: turn_counters(*args, xp=xp)
    ns['_blind_opponent_snapshot'] = lambda state: xp.sum(opponent_categories(
        state.league_kind, state.league_style, valid_style(state.league_style), xp=xp), axis=0).astype(xp.float32)
    ns['_blind_opponent_exposure'] = lambda state, learner_turn, kind, style, reassigned: opponent_exposure_counts(
        state.league_kind, state.league_style, valid_style(state.league_style), learner_turn,
        kind, style, valid_style(style), reassigned, xp=xp)
    ns['_PASSIVE_BLIND_NAMES'] = METRIC_NAMES + EXPOSURE_METRIC_NAMES
    ns['_PASSIVE_OPPONENT_START_NAMES'] = EXPOSURE_START_NAMES
    exec(compile(patched, 'passive_blind_make_train_step.py', 'exec'), ns)
    return dict(format=FORMAT, original_definition_sha256=hashlib.sha256(source.encode()).hexdigest(),
                instrumented_definition_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                metrics=list(ALL_METRIC_NAMES), extra_collection_scalar_leaves=len(ALL_METRIC_NAMES),
                training_state_schema_changed=False, ppo_batch_schema_changed=False,
                reward_changed=False, rng_changed=False, objective_changed=False,
                pending_state='Derived from persistent TrainingState.envs.bids at both rollout boundaries.')


def summarize_exposure(metrics):
    result = {}
    for name in EXPOSURE_METRIC_NAMES + EXPOSURE_START_NAMES:
        value = float(metrics[name])
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('Finite nonnegative integral opponent exposure required.')
        result[name] = int(value)
    if any(value for name, value in result.items() if '_invalid_' in name):
        raise ValueError('Invalid scripted training style observed in passive exposure.')
    return dict(counts=result, families=list(OPPONENT_FAMILIES[:-1]),
                assignments='Only accepted replacements after actual match completion or training truncation; generated but unused assignments excluded.',
                snapshots='envs_start is a collection-start snapshot; never sum it as an assignment event.',
                decisions='Actual scripted opponent turns and learner turns against each scripted family; self-play and snapshot games excluded.')


def summarize(metrics):
    """Host-only report; null means no denominator, never implied zero success."""
    raw = {}
    for name in COUNTER_NAMES + ('pending_start', 'pending_end', 'declaration_flow_error'):
        value = float(metrics[PREFIX + name])
        if not math.isfinite(value) or value != int(value):
            raise ValueError('Finite integral passive counter required: ' + name)
        raw[name] = int(value)
    if raw['declaration_flow_error'] != 0 or raw['scoring_coherence_failures'] != 0:
        raise ValueError('Blind declaration flow / actual score coherence failure.')
    if raw['pending_start'] + raw['declarations'] != raw['completed_declarations'] + raw['pending_end']:
        raise ValueError('Blind declaration boundary accounting differs.')
    signed = {'learner_actual_team_score_sum', 'dn_actual_team_score_sum',
              'dn_signed_contract_points_sum', 'dn_signed_partner_contract_points_sum',
              'non_dn_actual_team_score_sum', 'dn_dn_actual_team_score_sum',
              'dn_nil_actual_team_score_sum', 'dn_positive_actual_team_score_sum'}
    if any(value < 0 for name, value in raw.items() if name not in signed):
        raise ValueError('Passive event counts cannot be negative.')
    if not (raw['declarations'] <= raw['eligible_opportunities']
            and raw['joint_zero_and_team_successes'] <= raw['zero_trick_successes'] <= raw['completed_declarations']
            and raw['dn_team_contract_successes'] <= raw['dn_team_hands'] <= raw['completed_learner_team_hands']
            and raw['learner_team_contract_successes'] <= raw['completed_learner_team_hands']):
        raise ValueError('Passive success count exceeds its denominator.')
    if raw['mixed_completed_matches'] != raw['mixed_wins'] + raw['mixed_losses'] + raw['mixed_draws']:
        raise ValueError('Mixed match outcome partition differs.')
    if raw['zero_trick_successes'] != raw['joint_zero_and_team_successes'] + raw['zero_trick_team_failures']:
        raise ValueError('Individual and joint success partition differs.')
    component = (raw['dn_signed_contract_points_sum'] + raw['dn_signed_partner_contract_points_sum']
                 + raw['dn_overtrick_points_sum'] - raw['dn_bag_penalty_points_sum'])
    if raw['dn_actual_team_score_sum'] != component:
        raise ValueError('DN team score algebraic partition differs.')
    if raw['learner_actual_team_score_sum'] != raw['dn_actual_team_score_sum'] + raw['non_dn_actual_team_score_sum']:
        raise ValueError('Whole learner team score partition differs.')
    for suffix, total in [('team_hands', 'dn_team_hands'),
                          ('team_contract_successes', 'dn_team_contract_successes'),
                          ('actual_team_score_sum', 'dn_actual_team_score_sum')]:
        if sum(raw[pair+'_'+suffix] for pair in ('dn_dn', 'dn_nil', 'dn_positive')) != raw[total]:
            raise ValueError('DN partner-contract partition differs: ' + suffix)
    for pair in ('dn_dn', 'dn_nil', 'dn_positive'):
        if raw[pair+'_team_contract_successes'] > raw[pair+'_team_hands']:
            raise ValueError('DN pair success exceeds partnership-hand denominator.')
    rates = {name: dict(numerator=raw[num], denominator=raw[den],
                       value=raw[num]/raw[den] if raw[den] else None)
             for name, (num, den) in RATE_SPECS.items()}
    return dict(format=FORMAT, counts=raw, rates=rates,
                scope='On-policy training distribution; self-play includes both teams; mixed games include only the learner team.',
                completions='Completion-cohort denominator; declarations can precede this rollout. Pending boundaries reconcile the flow.',
                score_attribution='DN team score includes every contract component and bags, counted once per team hand. Signed DN points are algebraic, not causal or counterfactual.',
                strength='Mixed completed-match results are descriptive and exclude censored training truncations. Independent fixed held-out evaluation is required for strength claims.',
                optimization='Passive only; never select or optimize on zero-trick success alone.')


def aggregate_collections(rows):
    """Host-only aggregation of contiguous collect dictionaries, preserving carry.

    Call separately for gaps/restarts lacking contiguous evidence. Never sum the
    pending endpoints as if they were completed/declaration event counters.
    """
    if not rows:
        raise ValueError('At least one contiguous collection row required.')
    reports = [summarize(row) for row in rows]
    for left, right in zip(reports, reports[1:]):
        if left['counts']['pending_end'] != right['counts']['pending_start']:
            raise ValueError('Pending declarations differ across collection boundary.')
    result = {PREFIX + name: sum(r['counts'][name] for r in reports) for name in COUNTER_NAMES}
    result.update(blind_diag_pending_start=reports[0]['counts']['pending_start'],
                  blind_diag_pending_end=reports[-1]['counts']['pending_end'],
                  blind_diag_declaration_flow_error=0)
    return summarize(result)
