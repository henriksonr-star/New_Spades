def make_train_step(cfg):
    """Return step(state) -> (state, scalar_metrics).

    Rollout and optimization compile separately. Games persist across rollouts
    and sessions until a true result or a bootstrapped training sampling boundary.
    """
    cfg = learning_config(cfg)
    model, tx = model_for_config(cfg), _optimizer(cfg)
    _batch_constraint = _teacher_free_batch_constraint
    n, horizon = cfg["n_env"], cfg["rollout_len"]
    total = n * horizon
    n_minibatches = total // cfg["minibatch_size"]

    @jax.jit
    def collect(state):
        _blind_pending_start = _blind_pending(state)
        _blind_opponent_start = _blind_opponent_snapshot(state)
        def one_step(s, unused):
            key, k_action, k_env, k_heuristic, k_league = jax.random.split(s.rng, 5)
            obs = jax.vmap(observe)(s.envs)
            central_obs = jax.vmap(critic_observe)(s.envs)
            legal = jax.vmap(legal_mask)(s.envs)
            who = jax.vmap(actor)(s.envs)
            logits, values = model.apply({"params": s.params}, obs, central_obs)
            learner_turn = (s.league_kind == 0) | ((who % 2) == s.learner_team)
            historical_turn = (s.league_kind == 1) & ~learner_turn

            def historical_logits(_):
                # Batch by bank slot, never materialize one model per environment.
                bank_logits = jax.vmap(lambda p: model.apply({"params": p}, obs, method=model.actor))(
                    s.snapshot_params)
                return bank_logits[s.league_snapshot, jnp.arange(n)]

            old_logits = jax.lax.cond(jnp.any(historical_turn), historical_logits,
                                      lambda _: jnp.zeros_like(logits), operand=None)
            acting_logits = jnp.where(historical_turn[:, None], old_logits, logits)
            actions = jax.random.categorical(k_action, jnp.where(legal, acting_logits, -1e9), axis=-1)
            scripted_turn = (s.league_kind == 2) & ~learner_turn
            scripted = jax.lax.cond(jnp.any(scripted_turn),
                lambda _: jax.vmap(diverse_heuristic_action)(s.envs, jax.random.split(k_heuristic, n), s.league_style),
                lambda _: jnp.zeros((n,), jnp.int32), operand=None)
            actions = jnp.where(scripted_turn, scripted, actions).astype(jnp.int32)
            logs = _masked_log_probs(logits, legal)
            logp = jnp.take_along_axis(logs, actions[:, None], axis=-1)[:, 0]
            next_envs, rewards, done, info = batch_step(s.envs, actions, jax.random.split(k_env, n))
            engine_rewards = rewards
            rewards = _training_rewards(engine_rewards, info, cfg)
            hand_rewards = rewards
            next_envs, rewards, trace_discounts, truncated = _refresh_training_trajectories(
                s.params, next_envs, rewards, info['discount'], done, info['hand_done'], k_env, cfg)
            transition = TeacherFreeTransition(obs, central_obs, actions, logp, values, rewards,
                                    trace_discounts, learner_turn & (jnp.sum(legal, axis=-1) > 1),
                                    jnp.where(who % 2 == 0, 1., -1.), legal)
            new_assignments = _league_assignments(k_league, cfg, s.snapshot_count, s.snapshot_retiring)
            old_assignments = (s.league_kind, s.league_style, s.league_snapshot, s.learner_team)
            kind, style, slot, team = [jnp.where(done | truncated, new, old)
                                      for new, old in zip(new_assignments, old_assignments)]
            mixed = (s.league_kind != 0) & done
            won = info["winner"] == s.learner_team
            hand_end = info['hand_done']
            learner_teams = ((s.league_kind[:, None] == 0)
                             | (jnp.arange(2)[None, :] == s.learner_team[:, None]))
            our_contracts = hand_end[:, None] & learner_teams
            other_contracts = hand_end[:, None] & ~learner_teams
            learner_seats = hand_end[:, None] & learner_teams[:, jnp.arange(4) % 2]
            nil_bids = learner_seats & (s.envs.bids == 0)
            dn_bids = learner_seats & (s.envs.bids == -1)
            normal_bids = learner_seats & (s.envs.bids > 0)
            # The last trick's winner has not yet been counted in s.envs.tricks.
            final_table = s.envs.table.at[jnp.arange(n), s.envs.trick_pos].set(actions)
            final_players = s.envs.table_players.at[jnp.arange(n), s.envs.trick_pos].set(who)
            last_winners = jax.vmap(trick_winner)(final_table, final_players)
            scored_tricks = s.envs.tricks + jax.nn.one_hot(last_winners, 4, dtype=jnp.int32)
            actionable = learner_turn & (jnp.sum(legal, axis=-1) > 1)
            entropies = -jnp.sum(jnp.exp(logs) * logs, axis=-1)
            phases = jax.vmap(phase)(s.envs)
            entropy_masks = jnp.stack([actionable & (phases == ph) for ph in (BLIND, BID, PLAY)])
            entropy_sums = jnp.sum(entropy_masks * entropies[None, :], axis=1)
            entropy_counts = jnp.sum(entropy_masks, axis=1)
            eligible_blind = learner_turn & (phases == BLIND) & legal[:, DOUBLE_NIL]
            eligible_blind_metrics = (
                jnp.sum(eligible_blind),
                jnp.sum(jnp.where(eligible_blind, jnp.exp(logs[:, DOUBLE_NIL]), 0.)),
                jnp.sum(eligible_blind & (actions == DOUBLE_NIL)),
                jnp.sum(jnp.where(eligible_blind, entropies, 0.)),
            )
            _blind_diagnostics = _blind_turn_counters(
                s.envs, actions, legal, who, s.league_kind, s.learner_team,
                scored_tricks, info, truncated)
            _blind_exposure = _blind_opponent_exposure(
                s, learner_turn, kind, style, done | truncated)
            metric = jnp.stack([
                jnp.sum(done), jnp.sum(done & (info["winner"] == 0)),
                jnp.sum(done & (info["winner"] == 1)), jnp.sum(done & (info["winner"] == 2)),
                jnp.sum(info["hand_done"]), jnp.sum(info["nil_attempts"]), jnp.sum(info["nil_successes"]),
                jnp.sum(info["dn_attempts"]), jnp.sum(info["dn_successes"]), jnp.sum(info["illegal"]),
                jnp.sum(jnp.where(done, info["final_scores"][:, 0], 0)),
                jnp.sum(jnp.where(done, info["final_scores"][:, 1], 0)),
                jnp.sum(mixed), jnp.sum(mixed & won), jnp.sum(mixed & (info["winner"] == 2)),
                jnp.sum(done & (s.league_kind == 2)), jnp.sum(done & (s.league_kind == 2) & won),
                jnp.sum(done & (s.league_kind == 1)), jnp.sum(done & (s.league_kind == 1) & won),
                jnp.sum(info["contract_success"]),
                jnp.sum(our_contracts), jnp.sum(our_contracts & info['contract_success']),
                jnp.sum(jnp.where(our_contracts, info['score_delta'], 0)),
                jnp.sum(jnp.where(other_contracts, info['score_delta'], 0)), jnp.sum(other_contracts),
                jnp.sum(learner_seats), jnp.sum(nil_bids), jnp.sum(dn_bids),
                jnp.sum(nil_bids & (scored_tricks == 0)), jnp.sum(dn_bids & (scored_tricks == 0)),
                jnp.sum(jnp.where(normal_bids, s.envs.bids, 0)), jnp.sum(normal_bids),
                jnp.sum(jnp.abs(hand_rewards)), jnp.sum(jnp.abs(hand_rewards - engine_rewards)),
                *entropy_sums, *entropy_counts,
                jnp.sum(truncated), *eligible_blind_metrics, *_blind_diagnostics, *_blind_exposure,
            ]).astype(jnp.float32)
            s = s._replace(rng=key, envs=next_envs, league_kind=kind, league_style=style,
                           league_snapshot=slot, learner_team=team)
            return s, (transition, metric)

        state, (trajectory, metric_rows) = jax.lax.scan(one_step, state, None, length=horizon)
        last_values = model.apply({"params": state.params}, jax.vmap(critic_observe)(state.envs),
                                  method=model.value)

        def gae_step(carry, tr):
            next_advantage, next_value = carry
            delta = tr.rewards + tr.discounts * next_value - tr.values
            advantage = delta + _gae_trace_discount(tr.discounts, cfg) * next_advantage
            return (advantage, tr.values), advantage

        _, advantages = jax.lax.scan(gae_step, (jnp.zeros(n), last_values), trajectory, reverse=True)
        returns = advantages + trajectory.values
        signed_adv = advantages * trajectory.actor_sign
        mask = trajectory.policy_mask.astype(jnp.float32)
        mean_adv = _masked_mean(signed_adv, mask)
        var_adv = _masked_mean((signed_adv - mean_adv) ** 2, mask)
        normalized_adv = (signed_adv - mean_adv) / jnp.sqrt(var_adv + 1e-8)
        batch = TeacherFreePPOBatch(trajectory.obs, trajectory.critic_obs, trajectory.actions, trajectory.logp,
                         trajectory.values, returns, normalized_adv, mask, trajectory.legal)
        batch = jax.tree_util.tree_map(lambda a: a.reshape((total,) + a.shape[2:]), batch)
        metrics = dict(zip(_ROLLOUT_METRIC_NAMES + _TEACHER_FREE_BLIND_METRIC_NAMES + _PASSIVE_BLIND_NAMES,
                           jnp.sum(metric_rows, axis=0)))
        metrics.update(policy_samples=jnp.sum(mask), learner_fraction=jnp.mean(mask),
                       reward_mean=jnp.mean(trajectory.rewards),
                       raw_adv_std=jnp.sqrt(var_adv),
                       explained_variance=1. - jnp.var(returns - trajectory.values) /
                                           jnp.maximum(jnp.var(returns), 1e-8),
                       return_mean=jnp.mean(returns), value_mean=jnp.mean(trajectory.values))
        metrics['blind_diag_pending_start'] = _blind_pending_start.astype(jnp.float32)
        metrics['blind_diag_pending_end'] = _blind_pending(state).astype(jnp.float32)
        metrics['blind_diag_declaration_flow_error'] = (
            metrics['blind_diag_pending_start'] + metrics['blind_diag_declarations']
            - metrics['blind_diag_completed_declarations'] - metrics['blind_diag_pending_end'])
        metrics.update(zip(_PASSIVE_OPPONENT_START_NAMES, _blind_opponent_start))
        return state, batch, metrics

    _loss_names = _TEACHER_FREE_LOSS_NAMES

    def loss_fn(params, batch):
        batch = _batch_constraint(batch)
        logits, values = model.apply({"params": params}, batch.obs, batch.critic_obs)
        logs = _masked_log_probs(logits, batch.legal)
        chosen = jnp.take_along_axis(logs, batch.actions[:, None], axis=-1)[:, 0]
        log_ratio = chosen - batch.logp
        # Guard exponent overflow; ordinary PPO ratios are nowhere near these bounds.
        ratio = jnp.exp(jnp.clip(log_ratio, -20., 20.))
        unclipped = ratio * batch.advantages
        clipped = jnp.clip(ratio, 1. - cfg["clip_eps"], 1. + cfg["clip_eps"]) * batch.advantages
        policy_loss = -_masked_mean(jnp.minimum(unclipped, clipped), batch.policy_mask)
        clipped_values = batch.old_values + jnp.clip(values - batch.old_values,
                                                      -cfg["value_clip"], cfg["value_clip"])
        value_loss = .5 * jnp.mean(jnp.maximum((values - batch.returns) ** 2,
                                               (clipped_values - batch.returns) ** 2))
        entropy = _masked_mean(-jnp.sum(jnp.exp(logs) * logs, axis=-1), batch.policy_mask)
        approx_kl = _masked_mean((ratio - 1.) - log_ratio, batch.policy_mask)
        clip_fraction = _masked_mean((jnp.abs(ratio - 1.) > cfg["clip_eps"]).astype(jnp.float32),
                                     batch.policy_mask)
        loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
        return loss, jnp.stack([loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction])

    @jax.jit
    def optimize(state, batch):
        def epoch(carry, unused):
            params, opt_state, key, stopped = carry
            key, k_shuffle = jax.random.split(key)
            permutation = jax.random.permutation(k_shuffle, total)
            minibatches = jax.tree_util.tree_map(
                lambda a: a[permutation].reshape((n_minibatches, cfg["minibatch_size"]) + a.shape[1:]), batch)

            def minibatch(carry, mini):
                params, opt_state, key, stopped = carry

                def update(args):
                    params, opt_state = args
                    (_, diagnostics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mini)
                    grad_norm = optax.global_norm(grads)
                    updates, new_opt_state = tx.update(grads, opt_state, params)
                    new_params = optax.apply_updates(params, updates)
                    checks = _update_checks(diagnostics, grads, grad_norm, new_params, new_opt_state)
                    finite = jnp.all(checks)
                    params, opt_state = jax.lax.cond(finite,
                        lambda _: (new_params, new_opt_state), lambda _: (params, opt_state), operand=None)
                    # Stop subsequent minibatches after a large on-policy KL drift.
                    stop = ~finite | (diagnostics[4] > 2. * cfg["target_kl"])
                    row = (jnp.concatenate((diagnostics, grad_norm[None])),
                           jnp.array([1, finite.astype(jnp.int32)], jnp.int32), checks)
                    return (params, opt_state, key, stop), row

                return jax.lax.cond(stopped,
                    lambda _: ((params, opt_state, key, stopped),
                               (jnp.zeros(len(_loss_names)), jnp.zeros(2, jnp.int32),
                                jnp.ones(len(_UPDATE_CHECK_NAMES), jnp.bool_))),
                    update, (params, opt_state))

            carry, rows = jax.lax.scan(minibatch, (params, opt_state, key, stopped), minibatches)
            return carry, _reduce_update_rows(rows)

        (params, opt_state, key, stopped), rows = jax.lax.scan(
            epoch, (state.params, state.opt_state, state.rng, jnp.bool_(False)), None,
            length=cfg["ppo_epochs"])
        metrics = _update_metrics(rows, _loss_names)
        metrics["kl_early_stop"] = stopped & metrics["finite"]
        state = state._replace(params=params, opt_state=opt_state, rng=key,
                               updates=state.updates + 1)
        state = _refresh_snapshot(state, cfg)
        metrics.update(updates=state.updates,
                       snapshot_count=state.snapshot_count, snapshot_retiring=state.snapshot_retiring)
        return state, metrics

    def train_step(state):
        state, batch, rollout_metrics = collect(state)
        state, update_metrics = optimize(state, batch)
        metrics = dict(rollout_metrics)
        metrics.update(update_metrics)
        metrics["env_steps"] = state.env_steps
        blind_count = metrics['eligible_blind_count']
        # Zero count remains explicit; the derived zero is not evidence of collapse.
        metrics['eligible_blind_dn_probability_mean'] = metrics['eligible_blind_dn_probability_sum'] / jnp.maximum(blind_count, 1.)
        metrics['eligible_blind_dn_sample_rate'] = metrics['eligible_blind_dn_sample_count'] / jnp.maximum(blind_count, 1.)
        metrics['eligible_blind_entropy_mean'] = metrics['eligible_blind_entropy_sum'] / jnp.maximum(blind_count, 1.)
        matches = jnp.maximum(metrics["matches"], 1.)
        metrics["mean_final_score0"] = metrics.pop("final_score0_sum") / matches
        metrics["mean_final_score1"] = metrics.pop("final_score1_sum") / matches
        metrics["illegal_rate"] = metrics["illegal_actions"] / total
        metrics["hand_rate"] = metrics["hands"] / total
        metrics["match_rate"] = metrics["matches"] / total
        metrics["contract_success_rate"] = metrics["successful_contracts"] / jnp.maximum(2. * metrics["hands"], 1.)
        metrics['learner_contract_success_rate'] = metrics['learner_successful_contracts'] / jnp.maximum(metrics['learner_contracts'], 1.)
        metrics['learner_hand_score'] = metrics['learner_hand_score_sum'] / jnp.maximum(metrics['learner_contracts'], 1.)
        metrics['opponent_hand_score'] = metrics['opponent_hand_score_sum'] / jnp.maximum(metrics['opponent_contracts'], 1.)
        metrics['mean_normal_bid'] = metrics['learner_normal_bid_sum'] / jnp.maximum(metrics['learner_normal_bids'], 1.)
        metrics['nil_bid_rate'] = metrics['learner_nil_bids'] / jnp.maximum(metrics['learner_bid_seats'], 1.)
        metrics['dn_bid_rate'] = metrics['learner_dn_bids'] / jnp.maximum(metrics['learner_bid_seats'], 1.)
        metrics['learner_nil_success_rate'] = metrics['learner_nil_made'] / jnp.maximum(metrics['learner_nil_bids'], 1.)
        metrics['learner_dn_success_rate'] = metrics['learner_dn_made'] / jnp.maximum(metrics['learner_dn_bids'], 1.)
        metrics['hand_reward_mean_abs'] = metrics['hand_reward_abs_sum'] / jnp.maximum(metrics['hands'], 1.)
        metrics['score_bonus_mean_abs'] = metrics['score_bonus_abs_sum'] / jnp.maximum(metrics['hands'], 1.)
        for ph in ('blind', 'bid', 'play'):
            metrics['entropy_' + ph] = metrics['entropy_' + ph + '_sum'] / jnp.maximum(metrics['entropy_' + ph + '_count'], 1.)
        metrics.update(completed_matches=metrics['matches'],
                       mean_match_hands=jnp.mean(state.envs.hand_number.astype(jnp.float32)),
                       max_match_hands=jnp.max(state.envs.hand_number),
                       mean_score0=jnp.mean(state.envs.scores[:, 0].astype(jnp.float32)),
                       mean_score1=jnp.mean(state.envs.scores[:, 1].astype(jnp.float32)),
                       score_gap_saturated_fraction=jnp.mean((jnp.abs(
                           state.envs.scores[:, 0] - state.envs.scores[:, 1]) >= 1500).astype(jnp.float32)))
        return state, metrics

    # Diagnostic hooks also let the notebook's optional verification exercise
    # the actual loss/rollout rather than a second implementation of PPO.
    train_step.collect = collect
    train_step.optimize = optimize
    train_step.loss_fn = loss_fn
    train_step.loss_names = _loss_names
    train_step.objective = _TEACHER_FREE_OBJECTIVE
    return train_step
