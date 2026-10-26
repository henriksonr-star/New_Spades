"""JAX PPO, persistent league matchmaking, and optional supervised initialization.

This cell follows the engine and opponents cells in the standalone notebook.
All values needed to resume training live in TrainingState; no Python RNG is used.
"""
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

TRAINING_REVISION = 'strong-teacher-hand-gae-v3'


def learning_config(overrides=None):
    cfg = dict(seed=42, n_env=256, rollout_len=128, ppo_epochs=3,
               minibatch_size=4096, width=256, lr=2e-4,
               gae_lambda=0.995, gae_per_hand=False, clip_eps=0.20, value_clip=0.25,
               max_grad_norm=0.5, entropy_coef=0.015, value_coef=0.5,
               target_kl=0.04, snapshot_size=4,
               snapshot_interval_updates=100, selfplay_prob=0.50,
               snapshot_prob=0.25, bootstrap_epochs=2, bootstrap_minibatch_size=1024,
               teacher_warmup_decisions=6291456,
               hand_score_reward_coef=0.0, bidding_anchor_coef=0.1,
               play_anchor_coef=0.3, teacher_style=7, teacher_decline_blind=True,
               max_training_match_hands=128)
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError("Unknown training settings: " + ", ".join(sorted(unknown)))
        cfg.update(overrides)
    if int(cfg["seed"]) != cfg["seed"] or not 0 <= cfg["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    cfg["seed"] = int(cfg["seed"])
    for name in ("n_env", "rollout_len", "ppo_epochs", "minibatch_size", "width",
                 "snapshot_size", "snapshot_interval_updates", "bootstrap_epochs",
                 "max_training_match_hands", "bootstrap_minibatch_size", "teacher_warmup_decisions"):
        if int(cfg[name]) != cfg[name] or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
        cfg[name] = int(cfg[name])
    if cfg["n_env"] * cfg["rollout_len"] % cfg["minibatch_size"]:
        raise ValueError("n_env * rollout_len must be divisible by minibatch_size")
    # Tiny CPU fixtures can use one smaller teacher minibatch. Production batches
    # keep teacher SGD independent of the much larger PPO minibatch size.
    if cfg['n_env'] * cfg['rollout_len'] % min(cfg['bootstrap_minibatch_size'], cfg['n_env'] * cfg['rollout_len']):
        raise ValueError('Teacher minibatch size must divide the rollout size')
    for name in ('gae_per_hand', 'teacher_decline_blind'):
        if not isinstance(cfg[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
    if int(cfg['teacher_style']) != cfg['teacher_style'] or not 0 <= cfg['teacher_style'] < len(HEURISTIC_NAMES):
        raise ValueError('teacher_style must identify an existing public-information heuristic')
    cfg['teacher_style'] = int(cfg['teacher_style'])
    if not (0 <= cfg["selfplay_prob"] <= 1 and 0 <= cfg["snapshot_prob"] <= 1
            and cfg["selfplay_prob"] + cfg["snapshot_prob"] <= 1):
        raise ValueError("selfplay_prob and snapshot_prob must be nonnegative and sum to <= 1")
    if not 0 < cfg["gae_lambda"] <= 1:
        raise ValueError("gae_lambda must lie in (0, 1]")
    for name in ("lr", "clip_eps", "value_clip", "max_grad_norm", "target_kl"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("entropy_coef", "value_coef", "hand_score_reward_coef", "bidding_anchor_coef", "play_anchor_coef"):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    return cfg


def _network_features(features):
    """Compress extreme score inputs while retaining exact ordinary-range values.

    Actor and critic observations both begin with the same public feature layout.
    Scores occupy positions 133:135 and are already divided by 500 in the engine.
    Scores in [-500, 500] pass through unchanged; larger magnitudes use logarithmic
    tails so million-point negative games cannot overwhelm card/phase features.
    The game state, legal actions, and actual point totals remain untouched.
    """
    score = features[..., 133:135]
    magnitude = jnp.abs(score)
    compressed = jnp.sign(score) * (jnp.minimum(magnitude, 1.)
                                    + jnp.log(jnp.maximum(magnitude, 1.)))
    return features.at[..., 133:135].set(compressed)


class ResidualTower(nn.Module):
    width: int
    output_size: int
    actor_output: bool = False

    @nn.compact
    def __call__(self, features):
        x = nn.silu(nn.Dense(self.width, name="input")(_network_features(features)))
        for i in range(3):
            h = nn.LayerNorm(name=f"norm_{i}")(x)
            h = nn.silu(nn.Dense(self.width, name=f"hidden_{i}")(h))
            h = nn.Dense(self.width, name=f"residual_{i}")(h)
            x = nn.silu(x + 0.5 * h)
        init = nn.initializers.normal(0.01) if self.actor_output else nn.initializers.zeros
        return nn.Dense(self.output_size, kernel_init=init, name="output")(x)


class ActorCritic(nn.Module):
    """Actor and critic have completely separate parameters and encoders."""
    width: int = 256

    def setup(self):
        self.actor_network = ResidualTower(self.width, N_ACTIONS, True)
        self.critic_network = ResidualTower(self.width, 1, False)

    def actor(self, obs):
        return self.actor_network(obs)

    def value(self, critic_obs):
        return self.critic_network(critic_obs)[..., 0]

    def __call__(self, obs, critic_obs):
        return self.actor(obs), self.value(critic_obs)


def model_for_config(cfg):
    return ActorCritic(width=int(cfg.get("width", 256)))


def policy_logits(params, obs, cfg):
    model = model_for_config(cfg)
    return model.apply({"params": params}, obs, method=model.actor)


actor_logits = policy_logits


def value_apply(params, critic_obs, cfg):
    model = model_for_config(cfg)
    return model.apply({"params": params}, critic_obs, method=model.value)


def _optimizer(cfg):
    return optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]),
                       optax.adam(cfg["lr"], eps=1e-5))


class TrainingState(NamedTuple):
    params: object
    opt_state: object
    rng: object
    envs: object
    league_kind: object       # 0 current self-play; 1 historical; 2 scripted
    league_style: object
    league_snapshot: object
    learner_team: object     # 0 or 1, used only for mixed-opponent matches
    snapshot_params: object  # leading dimension snapshot_size
    snapshot_count: object
    snapshot_cursor: object
    snapshot_retiring: object
    last_snapshot_update: object
    updates: object
    decisions_per_update: object
    bootstrap_updates: object
    bootstrap_complete: object
    env_steps_low: object
    env_steps_high: object
    base_updates: object

    @property
    def env_steps(self):
        # Host-side exact arithmetic: a cumulative JAX int32 counter would wrap
        # after 2.1 billion decisions, and enabling x64 would slow a T4 down.
        prior = (int(self.env_steps_high) << 32) | int(self.env_steps_low)
        return prior + (int(self.updates) - int(self.base_updates)) * int(self.decisions_per_update)


class Transition(NamedTuple):
    obs: object
    critic_obs: object
    actions: object
    logp: object
    values: object
    rewards: object
    discounts: object
    policy_mask: object
    actor_sign: object
    legal: object
    teacher_actions: object
    teacher_mask: object


class PPOBatch(NamedTuple):
    obs: object
    critic_obs: object
    actions: object
    logp: object
    old_values: object
    returns: object
    advantages: object
    policy_mask: object
    legal: object
    teacher_actions: object
    teacher_mask: object


def _league_assignments(key, cfg, count, retiring):
    n = cfg["n_env"]
    k_kind, k_team, k_style, k_snapshot = jax.random.split(key, 4)
    u = jax.random.uniform(k_kind, (n,))
    kind = jnp.where(u < cfg["selfplay_prob"], 0,
                     jnp.where(u < cfg["selfplay_prob"] + cfg["snapshot_prob"], 1, 2))
    available = ((jnp.arange(cfg["snapshot_size"]) < count)
                 & (jnp.arange(cfg["snapshot_size"]) != retiring))
    slot = jax.random.categorical(k_snapshot, jnp.where(available, 0., -1e9), shape=(n,))
    kind = jnp.where((kind == 1) & ~jnp.any(available), 0, kind)
    team = jax.random.randint(k_team, (n,), 0, 2)
    style = sample_train_opponent_style_ids(k_style, n)
    return kind.astype(jnp.int32), style, slot.astype(jnp.int32), team


def initialize_training(cfg):
    cfg = learning_config(cfg)
    key, k_model, k_env, k_league = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 4)
    example = reset(jax.random.PRNGKey(0))
    model = model_for_config(cfg)
    params = model.init(k_model, observe(example), critic_observe(example))["params"]
    envs = jax.vmap(reset)(jax.random.split(k_env, cfg["n_env"]))
    kind, style, slot, team = _league_assignments(k_league, cfg, jnp.int32(1), jnp.int32(-1))
    bank = jax.tree_util.tree_map(lambda p: jnp.repeat(p[None], cfg["snapshot_size"], axis=0), params)
    return TrainingState(params, _optimizer(cfg).init(params), key, envs,
                         kind, style, slot, team, bank, jnp.int32(1), jnp.int32(1 % cfg["snapshot_size"]),
                         jnp.int32(-1), jnp.int32(0), jnp.int32(0),
                         jnp.int32(cfg["n_env"] * cfg["rollout_len"]), jnp.int32(0), jnp.bool_(False),
                         jnp.uint32(0), jnp.uint32(0), jnp.int32(0))


def _masked_log_probs(logits, legal):
    return jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)


def _tree_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))


_UPDATE_CHECK_NAMES = ("diagnostics_finite", "gradients_finite", "grad_norm_finite",
                       "params_finite", "optimizer_finite")


def _update_checks(diagnostics, grads, grad_norm, params, opt_state):
    # Adam's moments can overflow even when the resulting parameters are finite.
    # Check the entire candidate before committing any part of the update.
    return jnp.stack((jnp.all(jnp.isfinite(diagnostics)), _tree_finite(grads),
                      jnp.isfinite(grad_norm), _tree_finite(params), _tree_finite(opt_state)))


def _reduce_update_rows(rows):
    """Sum diagnostics/counts and AND safety flags without floating-point votes.

    Compiled float32 count / count can use an approximate reciprocal and round
    below 1. An averaged flag compared to exactly 1 can then report a
    failure despite every minibatch being finite. Flags always stay boolean;
    attempted/accepted counts stay int32 throughout both scan reductions.
    """
    diagnostics, counts, checks = rows
    return jnp.sum(diagnostics, axis=0), jnp.sum(counts, axis=0), jnp.all(checks, axis=0)


def _update_metrics(rows, names):
    diagnostics, counts, checks = _reduce_update_rows(rows)
    metrics = dict(zip(names, diagnostics / jnp.maximum(counts[0], 1)))
    metrics.update(zip(_UPDATE_CHECK_NAMES, checks))
    metrics.update(attempted_updates=counts[0], accepted_updates=counts[1],
                   rejected_updates=counts[0] - counts[1], finite=jnp.all(checks))
    return metrics


def _masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.)


def _refresh_snapshot(state, cfg):
    """Retire a bank slot, drain its active matches, and only then replace it.

    A match's historical opponent therefore cannot change underneath it. If all
    slots are busy, publication waits for that slot's existing matches to finish.
    """
    due = state.updates - state.last_snapshot_update >= cfg["snapshot_interval_updates"]
    unused = state.snapshot_count < cfg["snapshot_size"]
    candidate = jnp.where(unused, state.snapshot_count, state.snapshot_cursor)
    retiring = jnp.where((state.snapshot_retiring < 0) & due, candidate, state.snapshot_retiring)
    active = jnp.any((state.league_kind == 1) & (state.league_snapshot == retiring))
    ready = (retiring >= 0) & ~active

    def publish(s):
        bank = jax.tree_util.tree_map(lambda b, p: b.at[retiring].set(p), s.snapshot_params, s.params)
        return s._replace(snapshot_params=bank,
                          snapshot_count=jnp.minimum(s.snapshot_count + (retiring >= s.snapshot_count),
                                                     cfg["snapshot_size"]),
                          snapshot_cursor=(retiring + 1) % cfg["snapshot_size"],
                          snapshot_retiring=jnp.int32(-1), last_snapshot_update=s.updates)

    state = state._replace(snapshot_retiring=retiring)
    return jax.lax.cond(ready, publish, lambda s: s, state)


_ROLLOUT_METRIC_NAMES = (
    "matches", "team0_wins", "team1_wins", "draws", "hands",
    "nil_attempts", "nil_successes", "dn_attempts", "dn_successes",
    "illegal_actions", "final_score0_sum", "final_score1_sum",
    "opponent_matches", "learner_wins", "learner_draws",
    "scripted_matches", "scripted_wins", "historical_matches", "historical_wins",
    "successful_contracts", "learner_contracts", "learner_successful_contracts",
    "learner_hand_score_sum", "opponent_hand_score_sum", "opponent_contracts",
    "learner_bid_seats", "learner_nil_bids", "learner_dn_bids",
    "learner_nil_made", "learner_dn_made", "learner_normal_bid_sum",
    "learner_normal_bids", "hand_reward_abs_sum", "score_bonus_abs_sum",
    "entropy_blind_sum", "entropy_bid_sum", "entropy_play_sum",
    "entropy_blind_count", "entropy_bid_count", "entropy_play_count", "training_truncations")


def _training_rewards(engine_rewards, info, cfg):
    """Team-0 reward with an explicit dense score auxiliary for learning.

    The engine's match-win reward and potential term remain present. The extra
    term uses actual scored points (including failed contracts and bag penalties)
    and stays informative even in long, deeply negative matches. It changes the
    training objective; it is not policy-invariant potential shaping. Evaluation
    and human games always use the original rules and true match winners.
    """
    point_difference = (info['score_delta'][..., 0].astype(jnp.float32)
                        - info['score_delta'][..., 1].astype(jnp.float32))
    return engine_rewards + cfg['hand_score_reward_coef'] * point_difference / 100.


def teacher_action(state, key, cfg):
    """Coherent public-information advice, with no random blind gamble to imitate.

    The heuristic only reads the acting player's visible hand and public state.
    Declining blind is guidance, never a restriction on legal policy actions.
    PPO can still learn to declare Double Nil in appropriate game situations.
    """
    advice = heuristic_action(state, key, jnp.int32(cfg['teacher_style']))
    if cfg['teacher_decline_blind']:
        advice = jnp.where(phase(state) == BLIND, NOOP, advice)
    return advice


def _gae_trace_discount(discounts, cfg):
    # Rewards and the engine's gamma are hand-based. Apply lambda on that same
    # timescale so a hand does not incur 64 extra credit-decay factors.
    if cfg['gae_per_hand']:
        return discounts * jnp.where(discounts < 1., cfg['gae_lambda'], 1.)
    return discounts * cfg['gae_lambda']


def _refresh_training_trajectories(params, envs, rewards, discounts, done, hand_done, key, cfg):
    """Refresh very long training samples with proper time-limit bootstrapping.

    This is a rollout sampling boundary, never a game winner or draw. The critic
    bootstraps from the real continuing game before fresh games are installed;
    zeroing the trace discount then prevents GAE crossing into unrelated deals.
    Engine games, evaluation, and human play have no such sampling boundary.
    """
    truncated = hand_done & ~done & (envs.hand_number >= cfg['max_training_match_hands'])

    def refresh(_):
        model = model_for_config(cfg)
        continuation = model.apply({'params': params}, jax.vmap(critic_observe)(envs), method=model.value)
        fresh = jax.vmap(reset)(jax.random.split(jax.random.fold_in(key, 1), cfg['n_env']))
        result = jax.tree_util.tree_map(
            lambda new, old: jnp.where(truncated.reshape((cfg['n_env'],) + (1,) * (old.ndim - 1)),
                                      new, old), fresh, envs)
        return (result, rewards + jnp.where(truncated, discounts * continuation, 0.),
                jnp.where(truncated, 0., discounts))

    envs, rewards, discounts = jax.lax.cond(jnp.any(truncated), refresh,
        lambda _: (envs, rewards, discounts), operand=None)
    return envs, rewards, discounts, truncated


def make_train_step(cfg):
    """Return step(state) -> (state, scalar_metrics).

    Rollout and optimization compile separately. Games persist across rollouts
    and sessions until a true result or a bootstrapped training sampling boundary.
    """
    cfg = learning_config(cfg)
    model, tx = model_for_config(cfg), _optimizer(cfg)
    n, horizon = cfg["n_env"], cfg["rollout_len"]
    total = n * horizon
    n_minibatches = total // cfg["minibatch_size"]

    @jax.jit
    def collect(state):
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
            # Continue public-information guidance on the learner's own visited
            # positions, including card play that the old bidding-only anchor omitted.
            teacher_phase = (s.envs.turn < 4) | ((s.envs.turn >= 8) & (s.envs.turn < 12))
            teacher_actions = jax.lax.cond(jnp.any(teacher_phase) | (cfg['play_anchor_coef'] > 0),
                lambda _: jax.vmap(lambda env, key: teacher_action(env, key, cfg))(
                    s.envs, jax.random.split(k_heuristic, n)),
                lambda _: jnp.full(n, NOOP, jnp.int32), operand=None)
            teacher_mask = teacher_phase & learner_turn & (jnp.sum(legal, axis=-1) > 1)
            transition = Transition(obs, central_obs, actions, logp, values, rewards,
                                    trace_discounts, learner_turn & (jnp.sum(legal, axis=-1) > 1),
                                    jnp.where(who % 2 == 0, 1., -1.), legal, teacher_actions, teacher_mask)
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
                jnp.sum(truncated),
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
        batch = PPOBatch(trajectory.obs, trajectory.critic_obs, trajectory.actions, trajectory.logp,
                         trajectory.values, returns, normalized_adv, mask, trajectory.legal,
                         trajectory.teacher_actions, trajectory.teacher_mask)
        batch = jax.tree_util.tree_map(lambda a: a.reshape((total,) + a.shape[2:]), batch)
        metrics = dict(zip(_ROLLOUT_METRIC_NAMES, jnp.sum(metric_rows, axis=0)))
        metrics.update(policy_samples=jnp.sum(mask), learner_fraction=jnp.mean(mask),
                       reward_mean=jnp.mean(trajectory.rewards),
                       raw_adv_std=jnp.sqrt(var_adv),
                       explained_variance=1. - jnp.var(returns - trajectory.values) /
                                           jnp.maximum(jnp.var(returns), 1e-8),
                       return_mean=jnp.mean(returns), value_mean=jnp.mean(trajectory.values))
        return state, batch, metrics

    _loss_names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                   "clip_fraction", "bidding_anchor_loss", "play_anchor_loss",
                   "teacher_accuracy_bid", "teacher_accuracy_play", "grad_norm")

    def loss_fn(params, batch):
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
        teacher_logp = jnp.take_along_axis(logs, batch.teacher_actions[:, None], axis=-1)[:, 0]
        bidding_mask = batch.teacher_mask * batch.policy_mask
        card_mask = (batch.actions < 52) * batch.policy_mask
        bidding_anchor_loss = -_masked_mean(teacher_logp, bidding_mask)
        # When guidance is disabled collect deliberately skips generating labels.
        play_anchor_loss = (-_masked_mean(teacher_logp, card_mask)
                            if cfg['play_anchor_coef'] > 0 else jnp.float32(0.))
        agreement = (jnp.argmax(logs, axis=-1) == batch.teacher_actions).astype(jnp.float32)
        bid_accuracy = _masked_mean(agreement, bidding_mask)
        play_accuracy = (_masked_mean(agreement, card_mask)
                         if cfg['play_anchor_coef'] > 0 else jnp.float32(0.))
        loss = (policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
                + cfg['bidding_anchor_coef'] * bidding_anchor_loss
                + cfg['play_anchor_coef'] * play_anchor_loss)
        return loss, jnp.stack([loss, policy_loss, value_loss, entropy, approx_kl,
                               clip_fraction, bidding_anchor_loss, play_anchor_loss,
                               bid_accuracy, play_accuracy])

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
    return train_step


def make_bootstrap_step(cfg):
    """Coherent teacher initialization before PPO and on-policy guidance.

    Each call gathers one fresh rollout of scripted games and optimizes actor
    cross-entropy. Teachers are imperfect; their advice is a training aid.
    Call finish_bootstrap once afterward to start clean self-play matches.
    """
    cfg = learning_config(cfg)
    model, tx = model_for_config(cfg), _optimizer(cfg)
    n, horizon = cfg["n_env"], cfg["rollout_len"]
    total = n * horizon
    teacher_minibatch = min(cfg['bootstrap_minibatch_size'], total)
    nb = total // teacher_minibatch

    @jax.jit
    def collect(state):
        def teacher_turn(carry, unused):
            envs, key = carry
            key, k_action, k_step = jax.random.split(key, 3)
            obs, legal = jax.vmap(observe)(envs), jax.vmap(legal_mask)(envs)
            labels = jax.vmap(lambda env, key: teacher_action(env, key, cfg))(
                envs, jax.random.split(k_action, n))
            envs, _, _, _ = batch_step(envs, labels, jax.random.split(k_step, n))
            return (envs, key), (obs, legal, labels)

        (envs, key), data = jax.lax.scan(teacher_turn, (state.envs, state.rng), None, length=horizon)
        data = jax.tree_util.tree_map(lambda a: a.reshape((total,) + a.shape[2:]), data)
        return state._replace(envs=envs, rng=key), data

    @jax.jit
    def optimize(state, data):
        def supervised_loss(params, mini):
            obs, legal, labels = mini
            logits = model.apply({"params": params}, obs, method=model.actor)
            logs = _masked_log_probs(logits, legal)
            selected = jnp.take_along_axis(logs, labels[:, None], axis=-1)[:, 0]
            mask = (jnp.sum(legal, axis=-1) > 1).astype(jnp.float32)
            loss = -_masked_mean(selected, mask)
            accuracy = _masked_mean((jnp.argmax(logs, axis=-1) == labels).astype(jnp.float32), mask)
            return loss, accuracy

        def epoch(carry, unused):
            params, opt_state, key, stopped = carry
            key, k_shuffle = jax.random.split(key)
            permutation = jax.random.permutation(k_shuffle, total)
            minis = jax.tree_util.tree_map(
                lambda a: a[permutation].reshape((nb, teacher_minibatch) + a.shape[1:]), data)

            def one(carry, mini):
                params, opt_state, stopped = carry

                def update(args):
                    params, opt_state = args
                    (loss, accuracy), grads = jax.value_and_grad(supervised_loss, has_aux=True)(params, mini)
                    grad_norm = optax.global_norm(grads)
                    updates, candidate_opt = tx.update(grads, opt_state, params)
                    candidate = optax.apply_updates(params, updates)
                    diagnostics = jnp.array([loss, accuracy, grad_norm])
                    checks = _update_checks(diagnostics, grads, grad_norm, candidate, candidate_opt)
                    finite = jnp.all(checks)
                    params, opt_state = jax.lax.cond(finite, lambda _: (candidate, candidate_opt),
                                                    lambda _: (params, opt_state), operand=None)
                    return (params, opt_state, ~finite), (diagnostics,
                        jnp.array([1, finite.astype(jnp.int32)], jnp.int32), checks)

                return jax.lax.cond(stopped,
                    lambda _: ((params, opt_state, stopped),
                               (jnp.zeros(3), jnp.zeros(2, jnp.int32),
                                jnp.ones(len(_UPDATE_CHECK_NAMES), jnp.bool_))),
                    update, (params, opt_state))

            (params, opt_state, stopped), rows = jax.lax.scan(one, (params, opt_state, stopped), minis)
            return (params, opt_state, key, stopped), _reduce_update_rows(rows)

        (params, opt_state, key, _), rows = jax.lax.scan(
            epoch, (state.params, state.opt_state, state.rng, jnp.bool_(False)),
            None, length=cfg["bootstrap_epochs"])
        metrics = _update_metrics(rows, ("bootstrap_loss", "bootstrap_accuracy", "bootstrap_grad_norm"))
        metrics["bootstrap_updates"] = state.bootstrap_updates + 1
        return state._replace(params=params, opt_state=opt_state, rng=key,
                              bootstrap_updates=state.bootstrap_updates + 1), metrics

    def bootstrap_step(state):
        state, data = collect(state)
        return optimize(state, data)

    bootstrap_step.collect = collect
    bootstrap_step.optimize = optimize
    return bootstrap_step


def finish_bootstrap(state, cfg):
    """Commit warmed weights to the league and reset games/optimizer for PPO."""
    cfg = learning_config(cfg)
    key, k_env, k_league = jax.random.split(state.rng, 3)
    kind, style, slot, team = _league_assignments(k_league, cfg, jnp.int32(1), jnp.int32(-1))
    bank = jax.tree_util.tree_map(lambda p: jnp.repeat(p[None], cfg["snapshot_size"], axis=0), state.params)
    return state._replace(opt_state=_optimizer(cfg).init(state.params), rng=key,
                          envs=jax.vmap(reset)(jax.random.split(k_env, cfg["n_env"])),
                          league_kind=kind, league_style=style, league_snapshot=slot, learner_team=team,
                          snapshot_params=bank, snapshot_count=jnp.int32(1),
                          snapshot_cursor=jnp.int32(1 % cfg["snapshot_size"]),
                          snapshot_retiring=jnp.int32(-1), last_snapshot_update=state.updates,
                          bootstrap_complete=jnp.bool_(True))


def recover_legacy_training(state, cfg):
    """Retain the policy/counters and prepare an explicit trainer migration.

    Reset the critic fitted to pathological trajectories, Adam moments, deeply
    negative games, and stale league bank. Actor parameter tensors transfer
    exactly before the teacher refresh; extreme-score input encoding changes.
    The caller records this change and saves into a separate recovery folder.
    """
    cfg = learning_config(cfg)
    if int(state.decisions_per_update) != cfg['n_env'] * cfg['rollout_len']:
        raise ValueError('Recovery must keep the saved environment/rollout configuration.')
    fresh = initialize_training(cfg)
    old_actor = jax.tree_util.tree_leaves(state.params['actor_network'])
    new_actor = jax.tree_util.tree_leaves(fresh.params['actor_network'])
    if len(old_actor) != len(new_actor) or any(a.shape != b.shape for a, b in zip(old_actor, new_actor)):
        raise ValueError('Recovery must keep the saved model architecture.')
    params = dict(fresh.params)
    params['actor_network'] = state.params['actor_network']
    fresh = fresh._replace(params=params, rng=state.rng, updates=state.updates,
                            decisions_per_update=state.decisions_per_update,
                            env_steps_low=state.env_steps_low, env_steps_high=state.env_steps_high,
                            base_updates=state.base_updates)
    # This creates fresh games and snapshots from the transferred policy with
    # a fresh critic, and derives their random keys from the saved RNG stream.
    fresh = finish_bootstrap(fresh, cfg)
    return fresh._replace(bootstrap_updates=jnp.int32(0), bootstrap_complete=jnp.bool_(False))


def upgrade_training_workload(state, cfg, reset_critic=False):
    """Transfer learned weights into an explicit new batch/teacher experiment.

    Old files are never modified. Only the actor/critic parameters and random
    stream transfer; optimizer, games and league are initialized for the new
    workload. Lifetime decisions and update numbers remain exact after resizing.
    """
    cfg = learning_config(cfg)
    fresh = initialize_training(cfg)
    for name in ('actor_network', 'critic_network'):
        old, old_structure = jax.tree_util.tree_flatten(state.params[name])
        new, new_structure = jax.tree_util.tree_flatten(fresh.params[name])
        if old_structure != new_structure or any(a.shape != b.shape for a, b in zip(old, new)):
            raise ValueError('Workload migration must preserve the saved network width and architecture.')
    params = dict(state.params)
    if reset_critic:
        params['critic_network'] = fresh.params['critic_network']
    decisions = state.env_steps
    if not 0 <= decisions < 2 ** 64:
        raise ValueError('Lifetime decisions exceed the supported unsigned 64-bit counter.')
    fresh = fresh._replace(params=params, rng=state.rng, updates=state.updates,
                          env_steps_low=jnp.uint32(decisions & 0xffffffff),
                          env_steps_high=jnp.uint32(decisions >> 32), base_updates=state.updates)
    fresh = finish_bootstrap(fresh, cfg)
    return fresh._replace(bootstrap_updates=jnp.int32(0), bootstrap_complete=jnp.bool_(False))
