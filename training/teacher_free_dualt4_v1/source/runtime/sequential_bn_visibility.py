"""Explicit public BN response status without changing native/model dimensions.

Definitions and source-only tools; no checkpoint, training, provider or git work.
Install before compiling a new reviewed runtime. Issued source pins stay immutable.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

RULES_SHA256='d999807d27a11e25e0e326f0309d41ed6982a810bbec78b47f50f068deb488d1'
NOTEBOOK_SHA256={
    '3236d2c95d4d1a95bf31a5cb7299853be6871783691829ee775fbb38366d1389',
    'ddb7eae810d3ac37e005dddbcdf90399797610ac802cce9cdbcda99a784e91c8'}

PATCH_SOURCE='''# Sequential public blind-nil responses, version1.
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
'''


def identity(raw):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read_rules(notebook_path):
    path=Path(notebook_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Existing regular source notebook required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest() not in NOTEBOOK_SHA256:
        raise ValueError('Exact reviewed original or production source notebook required')
    rules=''.join(json.loads(raw)['cells'][6]['source'])
    if hashlib.sha256(rules.encode()).hexdigest()!=RULES_SHA256:
        raise ValueError('Exact reviewed engine rules cell6 required')
    return rules


def patched_rules_source(original_rules):
    if hashlib.sha256(original_rules.encode()).hexdigest()!=RULES_SHA256:
        raise ValueError('Exact unchanged base rules required; already-patched source rejected')
    result=original_rules+'\n\n'+PATCH_SOURCE
    ast.parse(result)
    return result


def install(namespace, source_notebook_path):
    """Apply once to the exact base rules namespace, before any JIT compilation.

    This does not change weights or load/save a checkpoint. A future production
    integration must explicitly update reviewed source pins, never an old grant.
    """
    rules=read_rules(source_notebook_path)
    if getattr(namespace.get('_observe_for'),'_sequential_bn_visibility_v1',False):
        raise ValueError('Sequential BN visibility is already installed')
    if (namespace.get('OBS_DIM'),namespace.get('CRITIC_DIM'),namespace.get('N_ACTIONS'))!=(753,1173,68):
        raise ValueError('Exact compatible base observation/action dimensions required')
    names={'_observe_for','_blind_turn','observe','critic_observe','actor','phase','legal_mask'}
    nodes=[n for n in ast.parse(rules).body if isinstance(n,ast.FunctionDef) and n.name in names]
    expected={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'expected_rule_definitions_only.py','exec'),expected)
    fields=('co_code','co_consts','co_names','co_varnames','co_argcount','co_kwonlyargcount')
    for name in names:
        current=namespace.get(name)
        if (not callable(current) or current.__globals__ is not namespace
                or any(getattr(current.__code__,field)!=getattr(expected[name].__code__,field) for field in fields)):
            raise ValueError('Exact unchanged rules callable required: '+name)
    exec(compile(PATCH_SOURCE,'sequential_bn_visibility_v1.py','exec'),namespace)
    return {'format':'spades-sequential-bn-public-status-v1','base_rules_sha256':RULES_SHA256,
        'patch_source':identity(PATCH_SOURCE.encode()),'status_codes':{'pending':0,'declined':1,'declared_bn':2},
        'actor_features':753,'critic_features':1173,'actions':68,'state_fields_changed':False,
        'checkpoint_array_migration_required':False,'reachable_policy_observation_semantics_changed':False,
        'training_started':False,'old_source_pins_reusable':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=patched_rules_source(read_rules(args.notebook)).encode()
    output=args.output.absolute()
    if output.suffix!='.py' or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('New nonsymlink Python source output required')
    with output.open('xb') as stream:stream.write(source)
    print(json.dumps({'status':'SOURCE_ONLY_WRITTEN','output':str(output),**identity(source),
        'original_source_modified':False,'checkpoint_modified':False,'job_launched':False},sort_keys=True))


if __name__=='__main__':main()
