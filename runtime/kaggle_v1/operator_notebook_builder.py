"""Build the fixed, data-input-only Kaggle operator for the AOT long-run API.

This builder and its self-tests never import JAX or execute a preparation/run.
The runner is owned separately. Only the final payload/operator hashes are baked
into the generated notebook; no binary payload or runner source is embedded.
"""

from __future__ import annotations

# The sibling runner is intentionally named operator.py. Remove this script's
# directory from CLI import search before stdlib enum imports stdlib operator.
import sys
if __name__ == '__main__' and sys.path:
    sys.path.pop(0)

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile


PAYLOAD_SCHEMA = 'spades-aot-kaggle-payload-v1'
PAYLOAD_MEMBERS = (
    'original.ipynb', 'production.ipynb', 'operator.py',
    'coordination_TEMPLATE.json', 'bundle_manifest.json',
)
PAYLOAD_LIMITS = {
    'original.ipynb': 377176,
    'production.ipynb': 429376,
    'operator.py': 512 * 1024,
    'coordination_TEMPLATE.json': 128 * 1024,
    'bundle_manifest.json': 64 * 1024,
}
ORIGINAL_SHA256 = '56da4fa7bc6e501249cecbd4e41768b27f3930b09fc8938c0fb8f1201a19c28e'
PRODUCTION_SHA256 = '16ee24065655d9d13d13896befe50772c65aa7dbcfda85e336868f42cb7ea494'
CANDIDATE_PACKAGE_SHA256 = '2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13'
CANDIDATE_UPDATE = 674
CANDIDATE_ENV_STEPS = 309198848
MIGRATION_AUTHORIZED_UTC = '2026-10-18T22:33:33Z'
PREPARATION_CAP_UTC = '2026-10-19T15:49:14Z'
REVIEW_GATES = (
    'run_authorized', 'parent_review_complete', 'runtime_exclusive',
    'candidate_native_export_verified',
)


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + str(key))
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError('Nonfinite JSON constant is unsupported: ' + value)


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('JSON numeric literal overflows a finite float.')
    return number


def parse_json_object(text):
    value = json.loads(text, object_pairs_hook=_unique_json_pairs,
                       parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    if not isinstance(value, dict):
        raise ValueError('Expected one JSON object.')
    return value


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def operator_definitions_tree(source):
    """Permit module docstring and inert class/function definitions only."""
    tree = ast.parse(source, filename='verified_operator.py')

    def function(node):
        if node.decorator_list:
            raise ValueError('Operator decorators are not definitions-only.')
        for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
            ast.literal_eval(default)
        annotations = [a.annotation for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
        annotations += [node.returns]
        annotations += [a.annotation for a in (node.args.vararg, node.args.kwarg) if a is not None]
        for annotation in annotations:
            if annotation is not None:
                if not isinstance(annotation, (ast.Name, ast.Constant)):
                    raise ValueError('Operator annotations may not evaluate expressions.')
                if isinstance(annotation, ast.Name) and annotation.id not in {
                        'dict', 'str', 'int', 'float', 'bool', 'list', 'tuple', 'object'}:
                    raise ValueError('Unreviewed operator annotation name.')

    names = set()
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {'RuntimeError', 'Exception'}:
            raise ValueError('Operator definitions may not rebind permitted built-in class bases.')
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, ast.FunctionDef):
            function(node)
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.decorator_list or node.keywords or any(
                    not isinstance(base, ast.Name) or base.id not in {'RuntimeError', 'Exception'}
                    for base in node.bases):
                raise ValueError('Operator class bases/decorators must be inert built-ins.')
            for child_index, child in enumerate(node.body):
                if isinstance(child, ast.Pass):
                    continue
                if (child_index == 0 and isinstance(child, ast.Expr)
                        and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str)):
                    continue
                if isinstance(child, ast.FunctionDef):
                    function(child)
                    continue
                raise ValueError('Operator class body contains executable setup.')
        else:
            raise ValueError('Operator module must contain only its docstring, classes and functions.')
    required = {'prepare_aot_long_run', 'run_until_barrier', 'ack_export', 'next_segment', 'preflight_kaggle_nojax', 'inventory_kaggle_nojax'}
    if not required <= names:
        raise ValueError('Missing stable operator API: ' + str(sorted(required - names)))
    return tree


def verify_and_extract_payload(payload_path, *, expected_sha256, expected_bytes,
                               expected_operator_sha256, extract_parent):
    """Verify everything first, then write a brand-new private extraction dir."""
    path = Path(payload_path).expanduser().resolve(strict=True)
    if not _is_sha256(expected_sha256) or not _is_sha256(expected_operator_sha256):
        raise ValueError('Exact lowercase SHA-256 pins are required.')
    if type(expected_bytes) is not int or not 0 < expected_bytes <= 16 * 2**20:
        raise ValueError('Payload byte count must be an exact bounded positive integer.')
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError('Payload byte count differs from the frozen pin.')
    if _file_sha256(path) != expected_sha256:
        raise ValueError('Payload SHA-256 differs from the frozen pin.')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [entry.filename for entry in infos]
        if len(names) != len(set(names)) or set(names) != set(PAYLOAD_MEMBERS):
            raise ValueError('Payload must contain exactly the five allowed root files.')
        for entry in infos:
            mode = entry.external_attr >> 16
            if (entry.is_dir() or entry.flag_bits & 1 or stat.S_ISLNK(mode)
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or entry.file_size <= 0 or entry.file_size > PAYLOAD_LIMITS[entry.filename]):
                raise ValueError('Unsafe or oversized payload member: ' + entry.filename)
        if archive.testzip() is not None:
            raise ValueError('Payload ZIP CRC verification failed.')
        blobs = {name: archive.read(name) for name in PAYLOAD_MEMBERS}
    manifest = parse_json_object(blobs['bundle_manifest.json'].decode('utf-8'))
    if set(manifest) != {'schema', 'members'} or manifest['schema'] != PAYLOAD_SCHEMA:
        raise ValueError('Unsupported payload manifest schema.')
    specs = manifest['members']
    expected_names = set(PAYLOAD_MEMBERS) - {'bundle_manifest.json'}
    if not isinstance(specs, dict) or set(specs) != expected_names:
        raise ValueError('Manifest must pin exactly four non-manifest members.')
    for name in sorted(expected_names):
        spec, data = specs[name], blobs[name]
        if (not isinstance(spec, dict) or set(spec) != {'bytes', 'sha256'}
                or type(spec['bytes']) is not int or spec['bytes'] != len(data)
                or not _is_sha256(spec['sha256'])
                or hashlib.sha256(data).hexdigest() != spec['sha256']):
            raise ValueError('Member hash/byte pin mismatch: ' + name)
    pins = {'original.ipynb': ORIGINAL_SHA256, 'production.ipynb': PRODUCTION_SHA256,
            'operator.py': expected_operator_sha256}
    for name, digest in pins.items():
        if specs[name]['sha256'] != digest:
            raise ValueError('Frozen source/operator pin mismatch: ' + name)
    for name in ('original.ipynb', 'production.ipynb'):
        if len(blobs[name]) != PAYLOAD_LIMITS[name]:
            raise ValueError('Frozen notebook byte count differs: ' + name)
        notebook = parse_json_object(blobs[name].decode('utf-8'))
        if len(notebook.get('cells', [])) != 38:
            raise ValueError('Expected the frozen 38-cell notebook: ' + name)
    template = parse_json_object(blobs['coordination_TEMPLATE.json'].decode('utf-8'))
    if any(template.get(key) is not False for key in REVIEW_GATES):
        raise ValueError('Coordination template review gates must remain explicitly false.')
    operator_tree = operator_definitions_tree(blobs['operator.py'].decode('utf-8'))
    parent = Path(extract_parent).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError('Extraction parent must already exist.')
    root = Path(tempfile.mkdtemp(prefix='spades_aot_payload_', dir=str(parent)))
    try:
        for name, data in blobs.items():
            with (root / name).open('xb') as stream:
                stream.write(data)
    except BaseException:
        shutil.rmtree(root)
        raise
    return dict(root=root, paths={name: root / name for name in PAYLOAD_MEMBERS},
                manifest=manifest, coordination_template=template, operator_tree=operator_tree,
                payload_sha256=expected_sha256, operator_sha256=expected_operator_sha256)


def _markdown(text, identity):
    return dict(cell_type='markdown', id=identity, metadata={}, source=text.splitlines(keepends=True))


def _code(text, identity):
    return dict(cell_type='code', id=identity, metadata={}, execution_count=None,
                outputs=[], source=text.splitlines(keepends=True))


def build_operator_notebook(*, payload_path, payload_sha256, payload_bytes,
                            operator_sha256, output_path):
    if not _is_sha256(payload_sha256) or not _is_sha256(operator_sha256):
        raise ValueError('Supply final payload and operator lowercase SHA-256 pins.')
    if type(payload_bytes) is not int or not 0 < payload_bytes <= 16 * 2**20:
        raise ValueError('Supply the final payload byte count.')
    if not isinstance(payload_path, str) or not payload_path.strip():
        raise ValueError('Supply the Kaggle payload path hint.')
    helper_names = (
        '_unique_json_pairs', '_reject_json_constant', '_finite_json_float', 'parse_json_object', '_is_sha256',
        '_file_sha256', 'operator_definitions_tree', 'verify_and_extract_payload',
    )
    loader = '''# Stage the pinned payload in private writable Kaggle storage first.
# This cell uses stdlib only; it does not connect a runtime or import JAX.
import ast, hashlib, json, math, os, re, shutil, stat, sys, tempfile, zipfile
from pathlib import Path
if 'AOT_SESSION' in globals() or globals().get('RECOVERY'):
    raise RuntimeError('Existing session/recovery retained. Do not reload; read status after return.')
'''
    for name in ('PAYLOAD_SCHEMA', 'PAYLOAD_MEMBERS', 'PAYLOAD_LIMITS',
                 'ORIGINAL_SHA256', 'PRODUCTION_SHA256', 'REVIEW_GATES'):
        loader += name + ' = ' + repr(globals()[name]) + '\n'
    loader += 'EXPECTED_PAYLOAD_SHA256 = ' + repr(payload_sha256) + '\n'
    loader += 'EXPECTED_PAYLOAD_BYTES = ' + repr(payload_bytes) + '\n'
    loader += 'EXPECTED_OPERATOR_SHA256 = ' + repr(operator_sha256) + '\n'
    loader += '\n'.join(inspect.getsource(globals()[name]).rstrip() for name in helper_names) + '\n\n'
    loader += 'DEFAULT_PAYLOAD_PATH = ' + repr(payload_path) + '\n'
    loader += '''KAGGLE_WORKING = Path('/kaggle/working')
if not KAGGLE_WORKING.is_dir() or not os.access(KAGGLE_WORKING, os.W_OK):
    raise RuntimeError('Expected existing writable /kaggle/working; no alternate platform fallback.')
payload_request_json = input('Pinned payload path JSON (blank uses displayed default): ')
if payload_request_json.strip():
    PAYLOAD_REQUEST = parse_json_object(payload_request_json)
    if (set(PAYLOAD_REQUEST) != {'payload_path'}
            or not isinstance(PAYLOAD_REQUEST['payload_path'], str)
            or not PAYLOAD_REQUEST['payload_path'].strip()):
        raise ValueError('Supply exactly one nonempty payload_path string.')
    payload_path = PAYLOAD_REQUEST['payload_path']
else:
    payload_path = DEFAULT_PAYLOAD_PATH
'''
    loader += '''PAYLOAD = verify_and_extract_payload(
    payload_path, expected_sha256=EXPECTED_PAYLOAD_SHA256,
    expected_bytes=EXPECTED_PAYLOAD_BYTES, expected_operator_sha256=EXPECTED_OPERATOR_SHA256,
    extract_parent=KAGGLE_WORKING)
OPERATOR_NS = {'__name__': '_spades_aot_operator', '__file__': str(PAYLOAD['paths']['operator.py'])}
exec(compile(PAYLOAD['operator_tree'], str(PAYLOAD['paths']['operator.py']), 'exec'), OPERATOR_NS)
prepare_aot_long_run = OPERATOR_NS['prepare_aot_long_run']
run_until_barrier = OPERATOR_NS['run_until_barrier']
ack_export = OPERATOR_NS['ack_export']
next_segment = OPERATOR_NS['next_segment']
preflight_kaggle_nojax = OPERATOR_NS['preflight_kaggle_nojax']
inventory_kaggle_nojax = OPERATOR_NS['inventory_kaggle_nojax']
print('Payload verified; operator definitions loaded. No JAX preparation or training was called.')
print('Private payload directory:', PAYLOAD['root'])
print('Coordination template (review gates are false; complete verified data before preparation):')
print(json.dumps(PAYLOAD['coordination_template'], indent=2, allow_nan=False))
'''
    preflight = '''# Inventory an already allocated, authorized runtime; this cell does not allocate a GPU.
# Do not run this notebook through Run All: each JSON/ACK requires an idle reviewed barrier.
if 'AOT_SESSION' in globals() or globals().get('RECOVERY'):
    raise RuntimeError('Existing session/recovery retained; do not rerun platform preflight.')
KAGGLE_INVENTORY = inventory_kaggle_nojax()
print(json.dumps(KAGGLE_INVENTORY, indent=2, allow_nan=False))
preflight_request_json = input('Kaggle preflight JSON with selected_gpu_uuid (blank cancels): ')
if not preflight_request_json.strip():
    print('No preflight called. Supply an actual physical T4 UUID after normal runtime allocation.')
else:
    PREFLIGHT_REQUEST = parse_json_object(preflight_request_json)
    if (set(PREFLIGHT_REQUEST) != {'selected_gpu_uuid'}
            or not isinstance(PREFLIGHT_REQUEST['selected_gpu_uuid'], str)
            or not PREFLIGHT_REQUEST['selected_gpu_uuid'].strip()):
        raise ValueError('Supply exactly one nonempty selected_gpu_uuid string.')
    KAGGLE_PREFLIGHT = preflight_kaggle_nojax(
        selected_gpu_uuid=PREFLIGHT_REQUEST['selected_gpu_uuid'])
    print(json.dumps(KAGGLE_PREFLIGHT, indent=2, allow_nan=False))
    print('Preflight returned. Review actual inventory and matching UUID before preparation.')
'''
    preparation = '''# Fixed stdin data input; parent/Luna supplies JSON without editing this cell.
prepare_request_json = input('Operator prepare JSON (blank cancels): ')
if not prepare_request_json.strip():
    print('Supply JSON with candidate_package_path, output_dir and coordination. No preparation called.')
else:
    if 'AOT_SESSION' in globals() or globals().get('RECOVERY'):
        raise RuntimeError('Existing session/recovery retained; do not prepare another session.')
    if 'KAGGLE_PREFLIGHT' not in globals():
        raise RuntimeError('Run and review the fixed Kaggle preflight before preparation.')
    PREPARE_REQUEST = parse_json_object(prepare_request_json)
    if set(PREPARE_REQUEST) != {'candidate_package_path', 'output_dir', 'coordination'}:
        raise ValueError('Prepare JSON must have exactly candidate_package_path, output_dir and coordination.')
    if (not isinstance(PREPARE_REQUEST['coordination'], dict)
            or any(not isinstance(PREPARE_REQUEST[key], str) or not PREPARE_REQUEST[key].strip()
                   for key in ('candidate_package_path', 'output_dir'))):
        raise ValueError('Supply nonempty paths and an actual coordination object.')
    if PREPARE_REQUEST['coordination'].get('selected_gpu_uuid') != KAGGLE_PREFLIGHT.get('platform', {}).get('selected_gpu_uuid'):
        raise ValueError('Coordination UUID must match this session preflight exactly.')
    output_target = Path(PREPARE_REQUEST['output_dir']).expanduser().resolve()
    if Path('/kaggle/working') not in output_target.parents:
        raise ValueError('A brand-new output directory inside /kaggle/working is required.')
    RECOVERY = {}
    try:
        AOT_SESSION = prepare_aot_long_run(
            original_path=PAYLOAD['paths']['original.ipynb'],
            production_path=PAYLOAD['paths']['production.ipynb'],
            candidate_package_path=PREPARE_REQUEST['candidate_package_path'],
            coordination=PREPARE_REQUEST['coordination'],
            output_dir=PREPARE_REQUEST['output_dir'], recovery_sink=RECOVERY)
        COORDINATION_PATH = Path(AOT_SESSION['root']) / 'fixed_coordination.json'
        with COORDINATION_PATH.open('x') as coordination_stream:
            coordination_stream.write(json.dumps(PREPARE_REQUEST['coordination'], indent=2,
                                                 sort_keys=True, allow_nan=False) + '\\n')
    except BaseException:
        print('Preparation stopped. Keep RECOVERY and all accepted files; do not rerun unchanged.')
        raise
    print('Preparation returned. Display the initial export barrier with the smoke/status cell next.')
'''
    smoke = '''# First call displays initial export; after its real ACK, one accepted update is the smoke.
if 'AOT_SESSION' not in globals():
    raise RuntimeError('No prepared live session. Do not bypass preparation.')
SMOKE_STATUS = run_until_barrier(AOT_SESSION, max_updates=1)
print(json.dumps(SMOKE_STATUS, indent=2, allow_nan=False))
if SMOKE_STATUS.get('status') == 'export_required':
    from IPython.display import FileLink, display
    display(FileLink(SMOKE_STATUS['package_path']))
    display(FileLink(str(Path(AOT_SESSION['root']) / 'status.json')))
    display(FileLink(str(Path(AOT_SESSION['root']) / 'fixed_coordination.json')))
print('Call returned: export/verify/ACK when required. Do not queue another cell while training is busy.')
'''
    acknowledgement = '''# Run only after training returned. Parent/Luna supplies verified receipt data via stdin.
ack_receipt_json = input('Verified Library receipt JSON (blank cancels): ')
if not ack_receipt_json.strip():
    print('No ACK called. A local path or file link alone is not durable persistence.')
else:
    if 'AOT_SESSION' not in globals():
        raise RuntimeError('No live session; inspect runtime continuity before any recovery.')
    ACK_RECEIPT = parse_json_object(ack_receipt_json)
    ACK_STATUS = ack_export(AOT_SESSION, ACK_RECEIPT)
    print(json.dumps(ACK_STATUS, indent=2, allow_nan=False))
    print('ACK returned. It records persistence only; it does not reset a health stop or start training.')
'''
    run = '''# Explicitly resume the same segment. Returns at 15-minute durability, p90, final or a stop.
if 'AOT_SESSION' not in globals():
    raise RuntimeError('No prepared live session; never recreate a lost session silently.')
RUN_STATUS = run_until_barrier(AOT_SESSION, max_updates=None)
print(json.dumps(RUN_STATUS, indent=2, allow_nan=False))
if RUN_STATUS.get('status') == 'export_required':
    from IPython.display import FileLink, display
    display(FileLink(RUN_STATUS['package_path']))
    display(FileLink(str(Path(AOT_SESSION['root']) / 'status.json')))
    display(FileLink(str(Path(AOT_SESSION['root']) / 'fixed_coordination.json')))
print('Call returned. Follow the current export/ACK status; no automatic new segment.')
'''
    next_segment_cell = '''# Optional explicit transition only after the prior segment-complete export ACK.
# This cell starts no GPU work. The reviewed max_segments bound and authorization still apply.
next_segment_request_json = input('Next segment confirmation JSON (blank cancels): ')
if not next_segment_request_json.strip():
    print('No segment transition called. Current accepted state remains retained.')
else:
    if 'AOT_SESSION' not in globals():
        raise RuntimeError('No live session. Do not silently restore or initialize another one.')
    NEXT_SEGMENT_REQUEST = parse_json_object(next_segment_request_json)
    if NEXT_SEGMENT_REQUEST != {'action': 'next_segment'}:
        raise ValueError('The only accepted confirmation object is {"action":"next_segment"}.')
    NEXT_SEGMENT_STATUS = next_segment(AOT_SESSION)
    print(json.dumps(NEXT_SEGMENT_STATUS, indent=2, allow_nan=False))
    print('Transition returned. Resume only if permitted, using the existing continue cell.')
'''
    status = '''# READ ONLY. Invoke only after the control call has returned and the kernel is idle.
if 'AOT_SESSION' in globals():
    status_path = Path(AOT_SESSION['root']) / 'status.json'
    if status_path.is_file():
        print(json.dumps(parse_json_object(status_path.read_text()), indent=2, allow_nan=False))
        from IPython.display import FileLink, display
        display(FileLink(str(status_path)))
        coordination_path = Path(AOT_SESSION['root']) / 'fixed_coordination.json'
        if coordination_path.is_file():
            display(FileLink(str(coordination_path)))
        package_path = Path(AOT_SESSION['root']) / 'export_package.zip'
        if package_path.is_file():
            display(FileLink(str(package_path)))
    else:
        print('No status file at', status_path, '; preserve live session and inspect the failure.')
else:
    print('No live AOT_SESSION. Do not silently initialize another one.')
print('Recovery keys only (state arrays are retained, not displayed):', sorted(globals().get('RECOVERY', {})))
'''
    cells = [
        _markdown('''# Spades — Kaggle single-T4 migration candidate

This is a separate platform migration of preserved C3, authorized 2026-10-18 22:33:33 UTC.
It is host-reviewed and has **no completed Kaggle GPU validation**. Open a private interactive
notebook through the normal free Kaggle interface only after account eligibility and quota are
confirmed. This code never allocates compute, purchases units, authenticates, installs packages,
or reconnects a lost runtime. No paid option or quota bypass is authorized.

**Do not use Run All or a batch execution expecting stdin.** Execute the fixed cells individually,
review each returned result, and supply plain JSON data at the explicit input prompts. The browser
operator performs downloads, Library upload/readback and verified ACK after the kernel returns idle.
No credentials or SDK are used. Kaggle local files and saved notebook output alone do not satisfy ACK.

Stage the separate pinned payload and verified P90 update674 package in private storage.
Its package SHA-256 is `2b38d9596fc75cf90eda5bad18dfb3c0b54881dc5346c0eab38f2b86559bed13`;
it contains update674 / 309,198,848 decisions. Later update1075 was not durably exported.
Preserve the inputs unchanged. New accepted checkpoint/export files go in a brand-new
`/kaggle/working` subdirectory; file links below are download conveniences, not persistence proof.

The original combined-JIT next-update continuity mismatch remains unresolved. This separate AOT
route makes no CUDA12 numerical-equivalence, expert-strength, promotion or long-run reliability claim.
One physical T4 is used, with UUID-bound telemetry; two T4s are never combined into one32GB budget.
No model, training configuration, acceptance, game rule or opponent change is part of this migration.
''', 'aot-intro'),
        _code(loader, 'aot-verify-load'),
        _markdown('''## Inventory and select the already allocated physical T4

Use the normal provider interface to allocate an eligible free runtime only once authorized.
The separate preflight records actual installed package versions and physical-board inventory,
checks the selected T4 UUID and driver, without importing JAX. Preparation sets the reviewed UUID before importing JAX.
The physical-board inventory is displayed before the fixed UUID-selection input. It never allocates.
Supply exactly `{"selected_gpu_uuid":"GPU-actual-uuid"}`. A version, mapping, driver, visibility
or memory mismatch must stop. Do not install packages, alter flags or try a different backend.
Keep its returned inventory with the run evidence. Preparation repeats the runtime gates.
''', 'aot-preflight-help'),
        _code(preflight, 'aot-preflight-data'),
        _markdown('''## Prepare from exact native state

Supply plain JSON through this fixed cell's stdin prompt, without editing the cell.
Use exactly `candidate_package_path`, a **brand-new** `output_dir`, and the complete
internally reviewed `coordination` object. The payload template has false review gates;
the coordinator fills real authorization, source pins, expiry and native-export evidence.
The exact pinned package must have only `native_checkpoint.zip` and `export_manifest.json`.
Package, manifest, derived native identity, stack/configuration, all 214 leaves and saved counters
are verified before preparation. Supply the outer package path; never substitute a loose native.
Use an eligible free Kaggle T4 runtime with actual JAX/JAXlib0.11.1, Flax0.12.10 and Optax0.2.8.
Record Python, CUDA plugin, driver and selected-board identity; CUDA12 equivalence is unproven.
Do not pre-import JAX. A mismatch stops preparation; this notebook contains no installer or repair path.
The reviewed `prepare_valid_until_utc` must not exceed `2026-10-19T15:49:14Z`;
the independent run authorization must cover preparation and the bounded requested run.
Preparation creates an initial pending export; **no training precedes its verified ACK**.
Keep `RECOVERY` if preparation stops. Do not prepare again or clear a healthy live session.
''', 'aot-prepare-help'),
        _code(preparation, 'aot-prepare-data'),
        _markdown('''## Initial barrier, then one-update smoke

Run this cell once to show the initial returned barrier. Export the exact package, independently
verify the complete native member, upload to Library, then use the ACK cell below.
Use the matching Kaggle migration standalone export auditor supplied in the delivery bundle; the package's
external manifest uses `spades-aot-external-export-kaggle-v1`. A successful host audit alone is not a Library readback.
Run this smoke cell again after that initial ACK for one accepted update and its returned barrier.
Download the exact package, `status.json` and generated inner `fixed_coordination.json`, preserving their bytes. If FileLink is unsupported
by Kaggle, use its private Files / Output panel for these already completed files; do not Save & Run All.
Use a fresh host-audit report, Library-upload the package, independently download it back and rerun
the auditor before ACK. Export and ACK the post-update native checkpoint too. Real export → readback → ACK → same-session
resume must work on the actual T4 before claiming launch readiness. One smoke update is not a long-run test.
''', 'aot-smoke-help'),
        _code(smoke, 'aot-smoke'),
        _markdown('''## Acknowledge only a real, current Library export

Wait until the preceding call has **returned and the kernel is idle**, then invoke this separate
fixed input cell and supply plain JSON through stdin without editing code. Supply the exact current
nonce, package/native paths, SHA-256 values and byte counts, successful Library ID, `verified: true`,
and explicit UTC `valid_until_utc`, using the reviewed contract. Never invent or reuse a stale receipt.
This cell only acknowledges persistence. A terminal, failed, expired or health-stopped session cannot
be resumed merely by acknowledging its export. If a response is lost, inspect status before retrying.
''', 'aot-ack-help'),
        _code(acknowledgement, 'aot-ack-data'),
        _markdown('''## Continue the same three-hour segment

After the initial and one-update smoke exports are verified and acknowledged, invoke this fixed cell.
It returns at each supplemental 15-minute durability, 90-minute, final or stop barrier.
Perform external export and ACK, then invoke
this same cell again only when the returned status permits continuation. The clock is not reset.
Never queue another execution while this cell is busy. No next segment starts automatically.
''', 'aot-run-help'),
        _code(run, 'aot-run'),
        _markdown('''## Optional next segment after final ACK

Use this fixed stdin cell only when the current segment returned `segment_complete` and its final
checkpoint has a verified external Library ACK. The coordinator must have reviewed a finite
`max_segments` value from 1 through 8 before preparation; the default permits only one segment.
Supply exactly `{"action":"next_segment"}`. This transition does not train. Read its returned status,
then use the existing continue cell only if allowed. Errors, health stops, expired authorization,
pending exports or a lost Python session forbid this transition. No automatic looping is provided.
''', 'aot-next-segment-help'),
        _code(next_segment_cell, 'aot-next-segment-data'),
        _markdown('''## Read-only status after return

While a training cell is busy, inspect its existing output through the frontend; a queued Python
status cell cannot prove a hang. After return, this cell reads host JSON only and shows recovery keys.
Browser disconnection does not prove kernel loss. Match run/session identity before any action.
''', 'aot-status-help'),
        _code(status, 'aot-status'),
        _markdown('''## Stop and recovery

On numerical, illegal, rejected-update, health, memory, config, compile, transport or persistence failure,
retain the last accepted full native checkpoint, pending export package, status, events and `RECOVERY`.
Do not auto-retry, resize, relax checks, acknowledge a reviewed fix that did not occur, or reset state.
No teacher/evaluation work or automatic promotion is part of this route.

If context is lost, do not silently recreate the session or claim uninterrupted continuation.
This candidate can prepare from the one exact pinned P90 input package in a fresh session;
it does not implement generic restart from an arbitrary later export. Report missing context.
A local `/kaggle/working` or `/tmp` path is not durable after VM loss.
Export final/stopped accepted work and verify its full native archive before releasing T4.
No new code, package repair, Drive OAuth or improvised transport after the freeze.
''', 'aot-recovery-help'),
    ]
    notebook = dict(cells=cells, metadata={
        'title': 'Spades_AOT_Kaggle_Single_T4_Migration_V1.ipynb',
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'spades_frozen_operator': {'schema': PAYLOAD_SCHEMA, 'payload_sha256': payload_sha256,
                                  'payload_bytes': payload_bytes, 'operator_sha256': operator_sha256,
                                  'candidate_package_sha256': CANDIDATE_PACKAGE_SHA256,
                                  'candidate_update': CANDIDATE_UPDATE,
                                  'candidate_env_steps': CANDIDATE_ENV_STEPS,
                                  'migration_authorized_utc': MIGRATION_AUTHORIZED_UTC,
                                  'preparation_cap_utc': PREPARATION_CAP_UTC,
                                  'physical_qa': 'Kaggle GPU untested; host validation only'},
    }, nbformat=4, nbformat_minor=5)
    validate_notebook_structure(notebook)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    return dict(path=str(destination), bytes=destination.stat().st_size,
                sha256=_file_sha256(destination), cells=len(cells), code_cells=8,
                payload_sha256=payload_sha256, operator_sha256=operator_sha256,
                gpu_executed=False, published=False)


def validate_notebook_structure(notebook):
    if notebook.get('nbformat') != 4 or notebook.get('nbformat_minor') != 5:
        raise ValueError('Expected notebook format4.5.')
    cells = notebook['cells']
    if len(cells) != 17 or len({c['id'] for c in cells}) != 17:
        raise ValueError('Expected17 uniquely identified fixed cells.')
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    if len(code_cells) != 8:
        raise ValueError('Expected eight fixed control/data cells.')
    for cell in cells:
        if not isinstance(cell['source'], list) or not all(isinstance(line, str) for line in cell['source']):
            raise ValueError('Notebook source must be string lines.')
        if cell['cell_type'] == 'code':
            if cell['execution_count'] is not None or cell['outputs']:
                raise ValueError('Generated operator must be unexecuted.')
            ast.parse(''.join(cell['source']))
    return True


def self_test(original_path, production_path):
    """Meaningful host-only integrity, definitions and notebook regression checks."""
    before_jax = 'jax' in sys.modules
    checks = []
    with tempfile.TemporaryDirectory(prefix='spades_operator_builder_test_') as directory:
        root = Path(directory)
        operator = b'''"""Fixture definitions only."""
class AOTLongRunBoundaryStop(RuntimeError):
    pass
def prepare_aot_long_run(**kwargs):
    raise AssertionError('Preparation must not be called by loader tests')
def run_until_barrier(session, max_updates=None):
    raise AssertionError('Training must not be called by loader tests')
def ack_export(session, receipt):
    raise AssertionError('ACK must not be called by loader tests')
def next_segment(session):
    raise AssertionError('Next segment must not be called by loader tests')
def preflight_kaggle_nojax(selected_gpu_uuid):
    raise AssertionError('Preflight must not be called by loader tests')
def inventory_kaggle_nojax():
    raise AssertionError('Inventory must not be called by loader tests')
'''
        original, production = Path(original_path).read_bytes(), Path(production_path).read_bytes()
        assert hashlib.sha256(original).hexdigest() == ORIGINAL_SHA256
        assert hashlib.sha256(production).hexdigest() == PRODUCTION_SHA256
        blobs = {'original.ipynb': original, 'production.ipynb': production,
                 'operator.py': operator,
                 'coordination_TEMPLATE.json': json.dumps({key: False for key in REVIEW_GATES}).encode()}

        def fixture(name, mutate=None, extra=None):
            values = dict(blobs)
            if mutate:
                mutate(values)
            manifest = {'schema': PAYLOAD_SCHEMA, 'members': {
                key: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                for key, value in values.items()}}
            values['bundle_manifest.json'] = json.dumps(manifest).encode()
            if extra:
                extra(values)
            target = root / name
            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for key, value in values.items():
                    archive.writestr(key, value)
            return target

        good = fixture('good.zip')
        operator_sha = hashlib.sha256(operator).hexdigest()

        def verify(target, **overrides):
            arguments = dict(expected_sha256=_file_sha256(target), expected_bytes=target.stat().st_size,
                             expected_operator_sha256=operator_sha, extract_parent=root)
            arguments.update(overrides)
            return verify_and_extract_payload(target, **arguments)

        def reject(name, action):
            dirs_before = {p.name for p in root.glob('spades_aot_payload_*')}
            try:
                action()
            except (ValueError, SyntaxError, zipfile.BadZipFile):
                assert {p.name for p in root.glob('spades_aot_payload_*')} == dirs_before
                checks.append({'name': name, 'passed': True})
            else:
                raise AssertionError('Expected rejection: ' + name)

        first, second = verify(good), verify(good)
        assert first['root'] != second['root']
        assert all(first['paths'][name].read_bytes() == value for name, value in blobs.items())
        checks.append({'name': 'valid_payload_exact_bytes_and_brand_new_extraction', 'passed': True})
        namespace = {'__name__': '_builder_fixture', '__file__': str(first['paths']['operator.py'])}
        exec(compile(first['operator_tree'], str(first['paths']['operator.py']), 'exec'), namespace)
        assert all(callable(namespace[name]) for name in ('prepare_aot_long_run', 'run_until_barrier', 'ack_export', 'next_segment'))
        checks.append({'name': 'class_and_function_definitions_only_no_runner_calls', 'passed': True})
        reject('wrong_outer_sha_before_extraction', lambda: verify(good, expected_sha256='0' * 64))
        reject('wrong_outer_byte_count_before_extraction', lambda: verify(good, expected_bytes=1))
        reject('wrong_operator_pin', lambda: verify(good, expected_operator_sha256='1' * 64))
        bad = fixture('extra.zip', extra=lambda values: values.update({'../escape.py': b'bad'}))
        reject('unallowed_or_traversal_member', lambda: verify(bad))
        bad = fixture('source.zip', mutate=lambda values: values.update({'original.ipynb': b'{}'}))
        reject('source_sha_substitution', lambda: verify(bad))
        bad = fixture('gates.zip', mutate=lambda values: values.update({
            'coordination_TEMPLATE.json': json.dumps({key: True for key in REVIEW_GATES}).encode()}))
        reject('coordination_template_must_not_authorize_itself', lambda: verify(bad))
        bad = fixture('manifest.zip', extra=lambda values: values.update({'bundle_manifest.json': b'{"schema":1}'}))
        reject('manifest_schema_and_member_map', lambda: verify(bad))
        bad = fixture('hash.zip', extra=lambda values: values.update({'operator.py': operator + b'\n'}))
        reject('manifest_member_hash_mismatch', lambda: verify(bad))
        for label, text in (
                ('top_level_import', 'import jax\n'),
                ('top_level_call', 'print(1)\n'),
                ('decorator_call', '@print(1)\ndef extra(): pass\n'),
                ('default_call', 'def extra(value=print(1)): pass\n'),
                ('class_setup', 'class Extra(RuntimeError):\n    value=print(1)\n')):
            reject(label, lambda text=text: operator_definitions_tree(operator.decode() + text))
        reject('duplicate_json_keys', lambda: parse_json_object('{"nonce":1,"nonce":2}'))
        reject('nonfinite_json_data', lambda: parse_json_object('{"verified":NaN}'))
        reject('overflowing_json_numeric_literal', lambda: parse_json_object('{"value":1e999}'))
        output = root / 'preview.ipynb'
        receipt = build_operator_notebook(payload_path='/kaggle/working/final_payload.zip',
            payload_sha256=_file_sha256(good), payload_bytes=good.stat().st_size,
            operator_sha256=operator_sha, output_path=output)
        notebook = json.loads(output.read_text())
        assert validate_notebook_structure(notebook)
        assert len([c for c in notebook['cells'] if c['cell_type'] == 'code']) == 8
        texts = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        assert 'max_updates=1' in texts['aot-smoke'] and 'max_updates=None' in texts['aot-run']
        assert 'run_until_barrier(' not in texts['aot-ack-data']
        assert 'prepare_aot_long_run(' not in texts['aot-verify-load']
        assert 'RECOVERY = {}' not in texts['aot-status']
        assert 'next_segment(' not in texts['aot-run']
        assert 'run_until_barrier(' not in texts['aot-next-segment-data']
        assert '@param' not in output.read_text()
        assert "input('Operator prepare JSON (blank cancels): ')" in texts['aot-prepare-data']
        assert "input('Verified Library receipt JSON (blank cancels): ')" in texts['aot-ack-data']
        assert "input('Next segment confirmation JSON (blank cancels): ')" in texts['aot-next-segment-data']
        blank_namespace = {'input': lambda prompt: '', 'print': lambda *args: None,
                           'RECOVERY': {'retained': 'accepted-state'}}
        exec(compile(texts['aot-prepare-data'], 'blank_prepare_cell', 'exec'), blank_namespace)
        exec(compile(texts['aot-ack-data'], 'blank_ack_cell', 'exec'), blank_namespace)
        exec(compile(texts['aot-next-segment-data'], 'blank_next_cell', 'exec'), blank_namespace)
        assert blank_namespace['RECOVERY'] == {'retained': 'accepted-state'}
        assert 'AOT_SESSION' not in blank_namespace and 'ACK_STATUS' not in blank_namespace
        checks.append({'name': '17cell_notebook_AST_API_separation_no_optional_bypass', 'passed': True})
        assert before_jax == ('jax' in sys.modules)
        checks.append({'name': 'no_JAX_import_or_GPU_execution', 'passed': True})
    return dict(status='PASS_HOST_ONLY', checks_passed=len(checks), checks_total=len(checks),
                checks=checks, gpu_executed=False, runner_called=False, published=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-path')
    parser.add_argument('--payload-sha256')
    parser.add_argument('--payload-bytes', type=int)
    parser.add_argument('--operator-sha256')
    parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--original', default='notebooks/source_recovered.ipynb')
    parser.add_argument('--production', default='notebooks/runtime_kaggle_v1.ipynb')
    arguments = parser.parse_args()
    if arguments.self_test:
        result = self_test(arguments.original, arguments.production)
    else:
        required = ('payload_path', 'payload_sha256', 'payload_bytes', 'operator_sha256', 'output')
        if any(getattr(arguments, name) is None for name in required):
            parser.error('Build requires --payload-path/--payload-sha256/--payload-bytes/--operator-sha256/--output.')
        result = build_operator_notebook(payload_path=arguments.payload_path,
            payload_sha256=arguments.payload_sha256, payload_bytes=arguments.payload_bytes,
            operator_sha256=arguments.operator_sha256, output_path=arguments.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
