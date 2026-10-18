"""Build a sanitized distribution payload and notebook; no GPU or private state."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def build(provider, output_dir):
    runtime = ROOT / 'runtime' / ('kaggle_v1' if provider == 'kaggle' else 'colab_c3')
    spec = importlib.util.spec_from_file_location('distribution_builder', runtime/'operator_notebook_builder.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    production = ROOT/'notebooks'/('runtime_kaggle_v1.ipynb' if provider=='kaggle' else 'runtime_safety_v2.ipynb')
    members = {'original.ipynb': (ROOT/'notebooks/source_recovered.ipynb').read_bytes(),
        'production.ipynb': production.read_bytes(), 'operator.py': (runtime/'operator.py').read_bytes(),
        'coordination_TEMPLATE.json': (runtime/'coordination_TEMPLATE.json').read_bytes()}
    manifest = dict(schema=builder.PAYLOAD_SCHEMA, members={k:identity(v) for k,v in members.items()})
    members['bundle_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    payload = output_dir/'payload.zip'
    with zipfile.ZipFile(payload,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            info=zipfile.ZipInfo(name, date_time=(2026,10,18,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100600 << 16
            archive.writestr(info,data)
    payload_info=identity(payload.read_bytes());operator_info=identity(members['operator.py'])
    with tempfile.TemporaryDirectory() as temporary:
        builder.verify_and_extract_payload(payload, expected_sha256=payload_info['sha256'],
            expected_bytes=payload_info['bytes'],expected_operator_sha256=operator_info['sha256'],extract_parent=temporary)
    target_path='/kaggle/working/payload.zip' if provider=='kaggle' else '/content/payload.zip'
    notebook=output_dir/'operator.ipynb'
    builder.build_operator_notebook(payload_path=target_path,payload_sha256=payload_info['sha256'],
        payload_bytes=payload_info['bytes'],operator_sha256=operator_info['sha256'],output_path=notebook)
    receipt=dict(distribution='sanitized source; operational release hashes differ',provider=provider,
        payload=payload_info,operator=operator_info,notebook=identity(notebook.read_bytes()),gpu_started=False,
        candidate_checkpoint_included=False,external_upload_verified=False)
    (output_dir/'build_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=('kaggle','colab'),required=True)
    parser.add_argument('--output-dir',required=True)
    arguments=parser.parse_args()
    print(json.dumps(build(arguments.provider, arguments.output_dir),indent=2))


if __name__=='__main__':
    main()
