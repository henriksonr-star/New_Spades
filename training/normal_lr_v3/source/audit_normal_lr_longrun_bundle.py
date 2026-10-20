"""Audit a portable normal-LR bundle with independently retained source support.

The source-native directory must contain the approved status, authorization and
evaluation archive prescribed by the source receipt. Hourly bundles intentionally
do not duplicate that original evaluation archive.
"""
import argparse,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','source-native','source-approval-receipt','report'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--expected-journal-head-sha256');args=parser.parse_args();target=Path(args.report)
    if target.exists() or target.is_symlink():parser.error('--report must be a new file; existing evidence cannot be overwritten.')
    path=ROOT/'normal_lr_longrun_contract.py';module=types.ModuleType('normal_lr_portable_audit');module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    try:
        report=module.audit_normal_lr_longrun_bundle(args.bundle,source_native_path=args.source_native,source_approval_receipt_path=args.source_approval_receipt,expected_journal_head_sha256=args.expected_journal_head_sha256);code=0
    except Exception as exc:
        report=dict(status='FAIL',audit_scope='CONTENT_ONLY',verified_content=False,launch_authorized=False,external_persistence_verified=False,error_type=type(exc).__name__,error=str(exc));code=1
    with target.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
