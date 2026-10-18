"""Run isolated host suites; no JAX, GPU, provider or credential operations."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',help='Optional new JSON report path.')
    arguments=parser.parse_args()
    if arguments.report and Path(arguments.report).exists():
        parser.error('Report path must be new.')
    results=[]
    for directory in ('tests','runtime/colab_c3','runtime/kaggle_v1'):
        command=[sys.executable,'-m','unittest','discover','-s',directory,'-p','test_*.py']
        process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        count=re.search(r'Ran (\d+) tests?',process.stderr)
        row=dict(suite=directory,tests=int(count[1]) if count else None,passed=process.returncode==0)
        results.append(row)
        if process.returncode:
            # Failure detail is local diagnostic output, never an automatic upload.
            print(process.stderr,file=sys.stderr)
    report=dict(status='PASS_HOST_ONLY' if all(r['passed'] for r in results) else 'FAIL',
        suites=results,tests=sum(r['tests'] or 0 for r in results),gpu_executed=False,
        checkpoint_loaded=False,external_upload_verified=False,playing_strength_evaluated=False)
    if arguments.report:
        with Path(arguments.report).open('x') as stream:
            stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS_HOST_ONLY' else 1


if __name__=='__main__':
    raise SystemExit(main())
