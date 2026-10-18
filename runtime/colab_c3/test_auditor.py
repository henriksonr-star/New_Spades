"""Focused adversarial resource and CLI checks; no JAX/GPU/native674 bytes."""
import hashlib,importlib.util,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('export_audit_owner_subject',ROOT/'audit_export_package.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class OwnerAuditChecks(unittest.TestCase):
 def fixture(self):
  arrays={f'leaf_{i:05d}':np.array([i],dtype=np.float32) for i in range(214)}
  buffer=io.BytesIO();np.savez(buffer,**arrays)
  return buffer.getvalue(),[dict(shape=[1],dtype='float32') for _ in range(214)]
 def rewrite(self,raw,name,data,duplicate=False):
  output=io.BytesIO()
  with zipfile.ZipFile(io.BytesIO(raw)) as old,zipfile.ZipFile(output,'w') as new:
   for item in old.namelist():new.writestr(item,data if item==name else old.read(item))
   if duplicate:new.writestr(name,data)
  return output.getvalue()
 def test_valid214_headers_pass_without_jax(self):
  raw,specs=self.fixture();audit.preflight_npz(raw,specs)
 def test_huge_shape_header_rejected_before_array_allocation(self):
  raw,specs=self.fixture();header=io.BytesIO();np.lib.format.write_array_header_1_0(header,dict(descr='<f4',fortran_order=False,shape=(10**12,)));header.write(b'\0'*4)
  bad=self.rewrite(raw,'leaf_00000.npy',header.getvalue())
  with patch.object(audit.np,'load',side_effect=AssertionError('allocation forbidden')):
   with self.assertRaisesRegex(ValueError,'header differs'):audit.preflight_npz(bad,specs)
 def test_object_header_rejected_before_array_allocation(self):
  raw,specs=self.fixture();header=io.BytesIO();np.lib.format.write_array_header_1_0(header,dict(descr='|O',fortran_order=False,shape=(1,)));header.write(b'\0'*8)
  with self.assertRaises(ValueError):audit.preflight_npz(self.rewrite(raw,'leaf_00000.npy',header.getvalue()),specs)
 def test_duplicate_npz_leaf_rejected(self):
  raw,specs=self.fixture()
  with self.assertRaisesRegex(ValueError,'unique'):audit.preflight_npz(self.rewrite(raw,'leaf_00000.npy',b'x',duplicate=True),specs)
 def test_duplicate_json_keys_rejected(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):audit.read_json('{"nonce":"x","nonce":"y"}')
 def test_overflow_json_float_rejected(self):
  with self.assertRaisesRegex(ValueError,'Nonfinite'):audit.read_json('{"x":1e999}')
 def test_type_stable_json_comparison(self):
  self.assertFalse(audit.same_json({'x':True},{'x':1}));self.assertFalse(audit.same_json({'x':1.0},{'x':1}))
 def test_wrong_baseline_identity_rejected(self):
  with self.assertRaisesRegex(ValueError,'trusted674'):audit.baseline_trust_check(dict(sha256='0'*64))
 def test_parsed_snapshot_bound_to_reported_identity(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'package.zip';path.write_bytes(b'original');info=audit.identity(path);path.write_bytes(b'different')
   with self.assertRaisesRegex(ValueError,'changed between'):audit.package_content(path,info)
 def test_cli_never_overwrites_report_or_input(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'keep.json';path.write_bytes(b'preserved')
   argv=['audit','--package',str(path),'--status',str(path),'--coordination',str(path),'--baseline-package',str(path),'--report',str(path)]
   with patch('sys.argv',argv),self.assertRaises(SystemExit) as caught:audit.main()
   self.assertEqual(caught.exception.code,2);self.assertEqual(path.read_bytes(),b'preserved')
if __name__=='__main__':unittest.main()
