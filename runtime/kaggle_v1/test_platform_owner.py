"""Mocked host-only platform adapters; never imports JAX or allocates compute."""
import ast,copy,hashlib,importlib.metadata,importlib.util,json,os,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('spades_kaggle_operator',ROOT/'operator.py')
op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
U0='GPU-12345678-1234-1234-1234-123456789abc'
U1='GPU-87654321-4321-4321-4321-cba987654321'
STACK={'jax':'0.11.1','jaxlib':'0.11.1','flax':'0.12.10','optax':'0.2.8','jax-cuda12-plugin':'0.11.1','jax-cuda12-pjrt':'0.11.1'}
ROW0=f'{U0},0,00000000:00:04.0,Tesla T4,15360,100,15200,580.95.05'
ROW1=f'{U1},1,00000000:00:05.0,Tesla T4,15360,100,15200,580.95.05'

class PlatformTests(unittest.TestCase):
 def mock_inventory(self,stdout=ROW0+'\n'+ROW1,versions=None,env=None):
  stack=dict(STACK if versions is None else versions)
  def version(name):
   if name not in stack:raise importlib.metadata.PackageNotFoundError(name)
   return stack[name]
  self.enterContext(patch('importlib.metadata.version',version))
  dist=SimpleNamespace(metadata={'Name':'nvidia-cublas-cu12'},version='12.9.1')
  self.enterContext(patch('importlib.metadata.distributions',lambda:[dist]))
  self.enterContext(patch.dict(os.environ,env or {},clear=True))
  query=self.enterContext(patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=stdout,stderr='')))
  return query
 def production(self,platform):
  device=SimpleNamespace(platform='gpu',device_kind='Tesla T4',memory_stats=lambda:dict(bytes_in_use=30*2**20,pool_bytes=40*2**20,bytes_limit=14*2**30))
  ns=dict(_SPADES_KAGGLE_PLATFORM=copy.deepcopy(platform),jax=SimpleNamespace(devices=lambda:[device]))
  exec(compile((ROOT/'production_cell12.py').read_text(),'mocked_cell12.py','exec'),ns)
  return ns,device
 def test_inventory_two_physical_cards_no_environment_mutation(self):
  q=self.mock_inventory(env={'CUDA_VISIBLE_DEVICES':'0,1','LD_LIBRARY_PATH':'/opt/cuda/lib'})
  before=dict(os.environ);r=op.inventory_kaggle_nojax()
  self.assertEqual(dict(os.environ),before);self.assertEqual(len(r['physical_gpu_inventory']),2)
  self.assertFalse(r['jax_imported']);self.assertFalse(r['compute_allocated']);self.assertEqual(r['cuda12_libraries'],{'nvidia-cublas-cu12':'12.9.1'})
  self.assertNotIn('jax',sys.modules)
 def test_select_second_physical_uuid_is_readonly(self):
  self.mock_inventory(env={'CUDA_VISIBLE_DEVICES':'0,1'});r=op.preflight_kaggle_nojax(U1)
  self.assertEqual(r['platform']['selected_gpu_uuid'],U1);self.assertEqual(r['platform']['selected_gpu_pci_bus_id'],'00000000:00:05.0')
  self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0,1')
 def test_ordinal_and_invalid_uuid_refused(self):
  self.mock_inventory()
  for value in ('0','GPU-1',None,U0+' '):
   with self.assertRaises(ValueError):op.preflight_kaggle_nojax(value)
 def test_package_mismatch_refused(self):
  versions=dict(STACK,jax='0.11.0');self.mock_inventory(versions=versions)
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_missing_cuda12_plugin_refused(self):
  versions=dict(STACK);versions.pop('jax-cuda12-plugin');self.mock_inventory(versions=versions)
  with self.assertRaises(importlib.metadata.PackageNotFoundError):op.preflight_kaggle_nojax(U0)
 def test_additional_cuda13_plugin_refused(self):
  versions=dict(STACK,**{'jax-cuda13-plugin':'0.11.1'});self.mock_inventory(versions=versions)
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_loaded_jax_refused_before_inventory(self):
  q=self.mock_inventory()
  with patch.dict(sys.modules,{'jax':SimpleNamespace()}):
   with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
  q.assert_not_called()
 def test_constraint_skip_even_false_refused(self):
  self.mock_inventory(env={'JAX_SKIP_CUDA_CONSTRAINTS_CHECK':'0'})
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_jax_visibility_override_refused(self):
  self.mock_inventory(env={'JAX_CUDA_VISIBLE_DEVICES':'0,1'})
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_old_driver_refused(self):
  self.mock_inventory(stdout=ROW0.replace('580.95.05','524.60.1'))
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_unknown_or_nont4_refused(self):
  self.mock_inventory(stdout=ROW0.replace('Tesla T4','Tesla T40'))
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U1)
 def test_malformed_or_duplicate_inventory_refused(self):
  for stdout in (ROW0+'\n'+ROW0,ROW0.replace('00000000:00:04.0','badpci'),ROW0.replace('15360,100,15200','15360,100,15300')):
   with self.assertRaises(ValueError):op._aot_kaggle_board_rows(stdout)
 def test_query_failure_does_not_fallback(self):
  q=self.mock_inventory();q.return_value.returncode=1
  with self.assertRaises(RuntimeError):op.preflight_kaggle_nojax(U0)
 def test_snapshot_queries_selected_uuid_not_physical_zero(self):
  q=self.mock_inventory();platform=op.preflight_kaggle_nojax(U1)['platform'];ns,device=self.production(platform)
  os.environ['CUDA_VISIBLE_DEVICES']=U1;q.return_value.stdout=ROW1
  snapshot=ns['_gpu_memory_snapshot']()
  self.assertEqual(snapshot['gpu_index'],1);self.assertEqual(snapshot['gpu_uuid'],U1)
  self.assertIn('--id='+U1,q.call_args.args[0]);self.assertEqual(snapshot['board_total_bytes'],15360*2**20)
 def test_changed_uuid_pci_name_driver_total_fail_closed(self):
  q=self.mock_inventory();platform=op.preflight_kaggle_nojax(U1)['platform'];ns,device=self.production(platform);os.environ['CUDA_VISIBLE_DEVICES']=U1
  for bad in (ROW1.replace(U1,U0),ROW1.replace(':05.0',':06.0'),ROW1.replace('Tesla T4','NVIDIA T4'),ROW1.replace('580.95.05','580.95.06'),ROW1.replace('15360','15361')):
   q.return_value.stdout=bad;snapshot=ns['_gpu_memory_snapshot']();self.assertNotIn('board_total_bytes',snapshot);self.assertIn('selected_board_telemetry_error',snapshot)
 def test_visibility_mutation_fails_before_board_query(self):
  q=self.mock_inventory();platform=op.preflight_kaggle_nojax(U1)['platform'];ns,_=self.production(platform);q.reset_mock();os.environ['CUDA_VISIBLE_DEVICES']=U0
  with self.assertRaises(RuntimeError):ns['_gpu_memory_snapshot']()
  q.assert_not_called()
 def test_multiple_jax_devices_refused(self):
  self.mock_inventory();platform=op.preflight_kaggle_nojax(U1)['platform'];ns,d=self.production(platform);os.environ['CUDA_VISIBLE_DEVICES']=U1;ns['jax'].devices=lambda:[d,d]
  with self.assertRaises(RuntimeError):ns['_gpu_memory_snapshot']()
 def test_operator_identity_checks_saved_platform_and_device(self):
  self.mock_inventory();platform=op.preflight_kaggle_nojax(U1)['platform'];ns,d=self.production(platform);os.environ['CUDA_VISIBLE_DEVICES']=U1
  s=dict(platform=platform,saved_platform=copy.deepcopy(platform),ns=ns,profile={'device':d});op._aot_kaggle_identity_guard(s)
  s['platform']['selected_gpu_name']='Tesla T40'
  with self.assertRaises(RuntimeError):op._aot_kaggle_identity_guard(s)
 def test_production_only_cell12_changed_learning_ast_preserved(self):
  old=json.loads(Path('notebooks/runtime_safety_v2.ipynb').read_text())
  new=json.loads(Path('notebooks/runtime_kaggle_v1.ipynb').read_text())
  self.assertEqual(len(new['cells']),38)
  for i in range(38):
   if i!=12:self.assertEqual(new['cells'][i],old['cells'][i])
   if new['cells'][i]['cell_type']=='code':ast.parse(''.join(new['cells'][i]['source']))
 def test_memory_budget_functions_exact_ast(self):
  frozen=json.loads(Path('notebooks/runtime_safety_v2.ipynb').read_text())
  old=ast.parse(''.join(frozen['cells'][12]['source']));new=ast.parse((ROOT/'production_cell12.py').read_text())
  a={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
  b={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
  for name in a:
   if name not in ('_gpu_memory_snapshot','_gpu_runtime_identity'):self.assertEqual(a[name],b[name],name)

if __name__=='__main__':unittest.main(verbosity=2)
