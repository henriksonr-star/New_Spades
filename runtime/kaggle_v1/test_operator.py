"""Focused operator orchestration fixtures. stdlib+NumPy; no real JAX/GPU.
The independent peer file supplies copied pure production journal helpers only.
"""
import ast,copy,hashlib,importlib.util,io,json,tempfile,unittest,zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ref=load('peer_pure_health_reference',ROOT.parents[1]/'tests'/'support'/'scientific_fixture.py')
op=load('owner_aot_operator',ROOT/'operator.py')

class OwnerChecks(unittest.TestCase):
 def build(self,mode='healthy',accepted=24):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
  clock=[0.];counts=dict(collect=0,optimize=0,save=0);cfg=copy.deepcopy(ref.CFG);recovery={};order=[]
  class FixedDatetime(datetime):
   @classmethod
   def now(cls,tz=None):return datetime(2026,10,18,15,0,tzinfo=timezone.utc)+timedelta(seconds=clock[0])
  self.enterContext(patch('datetime.datetime',FixedDatetime));self.enterContext(patch('time.monotonic',lambda:clock[0]))
  self.enterContext(patch.object(op,'_aot_kaggle_identity_guard',lambda s:None))
  self.enterContext(patch.object(op,'_aot_host_available_bytes',lambda:2**45))
  self.enterContext(patch.object(op,'_aot_resident_credit',lambda s,a,snap:dict(credit_bytes=0,scope='synthetic orchestration fixture; credit proof independently tested')))
  def collect(state):
   counts['collect']+=1;order.append('collect');clock[0]+=1
   if mode=='collect_error':raise RuntimeError('synthetic collect failure')
   result=copy.deepcopy(state);batch=tuple(np.ones((2,),np.float32) for _ in range(11));metrics={f'c{i}':np.float32(0) for i in range(47)};metrics['illegal_actions']=np.float32(1 if mode=='illegal' else 0)
   if mode=='nan_batch':batch=list(batch);batch[0][0]=np.nan;batch=tuple(batch)
   return result,batch,metrics
  def optimize(state,batch):
   counts['optimize']+=1;order.append('optimize');clock[0]+=1
   if mode=='opt_error':raise RuntimeError('synthetic optimizer failure')
   candidate=state._replace(updates=np.int32(int(state.updates)+1))
   metrics={k:np.float32(1) for k in ('finite','diagnostics_finite','gradients_finite','grad_norm_finite','params_finite','optimizer_finite')}
   metrics.update(attempted_updates=np.int32(accepted),accepted_updates=np.int32(accepted),rejected_updates=np.int32(0),kl_early_stop=np.bool_(accepted<24));metrics.update({f'm{i}':np.float32(0) for i in range(14)})
   if mode=='bad_work':metrics['accepted_updates']=np.float32(1.5)
   return candidate,metrics
  def save(state,cfg,history,directory,**kwargs):
   counts['save']+=1
   if mode=='save_error' and counts['save']>1:raise OSError('synthetic save failure')
   arrays={f'leaf_{i:05d}':np.asarray(v) for i,v in enumerate(ref.TreeUtil.tree_leaves(state))}
   payload=io.BytesIO();np.savez(payload,**arrays);raw=payload.getvalue()
   manifest=dict(cfg=cfg,updates=int(state.updates),env_steps=state.env_steps,sha256=hashlib.sha256(raw).hexdigest(),leaves=[dict(shape=list(a.shape),dtype=str(a.dtype)) for a in arrays.values()],training_revision='strong-teacher-hand-gae-v3',training_state_schema=2,versions=dict(jax='0.11.1',flax='0.12.10',optax='0.2.8'))
   path=Path(directory)/('native_'+str(counts['save'])+'.zip')
   with zipfile.ZipFile(path,'w') as z:z.writestr('manifest.json',json.dumps(manifest));z.writestr('state.npz',raw);z.writestr('history.json',json.dumps(history))
   return path
  def snapshot():return dict(platform='gpu',jax_live_bytes=100000,jax_memory_stats_raw=dict(pool_bytes=200000))
  def admit(est,opts,stage,**kw):
   return dict(status='admitted',overlap_credit_bytes=0,snapshot=kw.get('snapshot',snapshot()),budget_details=dict(pool_counter_source='pool_bytes',pool_counter_status='validated_current_total'),estimated_peak_bytes=est)
  ns=dict(_SPADES_KAGGLE_PLATFORM=dict(provider='kaggle',adapter='spades-kaggle-aot-v1',selected_gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',runtime_stack=dict(jax='0.11.1')),np=np,jax=SimpleNamespace(tree_util=ref.TreeUtil,device_get=lambda tree:tree),_gpu_runtime_identity=lambda:'runtime',_gpu_memory_admission=admit,_gpu_memory_snapshot=snapshot,_scalar_metrics=lambda m:{k:float(v) for k,v in m.items()},save_checkpoint=save)
  for name in ('_optimizer_resume_journal','_optimizer_totals','_optimizer_health','_optimizer_work_record','_optimizer_config_signature'):ns[name]=getattr(ref,name)
  options=dict(target_gib=14.,reserve_mib=1024,safety_factor=1.45,extra_guard_mib=256)
  profile=dict(options=options,execution_handles=dict(collect=collect,optimize=optimize),state_bytes=10000,batch_bytes=1000,estimated_peak_bytes=20000,collect=dict(total_bytes=10000),optimize=dict(total_bytes=20000))
  coordination=dict(authorization_valid_until_utc='2026-10-19T00:00:00Z',export_wait_timeout_seconds=3600,source_pins=dict(test='host_fixture'),max_segments=2,supplemental_export_interval_seconds=900)
  history=[dict(kind='fixed_budget_ppo',update=i,optimizer_work=dict(accepted=24)) for i in range(1,38)]
  state=ref.state(674)
  s=op._aot_new_session(ns=ns,state=state,cfg=cfg,history=history,profile=profile,coordination=coordination,output_dir=root/'run',recovery_sink=recovery)
  op._aot_checkpoint_barrier(s,'initial')
  return s,clock,counts,recovery,order
 def ack(self,s):
  receipt={k:s['pending_export'][k] for k in ('nonce','package_path','package_sha256','package_bytes','native_path','native_sha256','native_bytes')};receipt.update(library_receipt='TEST_EXTERNAL_RECEIPT',verified=True,valid_until_utc='2026-10-18T23:00:00Z');op.ack_export(s,receipt);return receipt
 def test_initial_barrier_returns_idle_until_exact_ack(self):
  s,t,c,r,o=self.build();self.assertEqual(c['collect'],0);self.assertEqual(op.run_until_barrier(s)['status'],'export_required');self.assertEqual(c['collect'],0);self.ack(s);result=op.run_until_barrier(s,max_updates=1);self.assertEqual(result['reason'],'validation');self.assertEqual(o,['collect','optimize']);self.assertEqual(s['accepted_update'],675)
 def test_wrong_nonce_hash_bytes_and_verified_flag_reject_without_work(self):
  s,t,c,r,o=self.build();base={k:s['pending_export'][k] for k in ('nonce','package_path','package_sha256','package_bytes','native_path','native_sha256','native_bytes')};base.update(library_receipt='TEST_EXTERNAL_RECEIPT',verified=True,valid_until_utc='2026-10-18T15:30:00Z')
  for field,value in (('nonce','stale'),('package_sha256','0'*64),('native_bytes',1),('verified',False)):
   bad=dict(base);bad[field]=value
   with self.assertRaises(ValueError):op.ack_export(s,bad)
  self.assertEqual(c['collect'],0)
 def test_mutated_local_export_after_ack_prevents_all_work(self):
  s,t,c,r,o=self.build();self.ack(s);Path(s['resume_grant']['package_path']).write_bytes(b'changed');status=op.run_until_barrier(s,max_updates=1);self.assertTrue(status['terminal']);self.assertEqual(c['collect'],0)
 def test_full_collection_health_prevents_optimizer(self):
  for mode in ('illegal','nan_batch'):
   s,t,c,r,o=self.build(mode);self.ack(s);status=op.run_until_barrier(s,max_updates=1);self.assertEqual(c['optimize'],0);self.assertTrue(status['terminal']);self.assertEqual(s['accepted_update'],674)
 def test_rejected_raw_counts_keep_previous_accepted_state(self):
  s,t,c,r,o=self.build('bad_work');self.ack(s);status=op.run_until_barrier(s,max_updates=1);self.assertEqual(s['accepted_update'],674);self.assertTrue(status['terminal'])
 def test_three_new_kl_rows_stop_after_valid_commit_and_export(self):
  s,t,c,r,o=self.build(accepted=4);self.ack(s);status=op.run_until_barrier(s,max_updates=5);self.assertEqual(s['accepted_update'],677);self.assertEqual(status['reason'],'optimizer_health_stop');self.assertEqual(len(s['journal_rows']),3);self.assertEqual(len([row for row in s['history'] if row['kind']=='fixed_budget_ppo']),37);self.assertEqual(status['observed_optimizer_totals']['accepted'],12)
 def test_exact_quarter_does_not_health_stop(self):
  s,t,c,r,o=self.build(accepted=6);self.ack(s);status=op.run_until_barrier(s,max_updates=3);self.assertFalse(status['terminal']);self.assertEqual(status['reason'],'validation')
 def test_p90_wait_counts_toward_same_three_hour_segment(self):
  s,t,c,r,o=self.build();self.ack(s);op.run_until_barrier(s,max_updates=1);start=s['segment_started'];self.ack(s);t[0]=5400.;status=op.run_until_barrier(s);self.assertEqual(status['reason'],'periodic90');self.assertEqual(s['segment_started'],start);self.ack(s);t[0]=10800.;status=op.run_until_barrier(s);self.assertTrue(status['terminal']);self.assertEqual(status['reason'],'segment_complete');self.assertEqual(c['collect'],1)
 def test_barrier_native_inner_readback_detects_corrupt_payload(self):
  s,t,c,r,o=self.build();native=Path(s['pending_export']['native_path']);native.write_bytes(b'corrupt')
  with self.assertRaises(zipfile.BadZipFile):op._aot_audit_native(s,native)
 def test_save_failure_retains_latest_committed_candidate_in_recovery(self):
  s,t,c,r,o=self.build('save_error');self.ack(s);status=op.run_until_barrier(s,max_updates=1);self.assertEqual(s['accepted_update'],675);self.assertEqual(int(r['state'].updates),675);self.assertEqual(status['status'],'save_or_package_failed');self.assertTrue(status['terminal'])
 def test_clean_deadline_after_collect_save_failure_preserves_state_without_keyerror(self):
  s,t,c,r,o=self.build();self.ack(s);op.run_until_barrier(s,max_updates=1);self.ack(s)
  def failed_save(*a,**kw):raise OSError('synthetic final save failure')
  s['ns']['save_checkpoint']=failed_save;t[0]=10799.
  status=op.run_until_barrier(s)
  self.assertEqual(status['status'],'save_or_package_failed');self.assertTrue(status['terminal'])
  self.assertEqual(s['accepted_update'],675);self.assertEqual(int(r['state'].updates),675)
  self.assertEqual(c['optimize'],1);self.assertIn('checkpoint_or_package_error',status['error'])
 def test_typed_cfg_and_coord_alias_mutations_cannot_launch(self):
  changes=(('cfg','n_env',3584.0),('cfg','hand_score_reward_coef',False),('coordination','max_segments',2.0),('coordination','supplemental_export_interval_seconds',900.0))
  for area,key,value in changes:
   s,t,c,r,o=self.build();self.ack(s);s[area][key]=value
   status=op.run_until_barrier(s,max_updates=1)
   self.assertTrue(status['terminal']);self.assertEqual(status['status'],'resume_verification_failed');self.assertEqual(c['collect'],0);self.assertEqual(c['optimize'],0)
 def future_coord(self):
  result=json.loads((ROOT/'coordination_TEMPLATE.json').read_text())
  result.update(run_authorized=True,parent_review_complete=True,runtime_exclusive=True,candidate_native_export_verified=True,authorization_valid_until_utc='2026-10-21T00:00:00Z')
  return result
 def test_deferred_preparation_after_code_freeze_reaches_host_input_without_jax(self):
  class FixedDatetime(datetime):
   @classmethod
   def now(cls,tz=None):return datetime(2026,10,18,16,0,tzinfo=timezone.utc)
  versions=dict(jax='0.11.1',jaxlib='0.11.1',flax='0.12.10',optax='0.2.8')
  with patch('datetime.datetime',FixedDatetime),patch('importlib.metadata.version',lambda n:versions[n]),patch.dict('os.environ',{},clear=True),patch.object(op,'preflight_kaggle_nojax',lambda uuid:dict(platform={})):
   with self.assertRaises(FileNotFoundError):op.prepare_aot_long_run(original_path='/missing_original',production_path='/missing_production',candidate_package_path='/missing_candidate',coordination=self.future_coord(),output_dir='/missing_out',recovery_sink={})
 def test_declared_future_prepare_beyond_cap_rejects_without_files_or_imports(self):
  co=self.future_coord();co['prepare_valid_until_utc']='2026-10-20T00:00:00Z'
  with self.assertRaisesRegex(ValueError,'Fixed launch window'):
   op.prepare_aot_long_run(original_path='/missing_original',production_path='/missing_production',candidate_package_path='/missing_candidate',coordination=co,output_dir='/missing_out',recovery_sink={})
 def test_future_prepare_cap_itself_is_expired_without_jax(self):
  class FixedDatetime(datetime):
   @classmethod
   def now(cls,tz=None):return datetime(2026,10,19,15,49,14,tzinfo=timezone.utc)
  versions=dict(jax='0.11.1',jaxlib='0.11.1',flax='0.12.10',optax='0.2.8')
  with patch('datetime.datetime',FixedDatetime),patch('importlib.metadata.version',lambda n:versions[n]),patch.dict('os.environ',{},clear=True),patch.object(op,'preflight_kaggle_nojax',lambda uuid:dict(platform={})):
   with self.assertRaisesRegex(RuntimeError,'Preparation expiry'):
    op.prepare_aot_long_run(original_path='/missing_original',production_path='/missing_production',candidate_package_path='/missing_candidate',coordination=self.future_coord(),output_dir='/missing_out',recovery_sink={})
 def test_no_setup_on_import_exact_two_AOT_targets(self):
  tree=ast.parse((ROOT/'operator.py').read_text());self.assertTrue(all(isinstance(n,(ast.FunctionDef,ast.ClassDef)) or isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) for n in tree.body));prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_aot_long_run');self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='compile' for n in ast.walk(prepare)),2)
 def test_authorization_must_cover_preparation_before_any_import_or_files(self):
  class FixedDatetime(datetime):
   @classmethod
   def now(cls,tz=None):return datetime(2026,10,18,15,0,tzinfo=timezone.utc)
  self.enterContext(patch('datetime.datetime',FixedDatetime))
  coordination=json.loads((ROOT/'coordination_TEMPLATE.json').read_text())
  coordination.update(run_authorized=True,parent_review_complete=True,runtime_exclusive=True,candidate_native_export_verified=True,authorization_valid_until_utc='2026-10-18T15:30:00Z')
  with self.assertRaisesRegex(ValueError,'entire preparation window'):
   op.prepare_aot_long_run(original_path='/nonexistent',production_path='/nonexistent',candidate_package_path='/nonexistent',coordination=coordination,output_dir='/nonexistent',recovery_sink={})
if __name__=='__main__':unittest.main()
