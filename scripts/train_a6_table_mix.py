"""A6-no-alpha actor/critic continuation, flat30/guided30/table40, fixed reward."""
import train_crawl_first as base
from smp.rl.tasks.getup import a6_table_mix as mix
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,random,os,sys,subprocess,hashlib
import torch,numpy as np
from mjlab.sensor.contact_sensor import ContactMatch
from train_v34_table_mix import audit_cpu,evaluate as rollout_evaluate
PARENT='/root/workplace/smp-v34-table-mix/artifacts/a6_no_alpha_9999.pt'
SHA='24f1a4513cf5cabb9bd0b9ef54e90a3d50e625a87774878379af5da5dcf8fbd1'

def config(n,evaluation,seed,escape_progress_scale=1.):
 cfg,agent=base.config(n,'CR0',evaluation,seed)
 original,_=base.base.build_config(n,'A6_no_alpha',evaluation,seed)
 cfg.rewards=original.rewards
 assert escape_progress_scale > 0
 cfg.rewards['plate_geometry_progress'].weight *= escape_progress_scale
 cfg.scene.sensors=tuple(replace(s,secondary=ContactMatch(mode='geom',pattern='escape_obstacle/escape_plate_geom')) if s.name=='free_contact' else s for s in cfg.scene.sensors)
 cfg.events['multiterrain_phase'].func=mix.update
 cfg.metrics.pop('crawl_contact',None)
 cfg.sim.contact_sensor_maxmatch=256
 agent.algorithm.learning_rate=1e-5;agent.algorithm.schedule='fixed';agent.save_interval=500;agent.run_name='A6Mix_G'+str(escape_progress_scale)+'_30_30_40_seed'+str(seed)
 return cfg,agent

class Wrapper(base.RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for s,name in [(0,'flat'),(1,'table'),(2,'guided')]:
   m=env._mix_scene==s
   for key,val in [('smp',env._fixed_smp_score),('task',env._fixed_task_score),('product',env._fixed_product),('head',base.mt.height(env)),('sr3_episode',env._tb_episode_sr3.float())]:log['Mix/'+name+'/'+key]=val[m].mean().detach()
  if env.cfg.episode_length_s>100:
   if not hasattr(env,'_mix_audit'):env._mix_audit={}
   z=base.mt.height(env)
   vals=torch.stack([env._fixed_smp_score,env._fixed_task_score,env._fixed_product],-1)
   for scene,name in [(0,'flat'),(1,'table'),(2,'guided')]:
    for phase,mask in [('low',z<.65),('upright',(z>=1.15)&(-env.scene['robot'].data.projected_gravity_b[:,2]>=.9))]:
     m=(env._mix_scene==scene)&mask;key=name+'/'+phase
     prev,cnt=env._mix_audit.get(key,(torch.zeros(3,device=env.device),0))
     env._mix_audit[key]=(prev+vals[m].sum(0),cnt+int(m.sum()))
  assert torch.isfinite(r).all()
  return obs,r,d,e

def evaluate(env,w,policy,obs,out,video=False):
 env._recording_label='A6 no alpha mix 30/30/40'
 result=rollout_evaluate(env,w,policy,obs,out,video)
 base.write(out/'reward_audit.json',{k:dict(samples=n,smp=float(v[0]/max(n,1)),task=float(v[1]/max(n,1)),product=float(v[2]/max(n,1))) for k,(v,n) in env._mix_audit.items()})
 return result

class Runner(base.MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  env=self.env.unwrapped
  super().save(path,infos={**(infos or {}),'a6_mix':dict(mean=env._smp_normalizer.mean.cpu(),count=env._smp_normalizer.count.cpu())})
  if not self.active:return
  i=self.current_learning_iteration;base.write(self.args.out/'progress.json',dict(iteration=i))
  if i==0 or (i%500 and i!=self.args.updates-1):return
  out=self.args.out/'validation'/str(i);out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',__file__,'--eval','--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs','512','--seed','20261071','--escape-progress-scale',str(self.args.escape_progress_scale)]
  if i%1000==0 or i==self.args.updates-1:cmd+=['--video']
  with open(out/'console.log','w') as f:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.args.eval_gpu,MUJOCO_EGL_DEVICE_ID=self.args.eval_gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((out/'summary.json').read_text());base.write(self.args.out/'progress.json',dict(iteration=i,validation=result))
  for k,v in result.items():self.logger.writer.add_scalar('Validation/'+k+'/SR3',v['sr3'],i)
  failed=all(result[s]['sr3']==0 for s in ['flat','table','guided'])
  self.zero_evals=getattr(self,'zero_evals',0)+1 if failed else 0
  if i>=1000 and self.zero_evals>=2:
   base.write(self.args.out/'retention_stop.json',dict(iteration=i,reason='Two successive all-scene zero SR3 evaluations'))
   raise RuntimeError('Retention stop: sustained loss of recovery')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',default=PARENT);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=6000);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--eval-gpu',default='3');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--escape-progress-scale',type=float,default=1.);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 if not a.eval:assert base.sha(a.checkpoint)==SHA
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);os.environ['V34_MIX_RESET']='bank';mix.install()
 cfg,agent=config(a.num_envs,a.eval,a.seed,a.escape_progress_scale);agent.max_iterations=a.updates;agent.logger='tensorboard' if a.eval or a.preflight else 'wandb'
 base.dump_yaml(a.out/'env.yaml',asdict(cfg));base.dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=base.ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.out),'cuda:0');runner.args=a
  if not hasattr(runner.logger,'writer'):runner.logger.writer=None
  runner.load(a.checkpoint,load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False});parent=torch.load(a.checkpoint,weights_only=False,map_location='cpu');base.verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);base.verify_nested(parent['critic_state_dict'],runner.alg.save()['critic_state_dict']);runner.alg.actor.update_normalization=lambda x:None
  ref=json.loads(Path('configs/a6_paper/smp_reward_reference_v1.json').read_text());env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'],device=env.device));env._smp_normalizer.count.copy_(torch.tensor(ref['count'],device=env.device))
  obs,_=env.reset();base.v.init(env);env._v_start_counter=-12000;env._v_ramp_updates=500
  assert obs['actor'].shape[-1]==93
  counts={name:int((env._mix_scene==s).sum()) for s,name in [(0,'flat'),(1,'table'),(2,'guided')]}
  base.write(a.out/'launch.json',dict(parent=a.checkpoint,parent_sha=base.sha(a.checkpoint),counts=counts,flat_middle=int(env._mix_middle.sum()),actor_exact=True,critic_exact=True,optimizer_fresh=True,seed=a.seed,escape_progress_scale=a.escape_progress_scale,code_sha=base.sha(Path(__file__)),lr=1e-5,updates=a.updates,prior='f2s2',ws=6.,terminations=list(cfg.terminations),events=list(cfg.events),weights={k:v.weight for k,v in cfg.rewards.items()}))
  if a.eval:evaluate(env,w,runner.get_inference_policy(),obs,a.out,a.video);return
  assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
  if a.preflight:
   audit_cpu(env,a.out);base.bd.audit_dynamics(env)
   ids=torch.tensor([0,3,7],device=env.device);m=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);m[ids]=False;before=env.sim.data.qpos[m].clone()
   with torch.inference_mode():env._reset_idx(ids)
   assert torch.equal(before,env.sim.data.qpos[m])
  runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;base.write(a.out/'completed.json',dict(iteration=runner.current_learning_iteration))
 except BaseException as e:base.write(a.out/'failed.json',dict(error=repr(e)));raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
