"""A6 reward transfer onto the audited V34 plate environment, with matched SMP."""
import json,pickle
from pathlib import Path
import torch
from dataclasses import replace
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor.contact_sensor import ContactSensorCfg,ContactMatch
from smp.rl.tasks.getup import multiterrain as mt,v33_path_ablation as pa,r2_ablation as ra,v33_reward_transfer as v
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

# This scene has a single flat ground, with no supplementary terrain bodies.
def ground_force(env,name):return env.scene[name].data.force
mt.ground_force=ground_force;v.ground_force=ground_force

def reset_bookkeeping(env,env_ids=None):
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 n=env.num_envs;d=env.device;r=env.scene['robot']
 if not hasattr(env,'_mt_bank'):
  env._mt_bank=True;env._bd_flat_support=True
  env._mt_scene=(env._vp_group<2).long();env._mt_tick=-1
  idx,_=r.find_geoms(COLLISION_PATTERN);env._mt_geoms=r.indexing.geom_ids[torch.tensor(idx,device=d)].long()
  e=env.scene['escape_obstacle'];gi,_=e.find_geoms(['escape_plate_geom']);gid=e.indexing.geom_ids[gi[0]].long();env._mt_plate_ids=[gid,gid]
  for name in ('ever_contact','escaped','invalid'):setattr(env,'_mt_'+name,torch.zeros(n,dtype=torch.bool,device=d))
  env._mt_clear_hold=torch.zeros(n,dtype=torch.long,device=d)
  for name in ('best','progress','clearance','force','initial_overlap','best_distance','separation_progress'):setattr(env,'_mt_'+name,torch.zeros(n,device=d))
  ra.initialize(env);env._pa_arm='A6';env._pa_tick=-1
  for name in ('best_score','best_clearance','initial_count','progress','clearance_score','gate','load_acc','load','support'):setattr(env,'_pa_'+name,torch.zeros(n,device=d))
  env._pa_joint_acc=torch.zeros_like(r.data.joint_pos);env._ra_initial=torch.zeros(n,device=d)
 env._ft_relaxed=True
 for name in ('ever_contact','escaped','invalid','clear_hold','progress','force','best_distance','separation_progress'):getattr(env,'_mt_'+name)[ids]=0
 score,_=mt.plate_geometry(env);env._mt_best[ids]=score[ids];env._mt_initial_overlap[ids]=(-score[ids]).clamp_min(.01)
 for name in ('timer','load_acc','hold','gate','values','prev_target','credit','taupeak','speedpeak','powerpeak','hightime','longest'):getattr(env,'_ra_'+name)[ids]=0
 env._ra_window[:,ids]=0;env._ra_best[ids]=0
 motion=torch.cat([r.data.joint_pos,mt.height(env)[:,None],-r.data.projected_gravity_b[:,2,None]],-1)
 env._ra_motion[:,ids]=motion[ids][None];env._ra_initial[ids]=(-ra.planar_score(env)[ids]).clamp_min(.01)
 count,score,clearance=pa.geometry(env)
 env._pa_best_score[ids]=score[ids];env._pa_best_clearance[ids]=clearance[ids];env._pa_initial_count[ids]=count[ids].float()
 for name in ('progress','clearance_score','gate','load_acc','load','support'):getattr(env,'_pa_'+name)[ids]=0
 env._pa_joint_acc[ids]=0;v.reset(env,ids);env._v_start_counter=-12000;env._v_ramp_updates=500

def configure(cfg,arm):
 assert arm in ('R0','R1','R2')
 cfg.events['init_smp_state'].params['ckpt_path']='datasets/pretrain_ckpt/pretrained_getup_f2s2.pt'
 cfg.rewards['task_smp_product'].params['ws']=6.
 if arm=='R0':
  cfg.rewards['task_smp_product'].params['constrained_scale']=1.
  return cfg
 with open('a6_reward_bundle.pkl','rb') as f:b=pickle.load(f)
 cfg.rewards=b['rewards'];cfg.rewards['task_smp_product'].params['ws']=6.
 # Inactive zero-weight terms need no unrelated metrics or reset buffers.
 cfg.rewards={k:t for k,t in cfg.rewards.items() if t.weight!=0}
 cfg.metrics.update(b['metrics'])
 sensors=list(b['sensors'])
 original=next(s for s in cfg.scene.sensors if s.name=='robot_obstacle_contact')
 for name in ('guided_contact','free_contact'):sensors.append(replace(original,name=name,fields=('found','force','dist'),reduce='mindist',num_slots=1))
 cfg.scene.sensors=tuple(cfg.scene.sensors)+tuple(sensors)
 cfg.events['a6_bookkeeping']=EventTermCfg(func=reset_bookkeeping,mode='reset')
 cfg.events['a6_geometry']=EventTermCfg(func=mt.update,mode='step')
 if arm=='R2':
  cfg.rewards['path_quiet_feet'].weight=0.;cfg.rewards['path_joint_stall'].weight=0.
 return cfg

def load_reference(env):
 ref=json.loads(Path('smp_reward_reference_v1.json').read_text())
 env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'],device=env.device))
 env._smp_normalizer.count.copy_(torch.tensor(ref['count'],device=env.device))
