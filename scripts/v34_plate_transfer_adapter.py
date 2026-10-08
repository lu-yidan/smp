"""Deployment adaptation of archived V34; import against archived SMP source."""
from dataclasses import dataclass,replace
import numpy as np,torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.event_manager import EventTermCfg,requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.mdp import events as old
from smp.rl.tasks.getup.master_deployment_contract import *

class DelayedAction(DeploymentPositionAction):
 def __init__(self,cfg,env):
  super().__init__(cfg,env);self.env=env;self.buf=self._processed_actions.unsqueeze(0).repeat(6,1,1);self.cursor=0;self.rows=torch.arange(self.num_envs,device=self.device)
 def reset(self,env_ids=None):
  super().reset(env_ids);ids=slice(None) if env_ids is None else env_ids
  if hasattr(self,'buf'):self.buf[:,ids]=self._entry[ids].unsqueeze(0)
 def apply_actions(self):
  self.buf[self.cursor]=self._processed_actions;lag=getattr(self.env,'_vp_lag',torch.zeros(self.num_envs,dtype=torch.long,device=self.device));target=self.buf[(self.cursor-lag)%6,self.rows];self.cursor=(self.cursor+1)%6
  self._entity.set_joint_position_target(target-self._entity.data.encoder_bias[:,self._target_ids],joint_ids=self._target_ids)
@dataclass(kw_only=True)
class DelayedActionCfg(DeploymentPositionActionCfg):
 def build(self,env):return DelayedAction(self,env)

def init(env,evaluation):
 if hasattr(env,'_vp_group'):return
 n=env.num_envs;dev=env.device;a=round(n*.45);b=n//2-a
 group=torch.cat([torch.full((a,),0),torch.full((a,),1),torch.full((b,),2),torch.full((b,),3)]).to(dev)
 env._vp_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+8191);env._vp_group=group[torch.randperm(n,device=dev,generator=env._vp_rng)]
 bank=np.load('outputs/v34_deploy_reset/'+('validation' if evaluation else 'train')+'.npz');env._vp_bank=[torch.tensor(bank['qpos'][bank['pose']==p],device=dev) for p in (0,1)]
 env._robust_reset_type=torch.zeros(n,dtype=torch.long,device=dev);env._vp_lag=torch.zeros(n,dtype=torch.long,device=dev);env._vp_mass=torch.ones(n,3,device=dev);env._vp_gain=torch.ones(n,6,device=dev)
 r=env.scene['robot'];model=env.sim.mj_model;env._vp_bodies=r.indexing.body_ids.long()
 def below(b,name):
  target=model.body('robot/'+name).id
  while b>0 and b!=target:b=int(model.body_parentid[b])
  return b==target
 env._vp_body_group=torch.tensor([0 if below(int(b),'left_hip_pitch_link') or below(int(b),'right_hip_pitch_link') else 1 if below(int(b),'torso_link') else 2 for b in env._vp_bodies],device=dev)
 env._vp_act_group=[]
 for act in r.actuators:
  name=act._target_names[0];env._vp_act_group.append(0 if 'waist' in name else 1 if 'hip' in name else 2 if 'knee' in name else 3 if 'ankle' in name else 4 if 'shoulder' in name or 'elbow' in name else 5)

@requires_model_fields('body_mass','body_inertia',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,evaluation=False):
 init(env,evaluation);ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids;r=env.scene['robot']
 if hasattr(env,'_robust_push_active'):
  env._robust_push_active[ids]=0;env._robust_push_recovery[ids]=0;env._robust_push_wait[ids]=100;env._robust_forces[ids]=0;env._robust_torques[ids]=0
 rand=lambda shape:torch.rand(shape,generator=env._vp_rng,device=env.device)
 n=len(ids);width=.1+.1*min(env.common_step_counter/(24*2000),1.);active=(rand((n,))>=.25)&(not evaluation)
 mass=1+(2*rand((n,3))-1)*width;gain=1+(2*rand((n,6))-1)*width;mass[~active]=1;gain[~active]=1
 env._vp_mass[ids]=mass;env._vp_gain[ids]=gain;env._vp_lag[ids]=torch.where(active,torch.randint(0,6,(n,),generator=env._vp_rng,device=env.device),0)
 bodies=env._vp_bodies;scale=mass[:,env._vp_body_group]
 for field in ['body_mass','body_inertia']:
  value=env.sim.get_default_field(field)[bodies][None]*(scale if field=='body_mass' else scale[:,:,None]);getattr(env.sim.model,field)[ids[:,None],bodies[None,:]]=value
 for act,g in zip(r.actuators,env._vp_act_group):act.set_gains(ids,kp=act.default_stiffness[ids]*gain[:,g,None],kd=act.default_damping[ids]*gain[:,g,None])
 for pose in (0,1):
  sel=ids[(env._vp_group[ids]%2)==pose];pool=env._vp_bank[pose];q=pool[torch.randint(len(pool),(len(sel),),generator=env._vp_rng,device=env.device)];state=r.data.default_root_state[sel].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[sel];state[:,3:7]=q[:,3:7];state[:,7:]=0
  r.write_root_state_to_sim(state,env_ids=sel);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=sel);env._robust_reset_type[sel]=2 if pose==0 else 1
 env.sim.forward();old._prime_smp_history_from_current_state(env,ids)

def plate_reset(env,env_ids=None,evaluation=False,**params):
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 # Initial arms already projected onto deployment self-collision constraints.
 params.update(support_ready_supine=False,reground_robot=False,collision_geom_pattern=COLLISION_PATTERN,inactive_xy=(20.,20.))
 if evaluation:params.update(plate_mass_range=(8.,8.),initial_max_mass=8.,longitudinal_offset_curriculum=(0.,0.),lateral_offset_curriculum=(0.,0.),overlap_curriculum_steps=1)
 for enabled in [True,False]:
  sel=ids[(env._vp_group[ids]<2)==enabled]
  if len(sel):old.reset_guided_escape_plate_curriculum(env,sel,**dict(params,obstacle_probability=float(enabled)))
 env.sim.forward();old._prime_smp_history_from_current_state(env,ids)

def contact_diagnostic(env,kind):
 n=env.num_envs;zero=torch.zeros(n,device=env.device)
 if not hasattr(env,'_escape_peak_contact_force'):return zero
 if kind=='force':return ((env._escape_peak_contact_force>1500)&(env._vp_group<2)).float()
 if kind=='wait':return ((env._escape_wait_steps>12)&(env._escape_phase==1)).float()
 return (env._escape_first_contact_head_height>.40).float()

def configure(cfg,evaluation=False):
 cfg=apply_deployment_contract(cfg);cfg.sim.contact_sensor_maxmatch=256
 cfg.actions['joint_pos']=DelayedActionCfg(**{k:v for k,v in vars(cfg.actions['joint_pos']).items()})
 cfg.scene.sensors=tuple(replace(s,primary=replace(s.primary,pattern=COLLISION_PATTERN)) if s.name=='robot_obstacle_contact' else s for s in cfg.scene.sensors)
 for t in list(cfg.events.values())+list(cfg.rewards.values()):
  if 'collision_geom_pattern' in t.params:t.params['collision_geom_pattern']=COLLISION_PATTERN
 # No GSI or replay substitution: the agreed four cohorts account for every reset.
 for name in ['gsi_reset','gsi_refresh','record_failure_states','failure_state_replay_reset']:cfg.events.pop(name,None)
 cfg.events['mixed_fall_reset']=EventTermCfg(func=reset,mode='reset',params={'evaluation':evaluation})
 oldplate=cfg.events['reset_escape_obstacle'];oldplate.func=plate_reset;oldplate.params['evaluation']=evaluation
 cfg.events['init_smp_state'].params.update(gsi_buffer_size=128,gsi_batch_size=128)
 if not evaluation:
  cfg.events['random_body_wrench']=EventTermCfg(func=mdp.random_body_wrench,mode='step',params=dict(force_range=(20.,80.),torque_range=(2.,8.),interval_steps=(150,300),duration_steps=(5,10),curriculum_steps=150000))
 else:
  for name in ['foot_friction','encoder_bias','base_com','random_body_wrench']:cfg.events.pop(name,None)
 # Keep penetration/unstable guards; historical impact and first-contact timing
 # filters are diagnostics in the deployment solver, not early episode resets.
 cfg.events['update_escape_phase'].params.update(max_contact_force=None,max_wait_steps=None,max_initial_contact_head_height=None)
 for kind in ['force','wait','height']:cfg.metrics['historical_contact_'+kind]=MetricsTermCfg(func=contact_diagnostic,params={'kind':kind})
 cfg.observations['actor'].enable_corruption=not evaluation
 return cfg
