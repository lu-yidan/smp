"""Instrument existing recovery evaluation; report physics-substep actuator loads.
No policy or reward changes. Contact history is net force per matched geom/body,
not individual contact-point impulse or a hardware safety certification.
"""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import Simulation
from smp.rl.tasks.getup.deploy_aligned_env_cfg import CONTROL,JOINT_NAMES

p=argparse.ArgumentParser(add_help=False);p.add_argument('--loads-output',type=Path,required=True)
p.add_argument('--no-contact-history',action='store_true')
a,rest=p.parse_known_args();sys.argv=['audit_recovery_distribution.py']+rest
old_init,old_step,old_close=ManagerBasedRlEnv.__init__,ManagerBasedRlEnv.step,ManagerBasedRlEnv.close
old_sim_step=Simulation.step
active={}

def init(self,cfg,*args,**kw):
 for sensor in cfg.scene.sensors:
  if not a.no_contact_history and sensor.name in ('flat_feet','flat_other'):sensor.history_length=cfg.decimation
 old_init(self,cfg,*args,**kw)
 self._load_audit=None;active[id(self.sim)]=self

def start(e):
 r=e.scene['robot'];shape=(e.num_envs,len(r.joint_names));zero=lambda:torch.zeros(shape,device=e.device)
 e._load_audit={'n':0,'control_n':0,'limits':torch.tensor([CONTROL['tau_limit'][JOINT_NAMES.index(n)] for n in r.joint_names],device=e.device),'names':list(r.joint_names),'peak_tau':zero(),'peak_dq':zero(),'peak_power':zero(),'sum_tau2':zero(),'sum_dq2':zero(),'sat_count':zero(),'speed_exceed_count':zero(),'effort_exceed_count':zero(),'early_peak_tau':zero(),'early_peak_dq':zero(),'early_peak_power':zero(),'tau_trace':[],'dq_trace':[],'control_trace':[],'stable_breaks':torch.zeros(e.num_envs,device=e.device),'prev_stable':torch.zeros(e.num_envs,device=e.device,dtype=torch.bool),'contact_peak':{},'contact_names':{},'action_delta_peak':torch.zeros(e.num_envs,device=e.device),'last_action':None}

 e._load_audit['speeds']=torch.tensor([20. if ('knee' in n or 'hip_roll' in n) else 32. if ('hip_pitch' in n or 'hip_yaw' in n or n=='waist_yaw_joint') else 22. if ('wrist_pitch' in n or 'wrist_yaw' in n) else 37. for n in r.joint_names],device=e.device)

 # Diagnostic reference exactly matches F-series reward, but does not change evaluation policy/dynamics.
 d=e._load_audit
 ref=dict(zip(JOINT_NAMES,CONTROL['default_joint_pos']))
 for side in ('left','right'):
  ref[side+'_hip_pitch_joint']=-.15;ref[side+'_knee_joint']=.3;ref[side+'_ankle_pitch_joint']=-.15
 d['reference']=torch.tensor([ref[n] for n in r.joint_names],device=e.device)
 d['tol']=torch.tensor([.35 if 'shoulder' in n else .4 if 'elbow' in n else .2 if 'waist' in n else .25 for n in r.joint_names],device=e.device)
 d['regions']=[[i for i,n in enumerate(r.joint_names) if any(k in n for k in keys)] for keys in [('shoulder','elbow','wrist'),('waist',),('hip','knee','ankle')]]
 d['phase_head']=r.find_sites(['head'],preserve_order=True)[0][0]
 d['phase']={k:torch.zeros(e.num_envs,device=e.device) for k in ['pre_gate_speed_peak','gate_active_speed_peak','speed_peak','speed_peak_time_s','speed_peak_head_height','speed_peak_upright']}
 d['near_n']=torch.zeros(e.num_envs,device=e.device);d['near_pose']=torch.zeros(e.num_envs,device=e.device)
 d['near_tau2']=zero();d['near_effort']=zero();d['step_tau2']=zero();d['step_effort']=zero()

def sim_step(self):
 old_sim_step(self)
 e=active.get(id(self));d=None if e is None else e._load_audit
 if d is None:return
 r=e.scene['robot'];tau=r.data.qfrc_actuator;dq=r.data.joint_vel;power=(tau*dq).abs()
 height=r.data.site_pos_w[:,d['phase_head'],2]-e.scene.env_origins[:,2];upright=-r.data.projected_gravity_b[:,2];gated=(height>1.)&(upright>.8)
 speed=dq.abs().amax(-1);phase=d['phase'];higher=speed>phase['speed_peak']
 for k,val in [('speed_peak',speed),('speed_peak_time_s',torch.full_like(speed,(d['n']+1)*e.physics_dt)),('speed_peak_head_height',height),('speed_peak_upright',upright)]:phase[k]=torch.where(higher,val,phase[k])
 phase['pre_gate_speed_peak']=torch.maximum(phase['pre_gate_speed_peak'],torch.where(gated,0.,speed))
 phase['gate_active_speed_peak']=torch.maximum(phase['gate_active_speed_peak'],torch.where(gated,speed,0.))
 for key,v in [('tau',tau.abs()),('dq',dq.abs()),('power',power)]:
  torch.maximum(d['peak_'+key],v,out=d['peak_'+key])
  if d['n']<2500:torch.maximum(d['early_peak_'+key],v,out=d['early_peak_'+key])
 speeds=d['speeds']
 d['speed_exceed_count']+=(dq.abs()>.5*speeds).float();d['effort_exceed_count']+=(tau.abs()>.7*d['limits']).float()
 d['step_tau2']+=(tau/d['limits']).square();d['step_effort']+=(tau.abs()>.7*d['limits']).float()
 d['sum_tau2']+=tau.square();d['sum_dq2']+=dq.square();d['sat_count']+=(tau.abs()>=.99*d['limits']).float()
 d['tau_trace'].append(tau[:4].clone());d['dq_trace'].append(dq[:4].clone());d['n']+=1

def step(self,action):
 if self._load_audit is None:start(self)
 result=old_step(self,action);d=self._load_audit;d['control_n']+=1
 act=self.action_manager.get_term('joint_pos').raw_action
 if d['last_action'] is not None:torch.maximum(d['action_delta_peak'],(act-d['last_action']).abs().amax(-1),out=d['action_delta_peak'])
 d['last_action']=act.clone()
 for name in ['flat_feet','flat_other']:
  sensor=self.scene[name];h=sensor.data.force_history
  force=(h if h is not None else sensor.data.force[:,:,None,:]).norm(dim=-1).amax(-1)
  if name not in d['contact_peak']:d['contact_peak'][name]=force.clone();d['contact_names'][name]=sensor.primary_names
  else:torch.maximum(d['contact_peak'][name],force,out=d['contact_peak'][name])
 stable=self._flat_metrics['stable'].bool();d['stable_breaks']+=(d['prev_stable']&~stable).float();d['prev_stable']=stable.clone()
 near=(self._flat_metrics['height']>=1.15)&(-self.scene['robot'].data.projected_gravity_b[:,2]>=.93)
 errors=((self.scene['robot'].data.joint_pos-d['reference'])/d['tol']).square()
 pose=sum(w*errors[:,ids].mean(-1) for w,ids in zip((1.,1.,.5),d['regions']))
 d['near_n']+=near.float();d['near_pose']+=pose*near
 d['near_tau2']+=d['step_tau2']*near[:,None];d['near_effort']+=d['step_effort']*near[:,None]
 d['step_tau2'].zero_();d['step_effort'].zero_()
 d['control_trace'].append(torch.stack([self._flat_metrics['height'][:4],self._flat_hold[:4]*self.step_dt,stable[:4].float()],dim=-1).clone())
 return result

def close(self):
 d=getattr(self,'_load_audit',None)
 if d is not None:
  a.loads_output.parent.mkdir(parents=True,exist_ok=True)
  out={'contact_sample_dt':self.step_dt if a.no_contact_history else self.physics_dt,'physics_dt':self.physics_dt,'control_dt':self.step_dt,'physics_samples':d['n'],'control_samples':d['control_n'],'joint_names':d['names'],'tau_limits':d['limits'].cpu().tolist(),'contact_names':d['contact_names'],'sampling':'Actuator generalized joint torque and joint velocity after each 2ms step; power=tau*dq at sampled endpoint. Contact sampled at contact_sample_dt, net force by matched geometry/body, not individual contact impulse. Peak metrics over full20s and first5s; not final-frame values.'}
  for k in ['peak_tau','peak_dq','peak_power','early_peak_tau','early_peak_dq','early_peak_power','stable_breaks','action_delta_peak']:out[k]=d[k].cpu().tolist()
  for name in ['tau','dq']:out['rms_'+name]=(d['sum_'+name+'2']/d['n']).sqrt().cpu().tolist()
  out['speed_exceed_fraction']=(d['speed_exceed_count']/d['n']).cpu().tolist();out['effort_exceed_fraction']=(d['effort_exceed_count']/d['n']).cpu().tolist()
  out['saturation_fraction']=(d['sat_count']/d['n']).cpu().tolist();out['contact_peak']={k:v.cpu().tolist() for k,v in d['contact_peak'].items()}
  count=d['near_n'].clamp_min(1)
  out['near_stand_time_s']=(d['near_n']*self.step_dt).cpu().tolist()
  out['near_stand_pose_error']=(d['near_pose']/count).cpu().tolist()
  out['near_stand_tau_ratio_rms']=(d['near_tau2']/(count[:,None]*self.cfg.decimation)).sqrt().cpu().tolist()
  out['near_stand_effort_fraction']=(d['near_effort']/(count[:,None]*self.cfg.decimation)).cpu().tolist()
  out['near_stand_definition']='head>=1.15m and upright>=0.93 at end of control step; associated 10 physics samples. No near-standing samples yields 0: interpret only with near_stand_time_s and recovery success.'
  out['phase']={k:v.cpu().tolist() for k,v in d['phase'].items()}
  a.loads_output.write_text(json.dumps(out))
  np.savez_compressed(a.loads_output.with_suffix('.npz'),tau=torch.stack(d['tau_trace']).cpu().numpy(),dq=torch.stack(d['dq_trace']).cpu().numpy(),control=torch.stack(d['control_trace']).cpu().numpy())
  self._load_audit=None
 old_close(self)

ManagerBasedRlEnv.__init__=init;ManagerBasedRlEnv.step=step;ManagerBasedRlEnv.close=close;Simulation.step=sim_step
runpy.run_path('/root/workplace/smp-flat93/scripts/audit_recovery_distribution.py',run_name='__main__')
