"""Mixed-scene recovery: support-relative task, per-object obstruction, paired closure.
Geometry/contact truth is used by reward and evaluation only, never the actor.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.recovery.mixed_geometry import SCENES,SHARES,GEOMS,BODIES,OBJECT_GEOMS
from smp.rl.tasks.getup import v33_reward_transfer as v,scratch_tradeoffs as quality
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.r2_ablation import stalled_update

FIELDS=('geom_size','geom_pos','geom_quat','geom_aabb','geom_rbound','body_mass','body_inertia','body_ipos','body_iquat','body_pos')
ARMS=('M0_R2_mix','M1_R2_mix','M2_A6_mix')

def quotas(n,evaluation=False):
 keys=[];weights=[]
 for scene,share in enumerate(SHARES):
  mix=(1.,0,0) if evaluation else (.4,.4,.2) if scene==0 else (.7,.2,.1) if scene in (5,6) else (1.,0,0)
  for kind,frac in enumerate(mix):
   if frac==0:continue
   for dr in range(4) if kind==0 else (-1,):
    for src in range(2) if kind==0 else (0,):
     keys.append((scene,kind,dr,src));weights.append(share*frac*((.75,.25)[src]/4 if kind==0 else 1))
 raw=np.array(weights)*n;counts=np.floor(raw).astype(int)
 for i in np.argsort(-(raw-counts))[:n-counts.sum()]:counts[i]+=1
 return keys,counts

def initialize(env,bank_path,evaluation,arm):
 if hasattr(env,'_mix'):return
 dev=env.device;n=env.num_envs;m=env.sim.mj_model;b=np.load(bank_path)
 keys,counts=quotas(n,evaluation);cohort=torch.repeat_interleave(torch.arange(len(keys),device=dev),torch.tensor(counts,device=dev))
 table=torch.tensor(keys,device=dev);tags=table[cohort]
 pools=[]
 for level in range(3):
  pools.append([])
  for scene,kind,dr,src in keys:
   ix=np.flatnonzero((b['scene']==scene)&(b['kind']==kind)&(b['direction']==dr)&(b['source']==src)&(b['level']==level));assert len(ix),(scene,kind,dr,src,level)
   pools[-1].append(ix)
 lengths=torch.tensor([[len(x) for x in row] for row in pools],device=dev)
 padded=np.zeros((3,len(keys),int(lengths.max())),int)
 for lev,row in enumerate(pools):
  for c,ix in enumerate(row):padded[lev,c,:len(ix)]=ix
 s=SimpleNamespace(arm=arm,evaluation=evaluation,tags=tags,cohort=cohort,counts=counts.tolist(),keys=keys,
  pool=torch.tensor(padded,device=dev),lengths=lengths,rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+904193),
  bank={k:torch.tensor(b[k],device=dev,dtype=torch.bool if k=='active' else torch.float32) for k in ('qpos','size','pos','quat','poses','mass','inertia','ipos','iquat','active')},
  gids=torch.tensor([m.geom(x).id for x in GEOMS],device=dev),bids=torch.tensor([m.body(x).id for x in BODIES],device=dev),
  tick=-1,subtick=0,cursor=0)
 s.qadr=[int(m.jnt_qposadr[m.body_jntadr[int(bid)]]) for bid in s.bids];s.dadr=[int(m.jnt_dofadr[m.body_jntadr[int(bid)]]) for bid in s.bids]
 for name in ('active','blocked','clear_age','ever'):
  setattr(s,name,torch.zeros(n,4,device=dev,dtype=torch.bool if name in ('active','blocked','ever') else torch.long))
 for name in ('best_depth','best_clear','initial_count','best_distance','count','depth','clearance','force'):
  setattr(s,name,torch.zeros(n,4,device=dev))
 s.escaped=torch.zeros(n,dtype=torch.bool,device=dev);s.invalid=s.escaped.clone();s.values=torch.zeros(n,7,device=dev)
 s.timer=torch.zeros(n,29,device=dev);s.stall_acc=torch.zeros_like(s.timer);s.motion=torch.zeros(25,n,29,device=dev)
 s.peaks=torch.zeros(n,29,3,device=dev);s.last_episode_peaks=torch.zeros_like(s.peaks);s.longest=torch.zeros_like(s.timer)
 s.sample_indices=torch.zeros(n,dtype=torch.long,device=dev)
 s.robot_gids=env.scene['robot'].indexing.geom_ids[torch.tensor(env.scene['robot'].find_geoms(__import__('smp.rl.tasks.getup.master_deployment_contract',fromlist=['COLLISION_PATTERN']).COLLISION_PATTERN)[0],device=dev)].long()
 # Robot-only dynamic randomization groups, retaining historical ranges/control chain.
 r=env.scene['robot'];s.robot_bids=r.indexing.body_ids.long()
 def subtree(name):
  root=m.body('robot/'+name).id;ids=[]
  for i,bid in enumerate(s.robot_bids.tolist()):
   j=bid
   while j>0 and j!=root:j=int(m.body_parentid[j])
   if j==root:ids.append(i)
  return set(ids)
 legs=subtree('left_hip_pitch_link')|subtree('right_hip_pitch_link');upper=subtree('torso_link')
 env._bd_groups=torch.tensor([0 if i in legs else 1 if i in upper else 2 for i in range(len(s.robot_bids))],device=dev)
 env._bd_bodies=s.robot_bids;env._bd_act_groups=[]
 for act in r.actuators:
  name=act._target_names[0];env._bd_act_groups.append(0 if 'waist' in name else 1 if 'hip' in name else 2 if 'knee' in name else 3 if 'ankle' in name else 4 if ('shoulder' in name or 'elbow' in name) else 5)
 env._bd_mass=torch.ones(n,4,device=dev);env._bd_gain=torch.ones(n,6,device=dev);env._bd_lag=torch.zeros(n,dtype=torch.long,device=dev);env._bd_nominal=torch.ones(n,dtype=torch.bool,device=dev)
 env._mix=s;env._mt_bank=True;env._bd_flat_support=False;env._ft_relaxed=True
 env._fixed_group=torch.where(tags[:,1]==1,1,torch.where(tags[:,1]==2,0,tags[:,2]+2));env._fixed_source=tags[:,3]
 v.init(env);env._v_start_counter=-12000;env._v_ramp_updates=500


def force(env,name):
 result=env.scene[name].data.force
 for suffix in ('support','guide','free0','free1','fixed'):
  result=result+env.scene[name+'_'+suffix].data.force
 return result


def ray_top(points,centers,rotation,half):
 """Highest vertical-ray OBB intersection not above a point (including tiny tolerance).
 Shapes: points N,P,3; centers/half N,G,3; rotation N,G,3,3.
 """
 origin=points[:,:,None,:]-centers[:,None,:,:]
 local=torch.einsum('ngji,npgj->npgi',rotation,origin)
 direction=rotation[:,:,2,:][:,None] # inverse R times world +Z
 safe=torch.where(direction.abs()>1e-7,direction,torch.ones_like(direction))
 t0=(-half[:,None]-local)/safe;t1=(half[:,None]-local)/safe
 lo=torch.minimum(t0,t1);hi=torch.maximum(t0,t1)
 parallel=direction.abs()<=1e-7
 outside=(parallel&(local.abs()>half[:,None]+1e-6)).any(-1)
 lo=torch.where(parallel,torch.full_like(lo,-torch.inf),lo).amax(-1)
 hi=torch.where(parallel,torch.full_like(hi,torch.inf),hi).amin(-1)
 valid=(lo<=hi)&~outside&(hi<=.03)&torch.isfinite(hi)
 return torch.where(valid,points[:,:,None,2]+hi,torch.zeros_like(hi)).amax(-1).clamp_min(0)


def support_reference(env):
 s=env._mix;r=env.scene['robot'];feet=r.data.body_link_pos_w[:,env._r_feet];root=r.data.root_link_pos_w
 # Foot link is above its sole; allow an OBB top below the ankle marker.
 points=torch.cat((feet,root[:,None]),1);ids=s.gids
 tops=ray_top(points,env.sim.data.geom_xpos[:,ids],env.sim.data.geom_xmat[:,ids],env.sim.model.geom_size[:,ids])
 footload=force(env,'quality_feet')[...,2].abs()
 both=(footload>20).all(-1)
 return torch.where(both,tops[:,:2].mean(-1),tops[:,2])


def height(env):
 r=env.scene['robot'];return r.data.site_pos_w[:,env._r_head,2]-support_reference(env)


def bounds(env,ids):
 d=env.sim.data;m=env.sim.model
 mat=d.geom_xmat[:,ids];local=m.geom_aabb[:,ids,0];ext=m.geom_aabb[:,ids,1]
 return d.geom_xpos[:,ids]+torch.einsum('ngij,ngj->ngi',mat,local),torch.einsum('ngij,ngj->ngi',mat.abs(),ext)


def geometry(env):
 s=env._mix;rp,re=bounds(env,s.robot_gids);op,oe=bounds(env,s.gids)
 ref=support_reference(env);counts=[];depths=[];clear=[];blocked=[];dist=[];forces=[]
 for j,indices in enumerate(OBJECT_GEOMS):
  ids=list(indices);p=op[:,ids];ext=oe[:,ids]
  overlap=re[:,:,None,:2]+ext[:,None,:,:2]-(rp[:,:,None,:2]-p[:,None,:,:2]).abs()
  planar=(overlap>0).all(-1)
  # Body primitives resting on/above an object are support, not overhead coverage.
  below=(rp[:,:,None,2]-re[:,:,None,2])<(p[:,None,:,2]+ext[:,None,:,2]-.025)
  contact=env.scene[('mix_contact_guide','mix_contact_free0','mix_contact_free1','mix_contact_fixed')[j]].data
  touch=(contact.found>0).any(-1);f=contact.force.norm(dim=-1).amax(-1);forces.append(f)
  envelope=torch.maximum(ref+1.25,(rp[:,:,2]+re[:,:,2]).amax(-1)+.05)
  reaches=(p[:,:,2]-ext[:,:,2]<envelope[:,None])|touch[:,None]
  covered=planar&below&reaches[:,None,:]
  covered&=s.active[:,j,None,None]
  counts.append(covered.any(-1).sum(-1))
  depths.append(torch.where(covered,overlap.clamp_min(0).amin(-1),0.).amax(-1).sum(-1))
  sep=(-overlap).clamp_min(0).norm(dim=-1)
  sep=torch.where(below&reaches[:,None,:],sep,torch.full_like(sep,.04))
  clear.append(sep.amin(-1).amin(-1).clamp_max(.04));blocked.append(covered.any(-1).any(-1))
  dist.append((rroot(env)[:,:2]-p.mean(1)[:,:2]).norm(dim=-1))
 return tuple(torch.stack(x,-1) for x in (counts,depths,clear,blocked,dist,forces))

def rroot(env):return env.scene['robot'].data.root_link_pos_w

@requires_model_fields(*FIELDS,recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,bank_path='',arm='M0_R2_mix',evaluation=False,stress_upper=1.):
 initialize(env,bank_path,evaluation,arm);s=env._mix;dev=env.device
 ids=torch.arange(env.num_envs,device=dev) if env_ids is None else env_ids;n=len(ids)
 level=2 if evaluation else min(2,env.common_step_counter//(24*1500))
 coh=s.cohort[ids];draw=torch.floor(torch.rand(n,generator=s.rng,device=dev)*s.lengths[level,coh]).long();ix=s.pool[level,coh,draw]
 s.sample_indices[ids]=ix
 for field,key in (('geom_size','size'),('geom_pos','pos'),('geom_quat','quat')):getattr(env.sim.model,field)[ids[:,None],s.gids[None]]=s.bank[key][ix]
 env.sim.model.geom_aabb[ids[:,None],s.gids[None],0]=0;env.sim.model.geom_aabb[ids[:,None],s.gids[None],1]=s.bank['size'][ix]
 env.sim.model.geom_rbound[ids[:,None],s.gids[None]]=s.bank['size'][ix].norm(dim=-1)
 for field in ('mass','inertia','ipos','iquat'):getattr(env.sim.model,'body_'+field)[ids[:,None],s.bids[None]]=s.bank[field][ix]
 poses=s.bank['poses'][ix]
 env.sim.model.body_pos[ids,s.bids[0]]=poses[:,0,:3];env.sim.data.qpos[ids,s.qadr[0]]=0;env.sim.data.qvel[ids,s.dadr[0]]=0
 for k in (1,2):
  env.sim.data.qpos[ids,s.qadr[k]:s.qadr[k]+7]=poses[:,k];env.sim.data.qvel[ids,s.dadr[k]:s.dadr[k]+6]=0
 r=env.scene['robot'];q=s.bank['qpos'][ix];state=r.data.default_root_state[ids].clone();state[:,:7]=q[:,:7];state[:,7:]=0
 r.write_root_state_to_sim(state,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids)
 # Preserve component mass/inertia, matched kp/kd, and 0--10ms physical command delay.
 width=.1+.1*min(env.common_step_counter/(24*2000),1.)
 rand=lambda *shape:torch.rand(shape,generator=s.rng,device=dev)
 enabled=(rand(n)>=.25)&(not evaluation);factors=1+(2*rand(n,4)-1)*width;gains=1+(2*rand(n,6)-1)*width
 factors[~enabled]=1;gains[~enabled]=1;factors[:,3]=1;factors[:,1]*=stress_upper
 env._bd_mass[ids]=factors;env._bd_gain[ids]=gains;env._bd_nominal[ids]=~enabled
 env._bd_lag[ids]=torch.where(enabled,torch.randint(0,6,(n,),generator=s.rng,device=dev),0)
 scale=factors[:,env._bd_groups]
 for field in ('body_mass','body_inertia'):
  base=env.sim.get_default_field(field)[s.robot_bids];getattr(env.sim.model,field)[ids[:,None],s.robot_bids[None]]=base[None]*(scale if field=='body_mass' else scale[:,:,None])
 for act,g in zip(r.actuators,env._bd_act_groups):act.set_gains(ids,kp=act.default_stiffness[ids]*gains[:,g,None],kd=act.default_damping[ids]*gains[:,g,None])
 s.active[ids]=s.bank['active'][ix];env.sim.forward()
 count,depth,clear,blocked,distance,_=geometry(env)
 s.best_depth[ids]=depth[ids];s.best_clear[ids]=clear[ids];s.initial_count[ids]=count[ids].float();s.best_distance[ids]=distance[ids]
 s.clear_age[ids]=0;s.ever[ids]=blocked[ids];s.blocked[ids]=blocked[ids];s.escaped[ids]=False;s.invalid[ids]=False;s.values[ids]=0
 s.last_episode_peaks[ids]=s.peaks[ids];s.timer[ids]=0;s.stall_acc[ids]=0;s.peaks[ids]=0;s.longest[ids]=0;s.motion[:,ids]=r.data.joint_pos[ids][None]
 prime_static_history(env,ids);v.reset(env,ids)


def sample_substep(env):
 if not hasattr(env,'_mix'):return torch.zeros(env.num_envs,device=env.device)
 s=env._mix;r=env.scene['robot']
 if s.subtick%env.cfg.decimation==0:s.stall_acc.zero_()
 s.timer,c=stalled_update(s.timer,r.data.qfrc_actuator/env._r_limits,r.data.joint_vel,env.physics_dt)
 s.stall_acc+=c;s.longest=torch.maximum(s.longest,s.timer)
 s.peaks=torch.maximum(s.peaks,torch.stack((r.data.qfrc_actuator.abs(),r.data.joint_vel.abs(),(r.data.qfrc_actuator*r.data.joint_vel).abs()),-1))
 s.subtick+=1
 return c.amax(-1)


def update(env):
 s=env._mix
 if s.tick==env.common_step_counter:return
 s.tick=env.common_step_counter
 count,depth,clear,blocked,distance,f=geometry(env)
 s.count=count;s.depth=depth;s.clearance=clear;s.force=f;s.blocked=blocked;s.ever|=blocked
 s.clear_age=torch.where(s.active&~blocked,s.clear_age+1,0)
 s.escaped=((~s.active)|(s.clear_age>=15)).all(-1)
 s.invalid=(s.active&(f>1500)).any(-1)
 for name in ('guide','free0','free1','fixed'):
  s.invalid|=(env.scene['mix_contact_'+name].data.dist<-.02).any(-1)
 dc=(s.best_depth-depth).clamp_min(0);dd=(clear-s.best_clear).clamp_min(0)
 s.best_depth=torch.minimum(s.best_depth,depth);s.best_clear=torch.maximum(s.best_clear,clear)
 sep=((distance-s.best_distance)/.025).clamp(0,1);s.best_distance=torch.maximum(s.best_distance,distance)
 supported=torch.maximum((env.scene['mix_hands'].data.found>0).float(),(env.scene['mix_hands_support'].data.found>0).float()).mean(-1)
 eligible=s.active&s.ever&~s.invalid[:,None]
 path=((dc/.025).clamp(0,1)+.5*(dd/.02).clamp(0,1))*eligible*blocked
 path=path.sum(-1)/s.active.sum(-1).clamp_min(1)*supported*(height(env)<=.9)
 dense=(.85*(1-count/s.initial_count.clamp_min(1)).clamp(0,1)+.15*(clear/.04).clamp(0,1))*s.active
 if s.arm!='M0_R2_mix':
  sep*=blocked;dense=torch.where(s.active&~blocked,torch.ones_like(dense),dense)
 sep=(sep*eligible).sum(-1)/s.active.sum(-1).clamp_min(1)
 active=s.active.any(-1)
 dense=dense.sum(-1)/s.active.sum(-1).clamp_min(1)
 contact_cost=((((f-300).clamp_min(0)/300).square())*s.active).amax(-1)
 r=env.scene['robot'];u=-r.data.projected_gravity_b[:,2];gate=((height(env)-.85)/.30).clamp(0,1)*((u-.70)/.23).clamp(0,1)
 foot=r.data.body_link_lin_vel_w[:,env._r_feet].square().sum(-1).mean(-1);quiet=gate*(foot/.1**2).clamp_max(10)
 s.motion[s.cursor]=r.data.joint_pos;s.cursor=(s.cursor+1)%25;span=s.motion.amax(0)-s.motion.amin(0)
 stall=(s.stall_acc/env.cfg.decimation*(1-span/.1).clamp(0,1)).amax(-1)
 s.values.copy_(torch.stack((path,dense,s.escaped*active,contact_cost,sep,quiet,stall),-1))


def task(env,task_terms,**kwargs):
 from smp.rl.tasks.getup.fixed_low_reset import recorded_task_smp_product
 update(env);result=recorded_task_smp_product(env,task_terms,**kwargs)
 factor=torch.where(env._mix.active.any(-1)&~env._mix.escaped,.05,1.)
 return result*factor

def reward(env,index):update(env);return env._mix.values[:,index]
def invalid(env):update(env);return env._mix.invalid

def stable(env):
 quality.cache_control(env);s=env._mix
 moving=torch.zeros(env.num_envs,device=env.device)
 for j,name in enumerate(('guide','free0','free1')):
  supporting=env.scene['quality_feet_'+name].data.force[...,2].abs().amax(-1)>20
  velocity=env.sim.data.qvel[:,s.dadr[j]:s.dadr[j]+(1 if j==0 else 6)].norm(dim=-1)
  moving=torch.maximum(moving,velocity*supporting)
 return env._r_stable&s.escaped&~s.invalid&(moving<.15)
