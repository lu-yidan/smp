"""Archived V34 reward/control with flat, guided, legged and legless fixed tables."""
from dataclasses import replace
from pathlib import Path
import numpy as np,torch,mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from mjlab.sensor.contact_sensor import ContactMatch
from smp.rl.tasks.getup.mdp import events as old
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
PARTS=['mo_top']+['mo_leg'+str(i) for i in range(4)]
BANKROOT=Path('/root/workplace/smp-v34-table-mix/outputs/crawl_bank_v1')
def part_spec():
 s=mujoco.MjSpec();b=s.worldbody.add_body(name='part');b.add_geom(name='box',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.5,.5,.025),rgba=(.53,.32,.15,1),friction=(1.2,.01,.001));return s

def init(env,evaluation=False):
 if hasattr(env,'_mo_scene'):return
 n=env.num_envs;dev=env.device
 # Even quotas allow exactly balanced poses within each scene.
 counts=[2*round(n*.3/2),2*round(n*.3/2),2*round(n*.2/2)];counts.append(n-sum(counts))
 env._mo_scene=torch.repeat_interleave(torch.arange(4,device=dev),torch.tensor(counts,device=dev));env._mo_pose=torch.arange(n,device=dev)%2
 env._mo_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+1381)
 b=np.load(BANKROOT/('validation.npz' if evaluation else 'train.npz'))
 env._mo_bank={k:torch.tensor(b[k],device=dev) for k in ['qpos','positions','sizes','dims','friction']}
 env._mo_pools={(p,s):torch.tensor(np.flatnonzero((b['pose']==p)&(b['source']==s)&(b['dims'][:,2]>=.32)&(b['dims'][:,2]<=.40)),device=dev) for p in range(2) for s in range(2)}
 assert all(len(x) for x in env._mo_pools.values())
 env._mo_source=(torch.arange(n,device=dev)//2)%2;env._mo_rows=torch.zeros(n,device=dev,dtype=torch.long)
 env._mo_gids=torch.stack([env.scene[name].indexing.geom_ids[0] for name in PARTS]).long()

def geometry(env,ids):
 _,_,pos,ext,_=old._collision_vertical_geometry(env,ids,COLLISION_PATTERN)
 gid=env._mo_gids[0];center=env.sim.data.geom_xpos[ids,gid];size=env.sim.model.geom_size[ids,gid]
 overlap=size[:,None,:2]+ext[:,:,:2]-(pos[:,:,:2]-center[:,None,:2]).abs();covered=(overlap>0).all(-1)
 depth=overlap.clamp_min(0).amin(-1);score=torch.where(covered,depth,0).sum(-1)
 clearance=(-overlap).clamp_min(0).norm(dim=-1).amin(-1)
 return covered.sum(-1),score,clearance

@requires_model_fields('geom_size','geom_aabb','geom_rbound','geom_friction','body_mass','body_inertia',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,evaluation=False,**params):
 init(env,evaluation);ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 # Retain original procedural robot reset for flat/guided. No cross-scene replay.
 for enabled in [False,True]:
  sel=ids[(env._mo_scene[ids]==1)==enabled]
  if len(sel):old.reset_guided_escape_plate_curriculum(env,sel,**dict(params,obstacle_probability=float(enabled),inactive_xy=(25.,25.)))
 park=torch.zeros((len(ids),7),device=env.device);park[:,3]=1
 for j,name in enumerate(PARTS):
  park[:,:3]=env.scene.env_origins[ids]+park.new_tensor([30+3*j,30,2]);env.scene[name].write_mocap_pose_to_sim(park,env_ids=ids)
 ti=ids[env._mo_scene[ids]>=2];r=env.scene['robot']
 if len(ti):
  for p in [0,1]:
   for s in [0,1]:
    sel=ti[(env._mo_pose[ti]==p)&(env._mo_source[ti]==s)];pool=env._mo_pools[p,s]
    env._mo_rows[sel]=pool[torch.randint(len(pool),(len(sel),),generator=env._mo_rng,device=env.device)]
  ix=env._mo_rows[ti];q=env._mo_bank['qpos'][ix];state=r.data.default_root_state[ti].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[ti];state[:,3:7]=q[:,3:7];state[:,7:]=0
  r.write_root_state_to_sim(state,env_ids=ti);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ti)
  env._robust_reset_type[ti]=torch.where(env._mo_pose[ti]==0,2,1)
  sizes=env._mo_bank['sizes'][ix];positions=env._mo_bank['positions'][ix]+env.scene.env_origins[ti,None,:]
  for j,name in enumerate(PARTS):
   gid=env._mo_gids[j];env.sim.model.geom_size[ti,gid]=sizes[:,j];env.sim.model.geom_aabb[ti,gid,0]=0;env.sim.model.geom_aabb[ti,gid,1]=sizes[:,j];env.sim.model.geom_rbound[ti,gid]=sizes[:,j].norm(dim=-1);env.sim.model.geom_friction[ti,gid,0]=env._mo_bank['friction'][ix]
   pose=torch.zeros((len(ti),7),device=env.device);pose[:,3]=1;pose[:,:3]=positions[:,j]
   if j:pose[env._mo_scene[ti]==3,:3]=env.scene.env_origins[ti[env._mo_scene[ti]==3]]+pose.new_tensor([30+3*j,30,2])
   env.scene[name].write_mocap_pose_to_sim(pose,env_ids=ti)
  env.sim.forward()
  # Verified initial entrapment activates guidance even before touching the roof.
  env._escape_phase[ti]=2;env._escape_contact_ever[ti]=True;env._escape_sensor_grace[ti]=0
  count,score,clearance=geometry(env,ti)
  for key,val in [('initial_covered_geom_count',count),('covered_geom_count',count),('best_covered_geom_count',count),('coverage_score',score),('best_coverage_score',score),('planar_clearance',clearance),('best_planar_clearance',clearance)]:getattr(env,'_escape_'+key)[ti]=val
  env._escape_geometry_initialized[ti]=True
  xy=env.sim.data.geom_xpos[ti,env._mo_gids[0],:2];env._escape_start_obstacle_xy[ti]=xy;env._escape_best_separation[ti]=(r.data.root_link_pos_w[ti,:2]-xy).norm(dim=-1)
 env.sim.forward();old._prime_smp_history_from_current_state(env,ids)

def update(env,env_ids=None,**params):
 if not hasattr(env,'_mo_scene'):return
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 gi=ids[env._mo_scene[ids]<2]
 if len(gi):old.update_escape_phase(env,gi,**params)
 ti=ids[env._mo_scene[ids]>=2]
 if not len(ti):return
 sensors=[env.scene['mo_contact'+str(i)].data for i in range(5)]
 contact=torch.stack([(x.found[ti]>0).any(-1) for x in sensors]).any(0)
 penetration=torch.stack([torch.where(x.found[ti]>0,(-x.dist[ti]).clamp_min(0),0).amax(-1) for x in sensors]).amax(0);force=torch.stack([x.force[ti].norm(dim=-1).amax(-1) for x in sensors]).amax(0)
 env._escape_peak_penetration[ti]=torch.maximum(env._escape_peak_penetration[ti],penetration);env._escape_peak_contact_force[ti]=torch.maximum(env._escape_peak_contact_force[ti],force)
 bad=penetration>.02;env._escape_invalid_contact[ti]|=bad;env._escape_phase[ti[bad]]=4
 active=env._escape_phase[ti]==2
 count,score,clearance=geometry(env,ti)
 env._escape_coverage_delta[ti]=torch.where(active,(env._escape_best_coverage_score[ti]-score).clamp_min(0),0)
 env._escape_clearance_delta[ti]=torch.where(active,(clearance-env._escape_best_planar_clearance[ti]).clamp_min(0),0)
 for key,val in [('covered_geom_count',count),('coverage_score',score),('planar_clearance',clearance)]:getattr(env,'_escape_'+key)[ti]=val
 env._escape_best_covered_geom_count[ti]=torch.minimum(env._escape_best_covered_geom_count[ti],count);env._escape_best_coverage_score[ti]=torch.minimum(env._escape_best_coverage_score[ti],score);env._escape_best_planar_clearance[ti]=torch.maximum(env._escape_best_planar_clearance[ti],clearance)
 r=env.scene['robot'];sep=(r.data.root_link_pos_w[ti,:2]-env._escape_start_obstacle_xy[ti]).norm(dim=-1);delta=(sep-env._escape_best_separation[ti]).clamp_min(0);env._escape_separation_delta[ti]=torch.where(active,delta,0);env._escape_best_separation[ti]=torch.maximum(env._escape_best_separation[ti],sep)
 hand=(env.scene['hand_ground_contact'].data.found[ti]>0).any(-1)&active;env._escape_hand_support_steps[ti]+=hand.long();env._escape_hand_supported_progress[ti]+=hand*delta
 support=(env._escape_hand_support_steps[ti]>=params['min_hand_support_steps'])&(env._escape_hand_supported_progress[ti]>=params['min_hand_supported_progress'])
 clear=active&(count==0)&(clearance>=.025)&~contact&support;hold=torch.where(clear,env._escape_clear_hold[ti]+1,0);env._escape_clear_hold[ti]=torch.where(env._escape_phase[ti]==3,env._escape_clear_hold[ti],hold);env._escape_phase[ti[hold>=15]]=3

def force_cost(env,sensor_name='robot_obstacle_contact',force_limit=300.,force_scale=300.):
 peak=env.scene[sensor_name].data.force.norm(dim=-1).amax(-1)
 if hasattr(env,'_mo_scene'):
  table_force=torch.stack([env.scene['mo_contact'+str(i)].data.force.norm(dim=-1).amax(-1) for i in range(5)]).amax(0)
  peak=torch.where(env._mo_scene>=2,table_force,peak)
 constrained=(env._escape_phase==1)|(env._escape_phase==2)
 return constrained.float()*((peak-force_limit).clamp_min(0)/force_scale).square()

def configure(cfg,evaluation=False):
 for name in PARTS:cfg.scene.entities[name]=EntityCfg(spec_fn=part_spec,init_state=EntityCfg.InitialStateCfg(pos=(30,30,2)))
 original=next(s for s in cfg.scene.sensors if s.name=='robot_obstacle_contact')
 cfg.scene.sensors+=tuple(replace(original,name='mo_contact'+str(i),secondary=ContactMatch(mode='geom',pattern=name+'/box')) for i,name in enumerate(PARTS))
 for name in ['record_failure_states','failure_state_replay_reset']:cfg.events.pop(name,None)
 cfg.events['reset_escape_obstacle'].func=reset;cfg.events['reset_escape_obstacle'].params['evaluation']=evaluation
 cfg.events['update_escape_phase'].func=update
 for term in cfg.rewards.values():
  if term.func.__name__=='escape_contact_force_excess_l2':term.func=force_cost
 cfg.episode_length_s=1000 if evaluation else 20
 return cfg
