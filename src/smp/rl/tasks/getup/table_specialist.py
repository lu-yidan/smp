"""Fixed-table specialist; frozen A6 task/Q/L costs plus explicit table interventions."""
from pathlib import Path
import numpy as np,torch,mujoco
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import multiterrain as mt,v33_path_ablation as pa,r2_ablation as ra,balanced_dynamics as bd
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
PARTS=('table_top','table_leg0','table_leg1','table_leg2','table_leg3')
ORIGINAL_MT_RESET=mt.reset

def part_spec():
 s=mujoco.MjSpec();b=s.worldbody.add_body(name='part');b.add_geom(name='box',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.5,.5,.025),rgba=(.53,.32,.15,1.),friction=(1.2,.01,.001));return s

def setup(env):
 if hasattr(env,'_tb_bank'):return
 n=env.num_envs;dev=env.device;b=np.load(env._tb_bank_path)
 env._tb_bank={k:torch.as_tensor(b[k],device=dev) for k in ('qpos','positions','sizes','dims','friction')}
 env._tb_pools={(f,p,s,h):torch.tensor(np.flatnonzero((b['family']==f)&(b['pose']==p)&(b['source']==s)&(b['band']==h)),device=dev) for f in range(3) for p in range(2) for s in range(2) for h in range(2)}
 assert all(len(x)>0 for x in env._tb_pools.values())
 env._tb_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+37103)
 # Fixed environment strata keep family/pose/source quotas exact across resets.
 counts=[2*round(n*.35/2),2*round(n*.40/2)];counts+=[n-sum(counts)]
 env._tb_family=torch.repeat_interleave(torch.arange(3,device=dev),torch.tensor(counts,device=dev));env._tb_pose=torch.arange(n,device=dev)%2;env._tb_source=torch.zeros(n,device=dev,dtype=torch.long)
 for f in range(3):
  for p in range(2):
   ids=torch.where((env._tb_family==f)&(env._tb_pose==p))[0];env._tb_source[ids[:round(len(ids)*.25)]]=1
 env._tb_band=torch.zeros(n,dtype=torch.long,device=dev);env._tb_rows=env._tb_band.clone()
 if env._tb_evaluation:env._tb_band=(torch.arange(n,device=dev)//2)%2
 env._tb_gids=torch.stack([env.scene[name].indexing.geom_ids[0] for name in PARTS]).long()
 env._mt_plate_ids=[env._tb_gids[0],env._tb_gids[0]]
 env._tb_done=torch.zeros(n,dtype=torch.bool,device=dev);env._tb_anchor=torch.zeros(n,2,device=dev);env._tb_drift=torch.zeros(n,device=dev);env._tb_ever_stand=torch.zeros(n,dtype=torch.bool,device=dev)
 env._tb_escape_age=torch.zeros(n,device=dev);env._tb_hold=torch.zeros(n,device=dev);env._tb_episode_sr3=torch.zeros(n,dtype=torch.bool,device=dev)
 env._tb_reset_counts=torch.zeros(3,2,device=dev);env._tb_validated=0

def table_data_reset(env,env_ids=None,bank_path=None):
 setup(env);ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 if not env._tb_evaluation:env._tb_band[ids]=(torch.rand(len(ids),generator=env._tb_rng,device=env.device)<env._tb_low_fraction).long()
 for f in range(3):
  for p in range(2):
   for s in range(2):
    for h in range(2):
     ei=ids[(env._tb_family[ids]==f)&(env._tb_pose[ids]==p)&(env._tb_source[ids]==s)&(env._tb_band[ids]==h)]
     if len(ei):
      pool=env._tb_pools[f,p,s,h];env._tb_rows[ei]=pool[torch.randint(len(pool),(len(ei),),generator=env._tb_rng,device=env.device)]
 ix=env._tb_rows[ids];q=env._tb_bank['qpos'][ix];r=env.scene['robot'];state=r.data.default_root_state[ids].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[ids];state[:,3:7]=q[:,3:7];state[:,7:]=0
 r.write_root_state_to_sim(state,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids)
 sizes=env._tb_bank['sizes'][ix];positions=env._tb_bank['positions'][ix]+env.scene.env_origins[ids,None,:]
 for j,name in enumerate(PARTS):
  gid=env._tb_gids[j];env.sim.model.geom_size[ids,gid]=sizes[:,j];env.sim.model.geom_aabb[ids,gid,0]=0;env.sim.model.geom_aabb[ids,gid,1]=sizes[:,j];env.sim.model.geom_rbound[ids,gid]=sizes[:,j].norm(dim=-1)
  env.sim.model.geom_friction[ids,gid,0]=env._tb_bank['friction'][ix]
  pose=torch.zeros((len(ids),7),device=env.device);pose[:,:3]=positions[:,j];pose[:,3]=1;env.scene[name].write_mocap_pose_to_sim(pose,env_ids=ids)
 park=torch.zeros((len(ids),7),device=env.device);park[:,:3]=env.scene.env_origins[ids]+park.new_tensor([20,20,.1]);park[:,3]=1
 guided=env.scene['escape_obstacle'];guided.write_mocap_pose_to_sim(park,env_ids=ids);guided.write_joint_state_to_sim(torch.zeros((len(ids),1),device=env.device),torch.zeros((len(ids),1),device=env.device),env_ids=ids)
 free=env.scene['free_obstacle'];fs=free.data.default_root_state[ids].clone();fs[:,:7]=park;fs[:,0]+=3;fs[:,7:]=0;free.write_root_state_to_sim(fs,env_ids=ids)
 env._mt_scene[ids]=1;env._mt_direction[ids]=env._tb_pose[ids];env._mt_source[ids]=env._tb_source[ids]
 env._fixed_group=env._tb_pose+2;env._fixed_source=env._tb_source;env._bd_flat_support=True
 env.sim.forward();score,_=mt.plate_geometry(env)
 env._mt_best[ids]=score[ids];env._mt_initial_overlap[ids]=(-score[ids]).clamp_min(.01)
 env._mt_best_distance[ids]=(r.data.root_link_pos_w[ids,:2]-positions[:,0,:2]).norm(dim=-1)
 for name in ('separation_progress','force','clear_hold','progress'):getattr(env,'_mt_'+name)[ids]=0
 env._mt_ever_contact[ids]=True # Eligibility from verified initial entrapment, not requiring a deliberate collision.
 env._mt_escaped[ids]=False;env._mt_invalid[ids]=False
 env._tb_escape_age[ids]=0;env._tb_done[ids]=False;env._tb_anchor[ids]=r.data.root_link_pos_w[ids,:2];env._tb_drift[ids]=0;env._tb_hold[ids]=0;env._tb_episode_sr3[ids]=False
 for f in range(3):
  for p in range(2):env._tb_reset_counts[f,p]+=((env._tb_family[ids]==f)&(env._tb_pose[ids]==p)).sum()
 prime_static_history(env,ids)

@requires_model_fields('body_mass','body_inertia','geom_size','geom_aabb','geom_rbound','geom_friction',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='T0',bank_path='outputs/table_specialist_bank_v1/train.npz',evaluation=False):
 env._tb_arm=arm;env._tb_bank_path=bank_path;env._tb_evaluation=evaluation
 if not hasattr(env,'_tb_low_fraction'):env._tb_low_fraction=0.
 # All legacy buffers and dynamics randomization are retained; only mt.reset's physical reset is replaced.
 pa.reset(env,env_ids,arm='A6',bank_path='outputs/multiterrain_bank/train.npz',dynamics=not evaluation,evaluation=True)
 env._ra_kind.zero_();env._fixed_group=env._tb_pose+2;env._fixed_source=env._tb_source

ORIGINAL_UPDATE=mt.update

def update(env,env_ids=None):
 if not hasattr(env,'_tb_bank') or env._mt_tick==env.common_step_counter:return
 ORIGINAL_UPDATE(env,env_ids)
 sensors=[env.scene[name].data for name in ('guided_contact',)+tuple('contact_'+x for x in PARTS[1:])]
 contact=torch.stack([(x.found>0).any(-1) for x in sensors]).any(0)
 env._mt_force=torch.stack([x.force.norm(dim=-1).amax(-1) for x in sensors]).amax(0)
 clear=(env._mt_clearance>=.025)&~contact
 # New completion adds clearance from all four legs, not just tabletop footprint.
 if env._tb_arm!='T0':
  pos,ext=mt.robot_bounds(env);pp=env.sim.data.geom_xpos[:,env._tb_gids];ss=env.sim.model.geom_size[:,env._tb_gids]
  sep=((pos[:,:,None,:]-pp[:,None,:,:]).abs()-ext[:,:,None,:]-ss[:,None,:,:]).amax(-1).amin(dim=(1,2))
  clear &= sep>=.025
 if not hasattr(env,'_tb_clear_hold'):env._tb_clear_hold=torch.zeros_like(env._mt_clear_hold)
 env._tb_clear_hold=torch.where(env.episode_length_buf<=1,0,env._tb_clear_hold)
 env._tb_clear_hold=torch.where(clear,env._tb_clear_hold+1,0)
 env._mt_escaped=env._tb_clear_hold>=15
 first=env._mt_escaped&~env._tb_done;r=env.scene['robot'];env._tb_anchor[first]=r.data.root_link_pos_w[first,:2];env._tb_done|=env._mt_escaped
 env._tb_escape_age+=env._tb_done*env.step_dt
 drift=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1);env._tb_drift=torch.maximum(env._tb_drift,torch.where(env._tb_done,drift,0))
 z=mt.height(env);u=-r.data.projected_gravity_b[:,2];feet=mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)
 good=(z>=1.15)&(u>=.90)&(feet>20);env._tb_hold=torch.where(good,env._tb_hold+env.step_dt,0);env._tb_episode_sr3|=env._tb_hold>=3

ORIGINAL_PA_REWARD=pa.reward
ORIGINAL_ESCAPE_REWARD=mt.escape_reward

def path_reward(env,index):
 value=ORIGINAL_PA_REWARD(env,index)
 if env._tb_arm!='T0' and index in (0,1):value=value*~env._tb_done
 return value

def escape_reward(env,index):
 value=ORIGINAL_ESCAPE_REWARD(env,index)
 if env._tb_arm!='T0' and index in (0,1,4):value=value*~env._tb_done
 return value

def displacement_cost(dist):
 return ((dist-.25).clamp_min(0)/.30).square().clamp_max(10)

def post_clear_position(env,activation="standing"):
 assert activation in ("standing","after_clear")
 r=env.scene['robot'];g=pa.quiet_gate(mt.height(env),-r.data.projected_gravity_b[:,2]);dist=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1)
 if activation=="after_clear":g=(env._tb_escape_age/.5).clamp(0,1)
 return env._tb_done*g*displacement_cost(dist)

def install():
 mt.reset=table_data_reset
