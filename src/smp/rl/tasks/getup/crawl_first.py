"""Fixed-table specialist; frozen A6 task/Q/L costs plus explicit table interventions."""
from pathlib import Path
import numpy as np,torch,mujoco
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import multiterrain as mt,v33_path_ablation as pa,r2_ablation as ra,balanced_dynamics as bd
from smp.rl.tasks.getup import v33_reward_transfer as v
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
PARTS=('table_top','table_leg0','table_leg1','table_leg2','table_leg3')
ORIGINAL_MT_RESET=mt.reset

def part_spec():
 s=mujoco.MjSpec();b=s.worldbody.add_body(name='part');b.add_geom(name='box',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.5,.5,.025),rgba=(.53,.32,.15,1.),friction=(1.2,.01,.001));return s

def setup(env):
 if hasattr(env,'_tb_bank'):return
 n=env.num_envs;dev=env.device;b=np.load(env._tb_bank_path)
 env._tb_bank={k:torch.as_tensor(b[k],device=dev) for k in ('qpos','positions','sizes','dims','friction')}
 env._tb_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+37103)
 counts=[2*round(n*.35/2),2*round(n*.40/2)];counts+=[n-sum(counts)]
 env._tb_family=torch.repeat_interleave(torch.arange(3,device=dev),torch.tensor(counts,device=dev));env._tb_pose=torch.arange(n,device=dev)%2;env._tb_source=torch.zeros(n,device=dev,dtype=torch.long)
 env._cr_slot=torch.zeros(n,device=dev);env._cr_side=torch.zeros(n,device=dev,dtype=torch.long)
 for f in range(3):
  for p in range(2):
   ids=torch.where((env._tb_family==f)&(env._tb_pose==p))[0];env._tb_source[ids[:len(ids)//2]]=1
   for source in range(2):
    si=ids[env._tb_source[ids]==source];env._cr_slot[si]=(torch.arange(len(si),device=dev)+.5)/max(len(si),1);env._cr_side[si]=torch.arange(len(si),device=dev)%4
 keys=('family','pose','source','kind','side','height_bin')
 cells=[]
 for f in range(3):
  for p in range(2):
   for source in range(2):
    for kind in range(3):
     for side in range(4):
      for hb in range(8):cells.append(np.flatnonzero(np.logical_and.reduce([b[k]==val for k,val in zip(keys,(f,p,source,kind,side,hb))])))
 assert len(set(map(len,cells)))==1 and len(cells[0])>0
 env._cr_cells=torch.tensor(np.stack(cells),device=dev)
 env._tb_band=torch.zeros(n,dtype=torch.long,device=dev);env._tb_rows=env._tb_band.clone();env._cr_kind=env._tb_band.clone();env._cr_height_bin=env._tb_band.clone()
 env._cr_beta=torch.zeros(n,device=dev);env._cr_clear_hold=torch.zeros(n,device=dev);env._cr_clear=torch.zeros(n,device=dev,dtype=torch.bool)
 env._cr_component_tick=-1;env._cr_phase_tick=-1;env._cr_force_tick=0;env._cr_force_sum=torch.zeros(n,device=dev);env._cr_force_peak=torch.zeros(n,device=dev)
 env._tb_gids=torch.stack([env.scene[name].indexing.geom_ids[0] for name in PARTS]).long()
 env._mt_plate_ids=[env._tb_gids[0],env._tb_gids[0]]
 env._tb_done=torch.zeros(n,dtype=torch.bool,device=dev);env._tb_anchor=torch.zeros(n,2,device=dev);env._tb_drift=torch.zeros(n,device=dev);env._tb_ever_stand=torch.zeros(n,dtype=torch.bool,device=dev)
 env._tb_escape_age=torch.zeros(n,device=dev);env._tb_hold=torch.zeros(n,device=dev);env._tb_episode_sr3=torch.zeros(n,dtype=torch.bool,device=dev)
 env._tb_reset_counts=torch.zeros(3,2,device=dev);env._tb_validated=0

def table_data_reset(env,env_ids=None,bank_path=None):
 setup(env);ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 updates=env.common_step_counter/24
 deep=.6
 if env._tb_arm=='CR5' and not env._tb_evaluation:deep=.3+.3*max(0.,min(1.,(updates-2000)/2000))
 slot=env._cr_slot[ids];kind=torch.where(slot<deep,0,torch.where(slot<.9,1,2));env._cr_kind[ids]=kind
 hb=torch.randint(8,(len(ids),),generator=env._tb_rng,device=env.device);env._cr_height_bin[ids]=hb;env._tb_band[ids]=(hb<4).long()
 cell=(((((env._tb_family[ids]*2+env._tb_pose[ids])*2+env._tb_source[ids])*3+kind)*4+env._cr_side[ids])*8+hb)
 choice=torch.randint(env._cr_cells.shape[1],(len(ids),),generator=env._tb_rng,device=env.device)
 env._tb_rows[ids]=env._cr_cells[cell,choice]
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
 env._mt_scene[ids]=torch.where(env._cr_kind[ids]==2,0,1);env._mt_direction[ids]=env._tb_pose[ids];env._mt_source[ids]=env._tb_source[ids]
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
 env._cr_force_peak[ids]=0
 outside=env._cr_kind[ids]==2
 env._tb_done[ids]=outside;env._cr_beta[ids]=outside.float();env._cr_clear[ids]=outside;env._cr_clear_hold[ids]=outside.float()*15
 env._mt_ever_contact[ids]=~outside
 if len(ids)==env.num_envs:env._cr_component_tick=-1;env._cr_phase_tick=-1;env._v_cache=-1
 prime_static_history(env,ids)

@requires_model_fields('body_mass','body_inertia','geom_size','geom_aabb','geom_rbound','geom_friction',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='CR0',bank_path='outputs/crawl_bank_v1/train.npz',evaluation=False):
 env._tb_arm=arm;env._tb_bank_path=bank_path;env._tb_evaluation=evaluation
 if not hasattr(env,'_tb_low_fraction'):env._tb_low_fraction=0.
 # All legacy buffers and dynamics randomization are retained; only mt.reset's physical reset is replaced.
 pa.reset(env,env_ids,arm='A6',bank_path='outputs/multiterrain_bank/train.npz',dynamics=not evaluation,evaluation=True)
 env._ra_kind.zero_();env._fixed_group=env._tb_pose+2;env._fixed_source=env._tb_source

ORIGINAL_UPDATE=mt.update

def update(env,env_ids=None):
 if not hasattr(env,'_tb_bank') or env._cr_phase_tick==env.common_step_counter:return
 env._cr_phase_tick=env.common_step_counter
 ORIGINAL_UPDATE(env,env_ids)
 sensors=[env.scene[name].data for name in ('guided_contact',)+tuple('contact_'+x for x in PARTS[1:])]
 contact=torch.stack([(x.found>0).any(-1) for x in sensors]).any(0)
 env._mt_force=torch.stack([x.force.norm(dim=-1).amax(-1) for x in sensors]).amax(0)
 pos,ext=mt.robot_bounds(env);pp=env.sim.data.geom_xpos[:,env._tb_gids];ss=env.sim.model.geom_size[:,env._tb_gids]
 xy=((pos[:,:,:2]-pp[:,None,0,:2]).abs()-ext[:,:,:2]-ss[:,None,0,:2]).amax(-1).amin(-1)
 gap=((pos[:,:,None,:]-pp[:,None,:,:]).abs()-ext[:,:,None,:]-ss[:,None,:,:]).amax(-1).amin(dim=(1,2))
 clear=(xy>=.025)&(gap>=.025)&~contact
 env._cr_clear_hold=torch.where(clear,env._cr_clear_hold+1,0)
 env._cr_clear=env._cr_clear_hold>=15
 env._cr_beta=torch.where(env._cr_clear,(env._cr_beta+env.step_dt/.5).clamp_max(1),0.)
 env._mt_escaped=env._cr_clear&(env._cr_kind!=2)
 first=env._mt_escaped&~env._tb_done;r=env.scene['robot'];env._tb_anchor[first]=r.data.root_link_pos_w[first,:2];env._tb_done|=env._mt_escaped
 env._tb_escape_age+=env._tb_done*env.step_dt
 drift=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1);env._tb_drift=torch.maximum(env._tb_drift,torch.where(env._tb_done,drift,0))
 z=mt.height(env);u=-r.data.projected_gravity_b[:,2];feet=mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)
 good=(z>=1.15)&(u>=.90)&(feet>20);env._tb_hold=torch.where(good,env._tb_hold+env.step_dt,0);env._tb_episode_sr3|=env._tb_hold>=3

ORIGINAL_COMPONENTS=v.components

def components(env):
 if getattr(env,'_tb_arm','CR0')=='CR0':return ORIGINAL_COMPONENTS(env)
 if env._cr_component_tick==env.common_step_counter:return env._v_task
 update(env)
 blocked=~env._cr_clear
 v.init(env);env._v_stage[blocked]=0;env._v_hold[blocked]=0
 result=ORIGINAL_COMPONENTS(env)
 env._v_stage[blocked]=0;env._v_hold[blocked]=0
 result[:,:4]*=env._cr_beta[:,None]
 env._cr_component_tick=env.common_step_counter
 return result

def clearance_cost(env):
 # Only head/torso/pelvis collisions currently covered by the tabletop.
 if not hasattr(env,'_cr_core'):
  names=[mujoco.mj_id2name(env.sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or '' for i in env._mt_geoms]
  env._cr_core=torch.tensor([any(x in name for x in ('head','torso','pelvis')) for name in names],device=env.device)
  assert env._cr_core.any(),names
 pos,ext=mt.robot_bounds(env);p=env.sim.data.geom_xpos[:,env._tb_gids[0]];s=env.sim.model.geom_size[:,env._tb_gids[0]]
 covered=(((pos[:,:,:2]-p[:,None,:2]).abs()-ext[:,:,:2]-s[:,None,:2])<0).all(-1)&env._cr_core[None,:]
 excess=(pos[:,:,2]+ext[:,:,2]-(p[:,None,2]-s[:,None,2]-.02)).clamp_min(0)
 return torch.where(covered,(excess/.05).square().clamp_max(4),0).amax(-1)

def contact_sample(env):
 if not hasattr(env,'_cr_force_sum'):return torch.zeros(env.num_envs,device=env.device)
 if env._cr_force_tick%env.cfg.decimation==0:env._cr_force_sum.zero_()
 # Reduced sensor contact-force magnitude, not an exact sum of all physical contacts.
 force=env.scene['guided_contact'].data.force.norm(dim=-1).amax(-1)
 env._cr_force_sum+=((force-85).clamp_min(0)/170).square().clamp_max(25)
 env._cr_force_peak=torch.maximum(env._cr_force_peak,force)
 if hasattr(env,'_cr_eval_contact_peak'):env._cr_eval_contact_peak=torch.maximum(env._cr_eval_contact_peak,torch.where(env._ap_alive,force,0))
 env._cr_force_tick+=1
 return force

def contact_cost(env):return env._cr_force_sum/env.cfg.decimation

ORIGINAL_PA_REWARD=pa.reward
ORIGINAL_ESCAPE_REWARD=mt.escape_reward

def path_reward(env,index):
 value=ORIGINAL_PA_REWARD(env,index)
 if  index in (0,1):value=value*~env._tb_done
 return value

def escape_reward(env,index):
 value=ORIGINAL_ESCAPE_REWARD(env,index)
 if  index in (0,1,4):value=value*~env._tb_done
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
 v.components=components
