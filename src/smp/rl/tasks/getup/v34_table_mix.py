"""V34 93D continuation: fixed tables / passive vertical plate / flat recovery."""
import numpy as np
import torch
from smp.rl.tasks.getup import crawl_first as c, multiterrain as mt
from smp.rl.tasks.getup.natural_low_reset import prime_static_history


def setup(env):
 c.setup(env)
 if hasattr(env,'_mix_scene'):return
 n=env.num_envs;dev=env.device;nt=2*round(n*.6/2);ng=2*round(n*.3/2)
 labels=torch.cat([torch.ones(nt//2,device=dev,dtype=torch.long),torch.full((ng//2,),2,device=dev,dtype=torch.long),torch.zeros((n-nt-ng)//2,device=dev,dtype=torch.long)])
 rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+8821);env._mix_scene=labels[torch.randperm(n//2,device=dev,generator=rng)].repeat_interleave(2)
 env._mix_middle=torch.zeros(n,device=dev,dtype=torch.bool);flat=torch.where(env._mix_scene==0)[0];env._mix_middle[flat[len(flat)//2:]]=True
 # Every table reset starts underneath: two thirds deep, one third near-edge.
 env._cr_slot[:]=((torch.arange(n,device=dev)//2)%3)*.3+.05
 env._mix_guided_gid=env.scene['escape_obstacle'].indexing.geom_ids[0].long()
 b=np.load('outputs/v34_mix_middle/validation.npz' if env._tb_evaluation else 'outputs/v34_mix_middle/train.npz')
 env._mix_middle_q=torch.tensor(b['qpos'],device=dev,dtype=torch.float32)
 env._mix_tick=-1


def data_reset(env,env_ids=None,bank_path=None):
 setup(env);ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 c.table_data_reset(env,ids,bank_path)
 env._mt_plate_ids=[env._tb_gids[0],env._mix_guided_gid]
 other=ids[env._mix_scene[ids]!=1];r=env.scene['robot']
 park=torch.zeros((len(other),7),device=env.device);park[:,:3]=env.scene.env_origins[other]+park.new_tensor([20,20,.1]);park[:,3]=1
 for j,name in enumerate(c.PARTS):
  park[:,:3]=env.scene.env_origins[other]+park.new_tensor([40+3*j,40,2]);env.scene[name].write_mocap_pose_to_sim(park,env_ids=other)
 middle=ids[env._mix_middle[ids]]
 if len(middle):
  pick=torch.randint(len(env._mix_middle_q),(len(middle),),generator=env._tb_rng,device=env.device);q=env._mix_middle_q[pick];state=r.data.default_root_state[middle].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[middle];state[:,3:7]=q[:,3:7];state[:,7:]=0
  r.write_root_state_to_sim(state,env_ids=middle);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=middle)
 env.sim.forward();pos,ext=mt.robot_bounds(env);gi=ids[env._mix_scene[ids]==2]
 if len(gi):
  pose=torch.zeros((len(gi),7),device=env.device);pose[:,3]=1;pose[:,:2]=r.data.root_link_pos_w[gi,:2]
  covered=((pos[gi,:,:2]-pose[:,None,:2]).abs()<ext[gi,:,:2]+pose.new_tensor([.46,.33])).all(-1)
  pose[:,2]=torch.where(covered,pos[gi,:,2]+ext[gi,:,2],-torch.inf).amax(-1)+.035+.002
  guided=env.scene['escape_obstacle'];guided.write_mocap_pose_to_sim(pose,env_ids=gi)
  bid=guided.indexing.body_ids[-1].long();mass=torch.full((len(gi),),8.,device=env.device) if env._tb_evaluation else 4+8*torch.rand(len(gi),generator=env._tb_rng,device=env.device)
  env.sim.model.body_mass[gi,bid]=mass;env.sim.model.body_inertia[gi,bid]=env.sim.get_default_field('body_inertia')[bid]*mass[:,None]/env.sim.get_default_field('body_mass')[bid]
 env._mt_scene[ids]=env._mix_scene[ids];flat=env._mix_scene[ids]==0
 env._cr_kind[ids]=torch.where(flat,2,env._cr_kind[ids].clamp_max(1));env._tb_done[ids]=flat;env._cr_clear[ids]=flat;env._cr_beta[ids]=flat.float();env._cr_clear_hold[ids]=flat.float()*15
 env._mt_ever_contact[ids]=~flat;env._mt_escaped[ids]=False;env._mt_invalid[ids]=False
 env.sim.forward();score,_=mt.plate_geometry(env);env._mt_best[ids]=score[ids];env._mt_initial_overlap[ids]=(-score[ids]).clamp_min(.01)
 which=(env._mt_scene[ids]==2).long();gids=torch.stack(env._mt_plate_ids)[which];env._mt_best_distance[ids]=(r.data.root_link_pos_w[ids,:2]-env.sim.data.geom_xpos[ids,gids,:2]).norm(dim=-1)
 env._tb_anchor[ids]=r.data.root_link_pos_w[ids,:2]
 if len(ids)==env.num_envs:env._mix_tick=-1
 prime_static_history(env,ids)


def update(env,env_ids=None):
 if not hasattr(env,'_mix_scene') or env._mix_tick==env.common_step_counter:return
 env._mix_tick=env.common_step_counter;mt.update(env,env_ids)
 table=env._mix_scene==1;guided=env._mix_scene==2;flat=env._mix_scene==0
 sensors=[env.scene[name].data for name in ('guided_contact',)+tuple('contact_'+x for x in c.PARTS[1:])]
 contact=torch.stack([(s.found>0).any(-1) for s in sensors]).any(0);force=torch.stack([s.force.norm(dim=-1).amax(-1) for s in sensors]).amax(0)
 gs=env.scene['free_contact'].data;env._mt_force=torch.where(guided,gs.force.norm(dim=-1).amax(-1),torch.where(table,force,0.))
 pos,ext=mt.robot_bounds(env);pp=env.sim.data.geom_xpos[:,env._tb_gids];ss=env.sim.model.geom_size[:,env._tb_gids]
 xy=((pos[:,:,:2]-pp[:,None,0,:2]).abs()-ext[:,:,:2]-ss[:,None,0,:2]).amax(-1).amin(-1)
 gap=((pos[:,:,None,:]-pp[:,None,:,:]).abs()-ext[:,:,None,:]-ss[:,None,:,:]).amax(-1).amin(dim=(1,2))
 clear=torch.where(table,(xy>=.025)&(gap>=.025)&~contact,torch.where(guided,(env._mt_clearance>=.025)&~(gs.found>0).any(-1),flat))
 env._cr_clear_hold=torch.where(clear,env._cr_clear_hold+1,0.);env._cr_clear=env._cr_clear_hold>=15
 env._cr_beta=torch.where(env._cr_clear,(env._cr_beta+env.step_dt/.5).clamp_max(1),0.)
 env._mt_escaped=env._cr_clear&~flat;first=env._mt_escaped&~env._tb_done;r=env.scene['robot'];env._tb_anchor[first]=r.data.root_link_pos_w[first,:2];env._tb_done|=env._mt_escaped
 env._tb_escape_age+=env._tb_done*env.step_dt;dist=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1);env._tb_drift=torch.maximum(env._tb_drift,torch.where(env._tb_done,dist,0.))
 good=(mt.height(env)>=1.15)&(-r.data.projected_gravity_b[:,2]>=.9)&(mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)>20)
 env._tb_hold=torch.where(good,env._tb_hold+env.step_dt,0.);env._tb_episode_sr3|=env._tb_hold>=3.


def contact_sample(env):
 if not hasattr(env,'_mix_scene'):return torch.zeros(env.num_envs,device=env.device)
 if env._cr_force_tick%env.cfg.decimation==0:env._cr_force_sum.zero_()
 force=torch.where(env._mix_scene==2,env.scene['free_contact'].data.force.norm(dim=-1).amax(-1),env.scene['guided_contact'].data.force.norm(dim=-1).amax(-1))
 env._cr_force_sum+=((force-85).clamp_min(0)/170).square().clamp_max(25);env._cr_force_peak=torch.maximum(env._cr_force_peak,force);env._cr_force_tick+=1
 return force


def install():
 mt.reset=data_reset;c.update=update
 from smp.rl.tasks.getup import v33_reward_transfer as v
 v.components=c.components
