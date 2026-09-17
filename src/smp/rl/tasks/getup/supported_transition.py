"""Bounded, once-per-progress credit for supported get-up after physical escape."""
import torch

def potential(env):
 r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
 u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
 return .5*((z-.35)/.80).clamp(0,1)+.5*u,z

def reset(env,env_ids=None):
 if not hasattr(env,'_transition_best'):
  env._transition_best=torch.zeros(env.num_envs,device=env.device)
  env._transition_credit=torch.zeros_like(env._transition_best);env._transition_tick=-1
 if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
 phi,_=potential(env);env._transition_best[env_ids]=phi[env_ids];env._transition_credit[env_ids]=0

def progress(env):
 if not hasattr(env,'_transition_best'):reset(env)
 if env._transition_tick==env.common_step_counter:return env._transition_credit
 env._transition_tick=env.common_step_counter
 r=env.scene['robot'];phi,z=potential(env)
 hand=(env.scene['hand_ground_contact'].data.found>0).any(-1)
 foot=env.scene['quality_feet'].data.force[...,2].abs()>20
 support=torch.where(z<.8,hand|foot.any(-1),torch.where(z<1.,foot.any(-1),foot.all(-1)))
 phase=env._escape_phase
 eligible=((phase==0)|(phase==3))&support
 eligible&=(r.data.site_lin_vel_w[:,env._r_head,2].abs()<.20)
 eligible&=(r.data.joint_vel.square().mean(-1).sqrt()<4.)&(r.data.root_link_ang_vel_w.norm(dim=-1)<2.)
 delta=(phi-env._transition_best).clamp_min(0)
 env._transition_credit=torch.where(eligible,(delta/.02).clamp_max(1.),0.)
 # Consume the entire achieved potential even if per-step credit was capped.
 env._transition_best=torch.where(eligible,torch.maximum(env._transition_best,phi),env._transition_best)
 return env._transition_credit
