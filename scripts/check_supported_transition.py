"""Reward invariants: no repeated credit, no constrained/unsupported/fast-motion credit."""
from types import SimpleNamespace as NS
import json
from pathlib import Path
import torch
from smp.rl.tasks.getup import supported_transition as t
class Scene(dict):pass
n=4;scene=Scene();scene.env_origins=torch.zeros(n,3)
data=NS(site_pos_w=torch.zeros(n,1,3),projected_gravity_b=torch.zeros(n,3),site_lin_vel_w=torch.zeros(n,1,3),joint_vel=torch.zeros(n,29),root_link_ang_vel_w=torch.zeros(n,3))
scene['robot']=NS(data=data)
scene['hand_ground_contact']=NS(data=NS(found=torch.ones(n,2)))
scene['quality_feet']=NS(data=NS(force=torch.zeros(n,2,3)));scene['quality_feet'].data.force[...,2]=30
# Env0 valid, env1 under board, env2 no support, env3 overspeed.
env=NS(scene=scene,num_envs=n,device='cpu',_r_head=0,_escape_phase=torch.tensor([0,2,0,0]),common_step_counter=0)
scene['hand_ground_contact'].data.found[2]=0;scene['quality_feet'].data.force[2]=0
data.site_pos_w[:,:,2]=.35;data.site_lin_vel_w[:,:,2]=.1;data.site_lin_vel_w[3,:,2]=1.
t.reset(env);total=torch.zeros(n)
for i in range(1,401):
 env.common_step_counter=i;data.site_pos_w[:,:,2]=.35+.8*i/400;data.projected_gravity_b[:,2]=-i/400
 first=t.progress(env).clone();assert torch.equal(first,t.progress(env))
 total+=first*.20*.02
assert abs(float(total[0])-.20)<1e-5 and torch.equal(total[1:],torch.zeros(3)),total
before=total.clone()
for i in range(401,1201):
 env.common_step_counter=i;f=abs(i-800)/400
 data.site_pos_w[:,:,2]=.35+.8*f;data.projected_gravity_b[:,2]=-f
 total+=t.progress(env)*.20*.02
assert torch.equal(before,total), (before,total)
result={'passed':True,'bounded_credit':total.tolist(),'no_cycle_credit':True,'no_constrained_credit':True,'no_unsupported_credit':True,'no_overspeed_credit':True,'idempotent_reads':True}
Path('outputs/verification').mkdir(parents=True,exist_ok=True);Path('outputs/verification/transition_invariants.json').write_text(json.dumps(result,indent=2));print(result)
