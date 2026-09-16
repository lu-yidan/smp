"""Behavioral invariants for stage caching, pose quality and vertical speed."""
from types import SimpleNamespace as N
import torch
from smp.rl.tasks.getup import v33_reward_transfer as m
n=4;q=torch.zeros(n,29);q[:,:2]=1.2
r=N(data=N(joint_pos=q,joint_vel=torch.zeros_like(q),joint_acc=torch.zeros_like(q),qfrc_actuator=torch.zeros_like(q),site_pos_w=torch.zeros(n,1,3),site_lin_vel_w=torch.zeros(n,1,3),projected_gravity_b=torch.zeros(n,3),body_link_lin_vel_w=torch.zeros(n,2,3),root_link_lin_vel_w=torch.zeros(n,3),root_link_ang_vel_w=torch.zeros(n,3)))
class Scene(dict):pass
scene=Scene(robot=r,quality_feet=N(data=N(force=torch.ones(n,2,3)*30)));scene.env_origins=torch.zeros(n,3)
e=N(scene=scene,num_envs=n,device='cpu',common_step_counter=0,cfg=N(decimation=10),physics_dt=.002,step_dt=.02,episode_length_buf=torch.ones(n,dtype=torch.long)*10,action_manager=N(action=torch.zeros_like(q)),_r_accum=True,_r_head=0,_r_feet=[0,1],_r_knees=[0,1],_r_limits=torch.ones(29)*25,_f_effort_ms=torch.zeros_like(q))
m.init(e);r.data.site_pos_w[:,:,2]=.35;r.data.projected_gravity_b[:,2]=-.2
r.data.site_lin_vel_w[:,0,2]=torch.tensor([.0,.06,.5,2.])
t=m.components(e).clone();assert t[1,1]>t[0,1]>t[2,1]>t[3,1]
m.sample_substep(e);assert e._v_accum[0,4]==0 and e._v_accum[3,4]>e._v_accum[2,4]>0
# Same control step cannot advance stages twice, including a different env reset.
e.common_step_counter+=1;e._v_stage[:]=0;r.data.site_pos_w[:,:,2]=.6;r.data.projected_gravity_b[:,2]=-.6;r.data.site_lin_vel_w.zero_()
m.components(e);before=e._v_hold.clone();m.components(e);assert torch.equal(before,e._v_hold)
m.reset(e,torch.tensor([0]));m.components(e);assert torch.equal(before[1:],e._v_hold[1:])
# An upright but deeply crouched pose must score below a normal standing pose.
e.common_step_counter+=1;e._v_stage[:]=3;r.data.site_pos_w[:,:,2]=1.15;r.data.projected_gravity_b[:,2]=-1
q[:,:2]=.3;q[1,:2]=1.5
t=m.components(e);assert t[0,0]>t[1,0]
print('PASS: slow rise preference, overspeed cost, one stage update per step, reset isolation, crouch penalty')
