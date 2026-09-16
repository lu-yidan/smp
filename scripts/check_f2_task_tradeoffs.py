from dataclasses import asdict
from types import SimpleNamespace
import math,torch
from train_f2_task_tradeoffs import build_config
from train_r1_quality import build_config as f_config
from smp.rl.tasks.getup import f2_task_tradeoffs as c,r1_quality as f
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
base,agent=f_config('F2',bank)
for arm in c.ARMS:
 cfg,a=build_config(arm,bank);x,y=asdict(cfg),asdict(base)
 assert cfg.rewards['sustained_effort'].weight==-.05
 assert cfg.rewards['early_speed'].weight==(c.SPEED_WEIGHT if arm in ('C2','C3') else 0)
 assert (cfg.rewards['task_smp_product'].params['task_terms'][0][0]==c.quality_upward)==(arm in ('C1','C3'))
 x['metrics'].pop('c_speed_substep');x['metrics'].pop('c_quality');x['rewards'].pop('early_speed')
 if arm in ('C1','C3'):x['rewards']['task_smp_product']=y['rewards']['task_smp_product']
 x['scene']['terrain']['spec_fn']=y['scene']['terrain']['spec_fn'];assert x==y and asdict(a)==asdict(agent)
# Do not reintroduce R3's idle-reward boost. Smoothly cross the old .9m discontinuity.
z=torch.tensor([.3,.3,.3]);v=torch.tensor([0.,.25,1.]);q=torch.zeros(3)
assert torch.allclose(c.blend_upward(z,v,q),torch.tensor([math.exp(-6.25),1.,1.]))
left=c.blend_upward(torch.tensor([.9-1e-6]),torch.zeros(1),torch.ones(1)*.7)
right=c.blend_upward(torch.tensor([.9+1e-6]),torch.zeros(1),torch.ones(1)*.7)
assert abs(float(left-right))<1e-4
assert torch.allclose(c.blend_upward(torch.ones(3)*1.2,v,torch.tensor([0.,.5,1.])),torch.tensor([0.,.5,1.]))
# Previously unpenalized early-phase speed is now measured; moderate early motion is allowed.
dq=torch.zeros(3,29);dq[0,0]=8;dq[1,0]=20;dq[2,0]=20
cost=c.speed_cost(dq,torch.ones(29)*20,torch.tensor([0.,0.,1.]))
assert cost[0]==0 and 0<cost[1]<cost[2]
e=SimpleNamespace(common_step_counter=600000,_f_start_counter=480000,_c_start_counter=600000)
assert f.ramp(e)==1 and c.ramp(e)==0
e.common_step_counter+=6000;assert f.ramp(e)==1 and c.ramp(e)==.5
print('PASS: paired 2x2 configs; F2 fully retained; low idle unchanged; smooth height transition; early overspeed covered; independent ramp')
# Exercise the actual quality callback: pose/velocity/support influence score, low idle stays low.
class Scene(dict):pass
n=4;data=SimpleNamespace(site_pos_w=torch.zeros(n,1,3),site_lin_vel_w=torch.zeros(n,1,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(n,1),body_link_pos_w=torch.zeros(n,2,3),body_link_lin_vel_w=torch.zeros(n,2,3),root_link_lin_vel_w=torch.zeros(n,3),root_link_ang_vel_w=torch.zeros(n,3),joint_vel=torch.zeros(n,29))
data.site_pos_w[:,:,2]=1.2;data.site_pos_w[3,:,2]=.3;data.body_link_pos_w[:,1,1]=.2
force=torch.zeros(n,2,3);force[:,:,2]=100;force[2,1,2]=0
scene=Scene(robot=SimpleNamespace(data=data,find_sites=lambda *args,**kw:([0],['head'])),quality_feet=SimpleNamespace(data=SimpleNamespace(force=force)),quality_other=SimpleNamespace(data=SimpleNamespace(force=torch.zeros(n,1,3))))
scene.env_origins=torch.zeros(n,3)
e=SimpleNamespace(scene=scene,num_envs=n,device='cpu',_r_accum=True,_f_effort_ms=True,_r_head=0,_r_feet=[0,1],common_step_counter=1,_f_cache_step=1,_f_values=torch.zeros(n,4),_f_pose_error=torch.tensor([0.,9.,0.,0.]),cfg=SimpleNamespace(decimation=10))
q=c.cache(e).clone();assert torch.allclose(q,torch.tensor([1.,.775,0.,1.]))
assert torch.isclose(e._c_new_up[3],torch.tensor(math.exp(-6.25)))
data.joint_vel[0]=4.;e.common_step_counter+=1;e._f_cache_step+=1;c.cache(e);assert e._c_quality[0]<q[0]
print('PASS: actual quality callback responds to posture, velocity and bilateral support; low-state idle reward remains unchanged')
