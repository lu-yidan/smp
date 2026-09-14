from dataclasses import asdict
from types import SimpleNamespace
import torch
from train_scratch_tradeoffs import build_config,COST_NAMES
from smp.rl.tasks.getup import scratch_tradeoffs as m
from train_fixed_low_ablation import build_config as original
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
r0,a=build_config('R0',bank);old,b=original('L4',bank)
# Original dynamics, observations, reset, algorithm and active rewards preserved.
x,y=asdict(r0),asdict(old)
x['scene']['sensors']=y['scene']['sensors'];x['metrics']=y['metrics']
for k in ['quiet_stand',*COST_NAMES]:assert x['rewards'].pop(k)['weight']==0
x['scene']['terrain']['spec_fn']=y['scene']['terrain']['spec_fn'];assert x==y and asdict(a)==asdict(b)
for i in range(1,8):
 cfg,agent=build_config(f'R{i}',bank);assert cfg.terminations['stood_up'].func==m.no_stand_termination
 assert asdict(agent)==asdict(a) and cfg.episode_length_s==10
 assert cfg.events['gsi_reset'].params==r0.events['gsi_reset'].params
 assert cfg.rewards['quiet_stand'].weight==(0 if i==1 else .1)
 expected=[-.05 if i in (4,7) else 0,-.1 if i in (4,7) else 0,-.005 if i in (5,7) else 0,-.1 if i in (6,7) else 0]
 assert [cfg.rewards[k].weight for k in COST_NAMES]==expected
 fn=cfg.rewards['task_smp_product'].params['task_terms'][0][0]
 assert (fn==m.slow_upward)==(i in (3,7))
for step,w in [(0,0),(120000,0),(180000,.5),(240000,1),(999999,1)]:assert m.ramp(SimpleNamespace(common_step_counter=step))==w
v=m.velocity_band(torch.tensor([-.1,0.,.1,.2,.3,.5,1.]))
assert torch.allclose(v[2:5],torch.ones(3)) and v[-1]<v[-2]<v[3] and v[0]<v[1]<v[2]
limits=torch.ones(29);speeds=torch.ones(29)*20;tau=torch.zeros(2,29);dq=torch.zeros_like(tau);tau[1,0]=1;dq[1,0]=20
e,s=m.load_costs(tau,dq,limits,speeds);assert torch.allclose(e,torch.tensor([0.,1/3]),atol=1e-6) and torch.allclose(s,e)
print('PASS: R0 original control, eight module assignments, fixed ramp, speed band and isolated-joint tail costs')

# Static/light contact is different from a dynamic head impact; reset ramp is explicit.
f=torch.tensor([0.,150.,1000.,1000.,1000.]);vz=torch.tensor([0.,0.,0.,.3,.3]);age=torch.tensor([1.,1.,1.,1.,0.])
assert torch.allclose(m.head_cost(f,vz,age),torch.tensor([0.,0.,.25,1.,0.]))
# Same posture: standing, low, single-foot, and moving. Reward remains graded under noise.
n=4;data=SimpleNamespace(joint_pos=torch.zeros(n,29),joint_vel=torch.zeros(n,29),site_pos_w=torch.zeros(n,1,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(n,1),body_link_pos_w=torch.zeros(n,2,3),body_link_lin_vel_w=torch.zeros(n,2,3),root_link_lin_vel_w=torch.zeros(n,3),root_link_ang_vel_w=torch.zeros(n,3))
data.site_pos_w[:,:,2]=1.2;data.site_pos_w[1,:,2]=.2;data.body_link_pos_w[:,1,0]=.2;data.joint_vel[3]=3.
forces=torch.zeros(n,2,3);forces[:,:,2]=100;forces[2,1,2]=0
scene={'robot':SimpleNamespace(data=data),'quality_feet':SimpleNamespace(data=SimpleNamespace(force=forces)),'quality_other':SimpleNamespace(data=SimpleNamespace(force=torch.zeros(n,1,3)))}
class Scene(dict):pass
scene=Scene(scene);scene.env_origins=torch.zeros(n,3)
target=torch.zeros(n,29);env=SimpleNamespace(scene=scene,num_envs=n,device='cpu',common_step_counter=60000,episode_length_buf=torch.ones(n,dtype=torch.long),step_dt=.02,cfg=SimpleNamespace(decimation=10),action_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(_processed_actions=target)),_r_accum=torch.zeros(n,3),_r_costs=torch.zeros(n,4),_r_prev_target=torch.ones(n,29),_r_cache_step=-1,_r_head=0,_r_feet=[0,1],_r_knees=[0,1],_r_hold=torch.zeros(n))
q=m.cache_control(env)
assert q[0]>.99 and q[1]==0 and q[2]==0 and 0<q[3]<q[0]
assert env._r_costs[:,2].sum()==0 and env._r_stable.tolist()==[True,False,False,False]
env.common_step_counter+=1;env.episode_length_buf+=1;target[:]=.1;m.cache_control(env)
assert torch.allclose(env._r_costs[:,2],torch.ones(n),atol=1e-6)
print('PASS: graded standing gate, strict metrics, reset target exclusion and dynamic head-contact cost')
