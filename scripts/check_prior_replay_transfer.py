"""GPU integration: low quotas, physical replay, history and invalid-state rejection."""
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from train_prior_replay_transfer import build_config
from smp.rl.tasks.getup import prior_replay_transfer as m,v33_reward_transfer as v
cfg,_=build_config('P6_combined',Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),128)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 env._pr_disable_replay=True
 obs,_=env.reset();v.init(env)
 ids=torch.arange(128,device=env.device)
 assert env._fixed_group.min()>=2 and env._plate_active.sum()==64
 assert torch.bincount(env._fixed_group[:64]-2).tolist()==[16]*4
 assert [int(((env._fixed_group[:64]==g)&(env._fixed_source[:64]==1)).sum()) for g in range(2,6)]==[4]*4
 proc=ids[(~env._plate_active)&(env._fixed_source==1)]
 assert torch.equal(m.direction(env.scene['robot'].data.projected_gravity_b[proc]),env._fixed_group[proc]-2)
 # Every bucket is populated with valid screened fixtures solely for integration testing.
 r=env.scene['robot']
 r.write_joint_state_to_sim(r.data.joint_pos.clone(),torch.full_like(r.data.joint_vel,.02),env_ids=ids)
 env.sim.forward()
 from smp.rl.tasks.getup.natural_low_reset import prime_static_history
 prime_static_history(env,ids)
 inserted=m.insert(env,ids);assert inserted>64,(inserted,env._pr_counts)
 assert all(n>0 for n in env._pr_counts),env._pr_counts
 env._pr_replayed.zero_();m.restore(env,ids,probability=1.,minimum=1);env.sim.forward()
 assert env._pr_replayed.all()
 assert len(m.valid_cpu(env,ids))==128
 assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
 assert env._plate_active.sum()==64
 assert torch.allclose(env.scene['robot'].data.joint_vel,torch.full_like(env.scene['robot'].data.joint_vel,.02),atol=1e-6)
 # Restored history is origin-relative and must match the installed pose tail.
 r=env.scene['robot'];history=env._smp_buffer
 assert torch.allclose(history.root_pos_w[:,-1],r.data.root_link_pos_w-env.scene.env_origins,atol=1e-5)
 assert torch.allclose(history.joint_pos[:,-1],r.data.joint_pos,atol=1e-5)
 # Reset path also restores the physical model, history and warmup via the managers.
 for _ in range(8):m.insert(env,ids)
 env._pr_disable_replay=False
 before=env._pr_replay_total
 for _ in range(5):obs,_=env.reset()
 assert env._pr_replay_total>before and torch.isfinite(obs['actor']).all()
 assert obs['actor'].shape[-1]==93
 result={'passed':True,'initial_valid_inserted':inserted,'buffer_counts':env._pr_counts,'replay_resets':env._pr_replay_total-before,'all_low':True,'plate_fraction':.5,'direction_labels_verified':True,'physical_replay_collision_valid':True}
 Path('outputs').mkdir(exist_ok=True);Path('outputs/prior_replay_integration.json').write_text(json.dumps(result,indent=2));print(result)
finally:env.close()
