"""Flat L4 transfer with paired reset mixtures and upward-speed ablation."""
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import balanced_dynamics as bd
from smp.rl.tasks.getup.natural_curriculum import weights_for,DIRECTIONS
from smp.rl.tasks.getup.natural_low_reset import prime_static_history

GROUPS=('late','middle','supine','prone','left_side_down','right_side_down')
def init(env,arm,evaluation,bank_path):
 bd.initialize(env,bank_path,multiterrain=False)
 if hasattr(env,'_ft_rng'):return
 n=env.num_envs;dev=env.device
 fractions=np.array([0,0,.25,.25,.25,.25] if evaluation else [.4,.4,.05,.05,.05,.05] if arm=='FT_R0' else [.2,.4,1/15,.2,1/15,1/15])
 raw=fractions*n;counts=np.floor(raw).astype(int)
 for i in np.argsort(-(raw-counts))[:n-counts.sum()]:counts[i]+=1
 env._ft_counts=counts.tolist();env._ft_group=torch.repeat_interleave(torch.arange(6,device=dev),torch.tensor(counts,device=dev))
 env._ft_source=torch.zeros(n,dtype=torch.long,device=dev)
 for group in range(2,6):
  ids=torch.where(env._ft_group==group)[0];env._ft_source[ids[:round(len(ids)*.25)]]=1
 env._mt_scene.zero_();env._mt_stratum.zero_();env._mt_direction=(env._ft_group-2).clamp_min(0);env._mt_source=env._ft_source
 env._fixed_group=env._ft_group;env._fixed_source=env._ft_source
 split='validation' if evaluation else 'train'
 nat=np.load(f'datasets/reset_banks/natural_curriculum_v1/{split}.npz');proc=np.load(f'datasets/reset_banks/procedural_low_v1/{split}.npz')
 q=np.concatenate([nat['qpos'],proc['qpos']]);labels=np.concatenate([nat['labels'],proc['labels']]);stages=np.concatenate([nat['stages'],proc['stages']]);sources=np.r_[np.zeros(len(nat['qpos']),int),np.ones(len(proc['qpos']),int)]
 weights=np.r_[weights_for(nat,0),np.ones(len(proc['qpos']))]
 env._ft_bank=torch.tensor(q,device=dev,dtype=torch.float32);env._ft_pools=[]
 for group in range(6):
  for source in (0,1):
   if not ((env._ft_group==group)&(env._ft_source==source)).any():continue
   mask=(stages==GROUPS[group]) if group<2 else ((stages=='low')&(labels==DIRECTIONS[group-2]))
   ids=np.where(mask&(sources==source))[0];assert len(ids)
   env._ft_pools.append((group,source,torch.tensor(ids,device=dev),torch.tensor(weights[ids]/weights[ids].sum(),device=dev,dtype=torch.float32)))
 env._ft_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+904193)
 env._ft_relaxed=arm=='FT_R2'

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='FT_R0',evaluation=False,bank_path='outputs/multiterrain_bank/train.npz',**kwargs):
 init(env,arm,evaluation,bank_path)
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 bd.reset(env,ids,bank_path=bank_path,evaluation=evaluation,multiterrain=False,**kwargs)
 rows=torch.empty(len(ids),dtype=torch.long,device=env.device)
 for group,source,pool,weights in env._ft_pools:
  mask=(env._ft_group[ids]==group)&(env._ft_source[ids]==source);n=int(mask.sum())
  if n:rows[mask]=pool[torch.multinomial(weights,n,replacement=True,generator=env._ft_rng)]
 q=env._ft_bank[rows];r=env.scene['robot'];state=r.data.default_root_state[ids].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[ids];state[:,3:7]=q[:,3:7];state[:,7:]=0
 r.write_root_state_to_sim(state,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids)
 env.sim.forward();prime_static_history(env,ids)
