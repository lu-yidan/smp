"""Read-only trajectory reward audit; AUDIT_PRIOR selects the archived entry point."""
import os,json,torch
if os.environ.get('AUDIT_PRIOR')=='v7':
 import train_v34_plate_transfer as run
else:
 import train_v34_reward_repair as run

def evaluate(env,w,policy,obs,out,video=False):
 if os.environ.get('AUDIT_PRIOR')=='f2s2':
  import v34_reward_repair
  v34_reward_repair.load_reference(env)
 group=env._vp_group.clone();r=env.scene['robot'];head=r.find_sites(['head'])[0][0]
 accum={};count={};timeline=[];alive=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
 maxz=torch.zeros(env.num_envs,device=env.device)
 for k in range(1150):
  with torch.inference_mode():obs,rew,done,_=w.step(policy(obs))
  alive &= ~done.bool();z=r.data.site_pos_w[:,head,2]-env.scene.env_origins[:,2];u=-r.data.projected_gravity_b[:,2]
  maxz=torch.maximum(maxz,torch.where(alive,z,0))
  smp=env._smp_score if hasattr(env,'_smp_score') else env._fixed_smp_score
  task=env._smp_task_score if hasattr(env,'_smp_task_score') else env._fixed_task_score
  raw=env._smp_raw_err
  terms=env.reward_manager._step_reward
  values=torch.cat([smp[:,None],task[:,None],raw[:,None],terms],-1)
  for label,mask in [('all',alive),('low',alive&(z<.65)),('upright',alive&(z>=1.15)&(u>=.9))]:
   accum[label]=accum.get(label,torch.zeros(values.shape[1],device=env.device))+values[mask].sum(0);count[label]=count.get(label,0)+int(mask.sum())
  if k%50==49:timeline.append(dict(t=(k+1)*.02,head=float(z.mean()),smp=float(smp.mean()),raw=float(raw.mean()),task=float(task.mean())))
 names=['smp','task','raw_mse']+env.reward_manager.active_terms
 data=dict(prior=os.environ.get('AUDIT_PRIOR'),count=count,phase_means={k:dict(zip(names,(v/max(count[k],1)).tolist())) for k,v in accum.items()},timeline=timeline,max_head_mean=float(maxz.mean()))
 run.write(out/'reward_audit.json',data)
run.evaluate=evaluate
if __name__=='__main__':run.main()
