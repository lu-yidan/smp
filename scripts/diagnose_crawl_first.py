"""Read-only checkpoint diagnosis on matched reset draws; no training updates."""
import train_crawl_first as run
import torch,numpy as np,json,os

def evaluate(env,w,policy,obs,a):
 if os.environ.get('DIAG_ACTION_NOISE'):
  base_policy=policy;std=float(os.environ['DIAG_ACTION_NOISE']);policy=lambda x:base_policy(x)+std*torch.randn((env.num_envs,29),device=env.device)
 live=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
 kind=env._cr_kind.clone();rows=env._tb_rows.clone();maxz=torch.zeros(env.num_envs,device=env.device);maxhold=maxz.clone();hold=maxz.clone();loose=maxz.clone();maxloose=maxz.clone();clear=maxz.clone();upright=maxz.clone();reward_sum=None
 phase_sum={};phase_n={};selected=[];discounted=torch.zeros_like(maxz);actual_return=torch.zeros_like(maxz);smp_sum=torch.zeros_like(maxz);task_sum=torch.zeros_like(maxz);timeline=[]
 for p in (0,1):
  for k in (0,2):
   ids=torch.where((env._tb_pose==p)&(kind==k))[0]
   if len(ids):selected.append(int(ids[0]))
 for step in range(1150):
  with torch.inference_mode():obs,rew,done,extra=w.step(policy(obs))
  live &=~done.bool();r=env.scene['robot'];z=run.mt.height(env);u=-r.data.projected_gravity_b[:,2];feet=run.mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)
  good=(z>=1.15)&(u>=.9)&(feet>20)&live;relaxed=(z>=1.)&(u>=.8)&live
  hold=torch.where(good,hold+.02,0);loose=torch.where(relaxed,loose+.02,0);maxhold=torch.maximum(maxhold,hold);maxloose=torch.maximum(maxloose,loose);maxz=torch.maximum(maxz,z);clear+=env._cr_clear.float();upright+=(z>=1.).float()
  discounted+=(.99**step)*rew;actual_return+=rew;smp_sum+=env._fixed_smp_score;task_sum+=env._fixed_task_score
  if step%50==49:timeline.append(dict(t=(step+1)*.02,head=float(z.mean()),smp=float(env._fixed_smp_score.mean()),task=float(env._fixed_task_score.mean()),reward=float(rew.mean())))
  terms=env.reward_manager._step_reward.clone()
  if reward_sum is None:reward_sum=torch.zeros_like(terms)
  reward_sum+=terms
  for label,mask in [('table',kind!=2),('outside',kind==2),('low',z<.65),('standing',good),('clear_low',env._cr_clear&(z<.65))]:
   phase_sum[label]=phase_sum.get(label,torch.zeros(terms.shape[1],device=env.device))+terms[mask].sum(0)
   phase_n[label]=phase_n.get(label,0)+int(mask.sum())
  if a.video and step in (49,249,499,999):
   from PIL import Image,ImageDraw
   import mujoco
   cam=env._offline_renderer._cam;cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=55;cam.elevation=-8
   for j in selected:
    env.cfg.viewer.env_idx=j;root=r.data.root_link_pos_w[j].cpu().numpy();cam.lookat[:]=root;cam.lookat[2]=.5;cam.distance=2.8
    im=Image.fromarray(env.render());ImageDraw.Draw(im).text((8,8),f'{a.arm} {a.checkpoint.name} env{j} kind{int(kind[j])} t={(step+1)*.02:.1f} z={z[j]:.2f} u={u[j]:.2f}',fill='white');im.save(a.out/f'env{j}_step{step+1}.png')
 names=env.reward_manager.active_terms
 out={'checkpoint':str(a.checkpoint),'timeline':timeline,'group':{},'reward_mean_per_step':{k:dict(zip(names,(v/max(phase_n[k],1)).tolist())) for k,v in phase_sum.items()}}
 for label,mask in [('table',kind!=2),('outside',kind==2)]:
  out['group'][label]={'n':int(mask.sum()),'ever_head1m':float((maxz[mask]>=1.).float().mean()),'ever_head1_15m':float((maxz[mask]>=1.15).float().mean()),'strict3s':float((maxhold[mask]>=2.999).float().mean()),'loose3s_z1_u08':float((maxloose[mask]>=2.999).float().mean()),'max_head_height_median':float(maxz[mask].median()),'mean_return_23s':float(actual_return[mask].mean()),'discounted_return_gamma099':float(discounted[mask].mean()),'mean_smp':float((smp_sum[mask]/1150).mean()),'mean_task':float((task_sum[mask]/1150).mean()),'mean_clear_fraction':float((clear[mask]/1150).mean())}
 run.write(a.out/'diagnosis.json',out);np.savez_compressed(a.out/'diagnosis_trials.npz',kind=kind.cpu(),rows=rows.cpu(),max_height=maxz.cpu(),max_hold=maxhold.cpu(),max_loose=maxloose.cpu(),reward_sum=reward_sum.cpu());print(json.dumps(out),flush=True)

run.evaluate=evaluate
if __name__=='__main__':run.main()
