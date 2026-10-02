"""Audit and summarize available fixed-10k evaluations; never fill missing seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=('C20_full','C20_no_geometry','C20_no_Q','C20_no_L')
SEEDS=(20261021,20261022,20261023)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(f,mask):
 def quantile(x):return float(np.percentile(x,95)) if len(x) else None
 clear=mask&(f['post_clear_samples']==150);stand=mask&(f['post_stand_samples']==150)
 return {'n':int(mask.sum()),'sr10':float(np.mean(f['best_hold'][mask]>=10-1e-4)),
         'sr1':float(np.mean(f['best_hold'][mask]>=1-1e-4)),
         'cleared':float(np.mean(f['first_clear'][mask]>=0)),
         'slip_m_per_s':float(f['foot_slip'][mask].sum()/f['near_time'][mask].sum()) if f['near_time'][mask].sum()>0 else None,
         'stall_p95_s':quantile(f['stall_time'][mask].max(-1)),
         'power_p95_w':quantile(f['peaks'][mask,:,2].max(-1)),
         'torque_p95_nm':quantile(f['peaks'][mask,:,0].max(-1)),
         'joint_speed_p95_rad_s':quantile(f['peaks'][mask,:,1].max(-1)),
         'post_clear_displacement_p95_m':quantile(f['post_clear_max_offset'][clear]),'post_clear_n':int(clear.sum()),
         'post_stand_displacement_p95_m':quantile(f['post_stand_max_offset'][stand]),'post_stand_n':int(stand.sum()),
         'terminated':int((~f['alive'][mask]).sum())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 initial=None;records=[];hashes={};data={};pending=[]
 for seed in SEEDS:
  for arm in ARMS:
   directory=a.root/f'seed{seed}'/arm/'validation/10000'
   if not all((directory/s/'per_trial.npz').exists() for s in ('nominal','upper130')):
    pending.append({'seed':seed,'arm':arm});continue
   for suite in ('nominal','upper130'):
    path=directory/suite
    for name in ('initial.npz','launch.json','summary.json','per_trial.npz'):hashes[str((path/name).relative_to(a.root))]=sha(path/name)
    f=dict(np.load(path/'per_trial.npz'));init=dict(np.load(path/'initial.npz'));meta=json.loads((path/'launch.json').read_text());original=json.loads((path/'summary.json').read_text())
    if initial is None:initial=init
    assert all(np.array_equal(v,initial[k]) for k,v in init.items()),path
    assert meta['num_envs']==2048 and meta['seed']==20261013 and not meta['actor_noise']
    assert meta['intervention']['blocked_alpha']==1 and len(f['scene'])==2048
    assert np.array_equal(f['scene'],init['scene']) and np.array_equal(f['direction'],init['direction'])
    row={'seed':seed,'arm':arm,'suite':suite,'checkpoint_sha256':meta['checkpoint_sha256'],'all_plates':metrics(f,f['scene']>0)}
    for i,scene in enumerate(('flat','guided_plate','free_plate')):
     mask=f['scene']==i;v=metrics(f,mask);assert abs(v['sr10']-original[scene]['stable_10s'])<1e-6
     assert v['n']==original[scene]['n'];assert abs(v['power_p95_w']-original[scene]['power_peak_p95'])<.05
     row[scene]=v
    prior=a.root/f'seed{seed}'/arm/'validation/5000'/suite/'summary.json'
    if prior.exists():
     old=json.loads(prior.read_text());row['monitor_5k_plate_sr10']=sum(old[s]['stable_10s']*old[s]['n'] for s in ('guided_plate','free_plate'))/sum(old[s]['n'] for s in ('guided_plate','free_plate'))
    records.append(row);data[(seed,arm,suite)]=f
 pairs=[]
 for seed in SEEDS:
  for arm in ARMS[1:]:
   for suite in ('nominal','upper130'):
    left=data.get((seed,'C20_full',suite));right=data.get((seed,arm,suite))
    if left is None or right is None:continue
    mask=left['scene']>0;both=mask&(left['best_hold']>=10-1e-4)&(right['best_hold']>=10-1e-4)
    l=metrics(left,mask);r=metrics(right,mask)
    pairs.append({'seed':seed,'ablated_arm':arm,'suite':suite,'delta_ablated_minus_full_sr10':r['sr10']-l['sr10'],
                  'common_success_n':int(both.sum()),'common_success_full':metrics(left,both) if both.any() else None,
                  'common_success_ablated':metrics(right,both) if both.any() else None})
 result={'scope':'Fixed model_10000.pt; available paired validation only, not final 20k or complete 3-seed experiment',
         'available_models':len(records)//2,'total_rollouts':len(records)*2048,'initial_states_identical_across_all_suites':True,
         'pending':pending,'records':records,'paired':pairs,'input_sha256':hashes}
 (a.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PASS:',len(records)//2,'models;',len(records)*2048,'rollouts;',len(pending),'missing models; all initial states and printed metrics audited')
 for row in records:
  if row['suite']=='nominal':
   v=row['all_plates'];print(row['seed'],row['arm'],'SR10',round(100*v['sr10'],2),'slip',round(v['slip_m_per_s'],5),'stall',round(v['stall_p95_s'],3),'power',round(v['power_p95_w'],1))

if __name__=='__main__':main()
