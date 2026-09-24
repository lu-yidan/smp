"""Summarize frozen final / preselected checkpoints without heldout selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
 old=json.loads((a.previous/'run/protocol.json').read_text());new=json.loads((a.root/'run/protocol.json').read_text())
 for key in ['development_index_sha256','seconds','physics_dt','control_dt','warmup_steps','entry_hold_s','controller','robot_dynamics','push','observation_noise','rollout_code_sha256']:
  assert old[key]==new[key],(key,old[key],new[key])
 sources=[a.previous,a.root];allrows={};summaries={}
 for root in sources:
  summaries.update(json.loads((root/'review/summary.json').read_text()))
  for policy in json.loads((root/'run/protocol.json').read_text())['policies']:
   allrows[policy]={p.stem:json.loads(p.read_text()) for p in (root/'review'/policy).glob('*.json') if not p.stem.startswith('flat_calibration')}
 case_ids=set(next(iter(allrows.values())))
 for rows in allrows.values():assert set(rows)==case_ids
 pairs={}
 for left,right in [('A6_9999','M2_2500'),('A6_9999','M2_8000'),('M2_2500','M2_8000'),('M2_2500','M2_9999'),('A6_9999','M2_9999'),('M1_9999','M2_9999'),('M2_8000','M2_9999'),('M0_9999','M1_9999'),('M1_9000','M2_8000')]:
  if left not in allrows or right not in allrows:continue
  sameclear=[c for c in sorted(case_ids) if allrows[left][c]['first_direct_lift_clear_s'] is not None and allrows[right][c]['first_direct_lift_clear_s'] is not None]
  samequiet=[c for c in sorted(case_ids) if allrows[left][c]['quiet_first_1s_s'] is not None and allrows[right][c]['quiet_first_1s_s'] is not None]
  r={'n':len(case_ids),'quiet_10s_gained':[],'quiet_10s_lost':[],'strict_10s_gained':[],'strict_10s_lost':[],'paired_clear_n':len(sameclear),'paired_quiet_n':len(samequiet)}
  for field,prefix in [('quiet_stable_10s','quiet_10s'),('success_10s','strict_10s')]:
   for c in sorted(case_ids):
    before=allrows[left][c][field];after=allrows[right][c][field]
    if after and not before:r[prefix+'_gained'].append(c)
    if before and not after:r[prefix+'_lost'].append(c)
  for field,ids in [('post_clear_path_m',sameclear),('post_clear_max_drift_m',sameclear),('post_quiet_path_m',samequiet)]:
   r[field]={p:float(np.mean([allrows[p][c][field] for c in ids])) if ids else None for p in [left,right]}
  pairs[left+' -> '+right]=r
 history=json.loads((a.root/'training_validation_history.json').read_text());fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
 selection={}
 for arm,color in [('M0_R2_mix','#3478b6'),('M1_R2_mix','#ce7b22'),('M2_A6_mix','#329569')]:
  vals=history[arm]['validation'];xs=sorted(map(int,vals));ys=[100*np.mean([x['stable_10s'] for x in vals[str(k)].values()]) for k in xs]
  selected=max(zip(ys,xs));selection[arm]={'selected_iteration':selected[1],'selected_macro_pct':selected[0],'final_macro_pct':ys[-1]}
  ax.plot(xs,ys,label=arm,color=color,linewidth=1.6);ax.scatter([selected[1]],[selected[0]],marker='*',s=130,color=color,zorder=3)
 ax.set(xlabel='Additional PPO updates',ylabel='Stable 10s: macro average (%)',title='Original 10-scene validation (nominal dynamics)',ylim=(0,100));ax.grid(alpha=.2);ax.legend(loc='lower right')
 fig.savefig(a.root/'training_validation_curve.png',dpi=170);plt.close(fig)
 allresult={'scope':'48 paired clutter development cases per policy; four flat calibration cases separately; no heldout evaluation',
            'protocol_compatible_with_previous':True,'selection':selection,'summaries':summaries,'paired_comparisons':pairs}
 (a.root/'comparison.json').write_text(json.dumps(allresult,indent=2)+'\n')
 (a.root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 # Review provenance without rewriting the immutable original rollout protocol.
 (a.root/'review/protocol.json').write_text(json.dumps({'source_protocol_sha256':digest(a.root/'run/protocol.json'),'reviewer_sha256':digest(Path(__file__).with_name('review_clutter_metrics.py')),'summarizer_sha256':digest(__file__),'physics_rerun':False,'strict_hold_match_tolerance_s':.04},indent=2)+'\n')
 for policy,x in summaries.items():
  d=x['all_clutter'];print(policy,'Q/E',d['quiet_stable_10s'],d['stable_10s'],'tau/speed/power P95',*[round(d[k],2) for k in ['peak_tau_nm_trial_max_p95','peak_dq_rad_s_trial_max_p95','peak_power_w_trial_max_p95']],'load90s',round(d['above_90pct_torque_s_trial_max_p95'],2),'flatQ/E',x['families']['flat_calibration']['quiet_stable_10s'],x['families']['flat_calibration']['stable_10s'])

if __name__=='__main__':main()
