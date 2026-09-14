import argparse,json
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report={}
for i in range(8):
    arm=f'R{i}';root=a.root/arm
    assert not (root/'failed.json').exists(),str(root/'failed.json')
    assert json.loads((root/'completed.json').read_text())['iteration']==3
    meta=json.loads((root/'launch.json').read_text());assert meta['from_scratch'] and not meta['checkpoint_loaded'] and meta['num_envs']==4096 and meta['iterations']==4
    assert meta['quota_counts']==[1638,1642,204,204,204,204]
    assert meta['standing_termination']==(i==0) and meta['quality_ramp_updates']==[5000,10000]
    init=torch.load(root/'initial.pt',weights_only=False,map_location='cpu');end=torch.load(root/'model_3.pt',weights_only=False,map_location='cpu')
    assert not init['optimizer_state_dict']['state'] and end['optimizer_state_dict']['state']
    state=end['infos']['scratch_tradeoffs'];assert state['quality_ramp']==0
    assert torch.isfinite(state['mean']).all() and torch.all(state['mean']>0) and state['count'].sum()>0
    assert state['learning_rate']==end['optimizer_state_dict']['param_groups'][0]['lr']
    reset=np.load(root/'initial_reset.npz');assert np.max(np.abs(reset['qvel']))==0
    loads=json.loads((root/'validation/3_loads.json').read_text());assert loads['physics_samples']==10000
    for key in ['peak_tau','peak_dq','peak_power','speed_exceed_fraction','effort_exceed_fraction']:
        values=np.asarray(loads[key]);assert values.shape==(128,29) and np.isfinite(values).all()
    for key in ['speed_exceed_fraction','effort_exceed_fraction']:assert np.max(loads[key])<=1 and np.min(loads[key])>=0
    report[arm]={'launch':meta,'natural':json.loads((root/'progress.json').read_text())['validation'],'procedural':json.loads((root/'procedural_progress.json').read_text())['validation'],'loads':json.loads((root/'load_progress.json').read_text())['summary']}
for arm in report:
    for key in ['actor_sha','critic_sha','initial_qpos_sha','train_bank_sha','reset_sources']:
        assert report[arm]['launch'][key]==report['R0']['launch'][key],(arm,key)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print('PASS: all eight scratch starts, common networks/poses/reset sources, saved SMP state, finite PPO and 3 evaluation paths')
