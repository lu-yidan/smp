"""Gate formal runs on every arm's real PPO/evaluation and physical replay integration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from train_prior_replay_transfer import SOURCE_SHA
from smp.rl.tasks.getup.prior_replay_transfer import ARMS,LOW_ARMS,REPLAY_ARMS
from train_fixed_low_ablation import atomic_json
p=argparse.ArgumentParser();p.add_argument('--launches',required=True,type=Path);a=p.parse_args()
rows=json.loads(a.launches.read_text());assert {r['arm'] for r in rows}==set(ARMS)
assert json.loads(Path('outputs/prior_replay_integration.json').read_text())['passed']
initial={};results={}
for row in rows:
 root=Path(row['log_dir']);arm=row['arm']
 assert (root/'completed.json').exists(),str(root)
 assert not (root/'failed.json').exists(),str(root)
 meta=json.loads((root/'launch.json').read_text())
 assert meta['source_sha256']==SOURCE_SHA and meta['episode_s']==10 and meta['num_envs']==4096
 assert meta['actor_noise'] and 'push_robot' in meta['events'] and meta['source_actor_exact']
 assert meta['all_low']==(arm in LOW_ARMS) and meta['failure_replay']==(arm in REPLAY_ARMS)
 assert ('smp_too_low' not in meta['terminations'])==(arm in ('P3_no_smp_term','P6_combined'))
 assert meta['reward_ws']==(4 if arm in ('P2_v7_ws4','P6_combined') else 6)
 x=np.load(root/'initial_reset.npz');initial[arm]=x['qpos']
 assert np.isfinite(x['qpos']).all() and abs(x['qvel']).max()<1e-6
 assert x['active'].mean()==.5
 if arm in LOW_ARMS:
  assert x['group'].min()>=2 and np.bincount(x['group'][:2048]-2).tolist()==[512]*4
  assert x['source'][:2048].sum()==512
 evaluation=json.loads((root/'validation/7/summary.json').read_text())
 assert set(evaluation)=={'flat','prone','plate_easy','plate_hard'}
 for case,v in evaluation.items():
  assert v['protocol_version']==2 and v['initial_penetration_max_m']<=.0011
  assert all(np.isfinite(t) for t in v.values())
  assert (root/f'validation/7/{case}_transitions.json').exists()
 results[arm]={'passed':True,'launch':meta,'evaluation':evaluation}
for arm in ARMS[1:5]:assert np.array_equal(initial['P0_control'],initial[arm]),arm
assert np.array_equal(initial['P5_all_low'],initial['P6_combined'])
files=list(Path('src/smp/rl/tasks/getup').glob('*.py'))+list(Path('scripts').glob('*prior_replay*.py'))+[Path('scripts/train_plate_transfer.py'),Path('scripts/train_v33_reward_transfer.py'),Path('scripts/calibrate_v7_reference.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assets=['outputs/v7_reference.pt','datasets/pretrain_ckpt/pretrained_getup_f2s2.pt','datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt']
hashes.update({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in assets})
atomic_json('outputs/prior_replay_preflight_verified.json',{'passed':True,'source_sha256':SOURCE_SHA,'code_hashes':hashes,'matched_resets':True,'results':results,'launches':str(a.launches)})
print('ALL SEVEN PREFLIGHTS VERIFIED')
