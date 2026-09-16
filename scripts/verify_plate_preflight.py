"""Require matched reset, validated contact geometry and complete PPO/eval smoke."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from train_plate_transfer import SOURCE_SHA
from smp.rl.tasks.getup.plate_transfer import ARMS
from train_fixed_low_ablation import atomic_json
p=argparse.ArgumentParser();p.add_argument('--launches',type=Path,required=True);a=p.parse_args();rows=json.loads(a.launches.read_text());assert {r['arm'] for r in rows}==set(ARMS)
initial={};results={}
for row in rows:
 root=Path(row['log_dir']);assert (root/'completed.json').exists(),str(root)
 assert not (root/'failed.json').exists(),str(root)
 meta=json.loads((root/'launch.json').read_text());assert meta['num_envs']==4096 and meta['source_sha256']==SOURCE_SHA
 x=np.load(root/'initial_reset.npz');initial[row['arm']]=x['qpos'][:,:36]
 assert np.isfinite(x['qpos']).all() and np.max(abs(x['qvel']))<1e-6
 expected=.5 if row['arm'] in ('E2_plate','E3_guided') else 0.
 assert float(x['active'].mean())==expected
 report=json.loads((root/'validation/3/summary.json').read_text());assert set(report)=={'flat','prone','plate_easy','plate_hard'}
 for case,val in report.items():
  assert val['initial_penetration_max_m']<=.0011,(case,val)
  assert all(np.isfinite(v) for v in val.values())
 results[row['arm']]={'completed':True,'physical_plate_fraction':expected,'evaluation':report}
for arm in ('E2_plate','E3_guided'):assert np.array_equal(initial['E1_prone'],initial[arm]),arm
files=list(Path('src/smp/rl/tasks/getup').glob('plate_*.py'))+[Path('scripts/train_plate_transfer.py'),Path('scripts/evaluate_plate_transfer.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
atomic_json('outputs/plate_preflight_verified.json',{'passed':True,'source_sha256':SOURCE_SHA,'code_hashes':hashes,'matched_E1_E2_E3_initial_qpos_exact':True,'results':results,'launches':str(a.launches)})
print('ALL FOUR 4096-ENV PREFLIGHTS AND MATCHED RESET CHECK PASSED')
