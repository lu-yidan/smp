"""Screen existing natural middle resets against the frozen deployment model."""
from pathlib import Path
import numpy as np,mujoco,json,hashlib
from build_table_specialist_bank import model
out=Path('outputs/v34_mix_middle');out.mkdir(parents=True,exist_ok=True);m=model();d=mujoco.MjData(m);report={}
for split in ['train','validation']:
 b=np.load(f'datasets/reset_banks/natural_curriculum_v1/{split}.npz');pool=np.flatnonzero(b['stages']=='middle');keep=[]
 for i in pool:
  mujoco.mj_resetData(m,d);d.qpos[:]=b['qpos'][i];mujoco.mj_forward(m,d)
  if all(c.dist>=-.001 for c in d.contact):keep.append(i)
 assert len(keep)>20,(split,len(keep));np.savez_compressed(out/(split+'.npz'),qpos=b['qpos'][keep],source_rows=keep)
 report[split]=dict(candidates=len(pool),kept=len(keep),sha256=hashlib.sha256((out/(split+'.npz')).read_bytes()).hexdigest());print(split,report[split],flush=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2))
