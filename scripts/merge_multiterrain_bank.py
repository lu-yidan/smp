import json,hashlib
from pathlib import Path
import numpy as np
parts=Path('outputs/multiterrain_parts_v2');out=Path('outputs/multiterrain_bank');out.mkdir(exist_ok=True)
manifest={}
for split in ('train','validation'):
 arrays=[np.load(parts/str(i)/f'{split}.npz') for i in range(8)]
 combined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0].files}
 assert set(combined['stratum'])==set(range(8))
 for st in range(8):
  for dr in range(4):
   for source in (0,1):assert ((combined['stratum']==st)&(combined['direction']==dr)&(combined['source']==source)).any()
 np.savez_compressed(out/f'{split}.npz',**combined)
 manifest[split]={'n':len(combined['qpos']),'sha256':hashlib.sha256((out/f'{split}.npz').read_bytes()).hexdigest(),'parts':[json.load(open(parts/str(i)/'manifest.json'))[split] for i in range(8)]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print({k:v['n'] for k,v in manifest.items()})
