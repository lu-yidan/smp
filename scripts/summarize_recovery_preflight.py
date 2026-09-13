"""Verify the five completed short runs and collect launch diagnostics."""
import argparse,json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result={}
for arm in ['L3','L4','L5','L4-Q1','L4-Q2']:
 root=(a.quality if '-Q' in arm else a.controls)/arm
 assert not (root/'failed.json').exists(),str(root/'failed.json')
 done=json.loads((root/'completed.json').read_text());assert done['iteration']==10003
 launch=json.loads((root/'launch.json').read_text())
 init=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
 end=torch.load(root/'model_10003.pt',map_location='cpu',weights_only=False)
 for key in ['mean','count']:
  assert torch.equal(init['infos']['recovery_continuation'][key],end['infos']['recovery_continuation'][key])
 assert end['infos']['recovery_continuation']['learning_rate']==end['optimizer_state_dict']['param_groups'][0]['lr']
 tb=EventAccumulator(str(root)).Reload()
 costs={k:[{'iteration':x.step,'value':x.value} for x in tb.Scalars(k)] for k in tb.Tags()['scalars'] if k.startswith('Quality/') or k.startswith('LowGuidance/low/')}
 result[arm]={'root':str(root),'launch':launch,'natural':json.loads((root/'progress.json').read_text())['validation'],'procedural':json.loads((root/'procedural_progress.json').read_text())['validation'],'loads':json.loads((root/'load_summary.json').read_text()),'costs':costs}
for arm in ['L4-Q1','L4-Q2']:
 for key in ['source_sha256','reference_sha256','initial_qpos_sha','restored_lr']:
  assert result[arm]['launch'][key]==result['L4']['launch'][key],(arm,key)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
for arm,d in result.items():
 print(arm,'natural_low_10s',d['natural']['low']['stable_10s'],'procedural_10s',sum(x['stable_10s']*x['n'] for x in d['procedural'].values())/512,'peak_speed',d['loads']['peak_dq'])
print('PASS: all five completed; restore state, paired L4 starts and saved SMP references verified')
