"""Check independent generator normalization and unchanged default sampling."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from smp.rl.utils import load_denoiser
from smp.rl.events import _ddpm_sample
p=argparse.ArgumentParser();p.add_argument('--prior-dir',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)
f=load_denoiser(str(a.prior_dir/'pretrained_getup_f2s2.pt'),'cpu')
v=load_denoiser(str(a.prior_dir/'pretrained_getup_lafan6_v6.pt'),'cpu')
def sample(env):
 torch.manual_seed(73)
 with torch.inference_mode():return _ddpm_sample(env,8)
base=SimpleNamespace(device='cpu',_smp_bundle=f)
separate=SimpleNamespace(device='cpu',_smp_bundle=f,_gsi_bundle=v)
direct=SimpleNamespace(device='cpu',_smp_bundle=v)
assert torch.equal(sample(separate),sample(direct))
assert separate._smp_bundle is f
legacy=sample(base);base._gsi_bundle=f
assert torch.equal(legacy,sample(base))
assert f[-2:]==v[-2:]==(59,10)
print('PASS: V6 own normalization, f2s2 scoring preserved, legacy generation unchanged')
