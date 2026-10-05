"""Executable regression checks for the reward loopholes and paired contract."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import numpy as np
import torch
from mjlab.utils.os import dump_yaml
from smp.rl.tasks.getup.guidance_strengthening import auxiliary_step, scene_layout
from train_smp_guidance_strengthening import build_config


def main():
    for n in (32, 64, 4096):
        s, d = scene_layout(n)
        assert len(s) == n
        for k in range(3): assert np.ptp(np.bincount(d[s==k],minlength=4)) == 0
    assert np.bincount(scene_layout(4096)[0]).tolist() == [1228, 2048, 820]
    # Eligible and inactive worlds; revisiting previous lateral distance earns 0.
    timer=torch.tensor([1.49,1.49,1.49,1.49])
    zeros=torch.zeros(4);ones=torch.ones(4)
    args=[timer,ones,zeros,zeros,torch.tensor([.10,.10,.10,.10]),
          ones,zeros,zeros,torch.tensor([.05,.125,.125,.125]),
          torch.tensor([True,True,False,True]),torch.tensor([True,True,True,False]),.02]
    out=auxiliary_step(*args)
    assert torch.allclose(out[-1],torch.tensor([0.,1.,0.,0.]),atol=1e-6)
    assert torch.allclose(out[-2],torch.tensor([1.,1.,0.,1.]))
    args[5]=ones-.002
    out=auxiliary_step(*args)
    assert (out[0]==0).all() and (out[-2]==0).all()
    # Global configs match once exactly the three named weights are restored.
    off, a=build_config(64,'off',20261006);on,b=build_config(64,'on',20261006)
    changed=[k for k in off.rewards if off.rewards[k].weight != on.rewards[k].weight]
    assert set(changed)=={'plate_geometry_progress','plate_clearance','plate_separation'}
    for k in changed:off.rewards[k].weight=on.rewards[k].weight
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        for name,obj in [('off',off),('on',on),('a',a),('b',b)]:dump_yaml(p/(name+'.yaml'),asdict(obj))
        assert (p/'off.yaml').read_bytes()==(p/'on.yaml').read_bytes()
        assert (p/'a.yaml').read_bytes()==(p/'b.yaml').read_bytes()
    print('PASS: balanced quotas, no oscillation credit, stopped/invalid gates, meaningful progress, paired config')


if __name__=='__main__':main()
