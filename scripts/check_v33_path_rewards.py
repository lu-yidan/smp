"""Counterexamples for contact-gate and unrelated-joint loopholes."""
import json
from pathlib import Path
import torch
from smp.rl.tasks.getup import v33_path_ablation as p,r2_ablation as r
# Pure vertical movement of board/robot cannot change planar coverage.
pos=torch.tensor([[[0.,0.,.3],[.2,0.,.2]]]);ext=torch.ones_like(pos)*.05;board=torch.tensor([[0.,0.,.5]]);size=torch.tensor([[.45,.32,.035]])
a=p.footprint(pos,ext,board,size);up=board.clone();up[:,2]+=1.;b=p.footprint(pos,ext,up,size)
assert all(torch.equal(x,y) for x,y in zip(a,b))
moved=pos.clone();moved[:,:,0]+=1.;c=p.footprint(moved,ext,board,size)
assert c[0].item()==0 and c[1].item()==0 and c[2].item()>0
# No hand support / high posture / inactive phase earn no path-progress reward.
x=torch.tensor([.025]);d=torch.tensor([0.]);active=torch.tensor([True])
assert p.path_progress(x,d,torch.tensor([.7]),torch.ones(1),active).item()==1
assert p.path_progress(x,d,torch.tensor([.7]),torch.zeros(1),active).item()==0
assert p.path_progress(x,d,torch.tensor([1.]),torch.ones(1),active).item()==0
assert p.quiet_gate(torch.tensor([1.15]),torch.tensor([.93])).item()>.999
# Other joint motion leaves the loaded joint eligible. Smooth own-joint progress.
span=torch.zeros(1,29);before=p.joint_stall_gate(span);span[:,20]=2.;after=p.joint_stall_gate(span)
assert before[0,11]==after[0,11]==1 and after[0,20]==0
# Cost time scale: brief effort is exempt, sustained stationary effort is not.
timer=torch.zeros(1,29);total=0.
for _ in range(150):
 timer,cost=r.stalled_update(timer,torch.ones_like(timer)*.95,torch.zeros_like(timer),.02);total+=cost.max().item()*.02
assert total>2
out={'vertical_lift_no_planar_credit':True,'unsupported_and_high_pose_no_path_credit':True,'quiet_gate_has_no_contact_dependency':True,'unrelated_joint_cannot_clear_load':True,'stationary_3s_raw_integral':total}
Path('outputs').mkdir(exist_ok=True);Path('outputs/path_reward_invariants.json').write_text(json.dumps(out,indent=2));print(out)
