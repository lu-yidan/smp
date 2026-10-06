"""Check the intended phase/position boundaries before spending training compute."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from smp.rl.tasks.getup import table_specialist as t
from train_table_specialist import config

def main():
 r=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[.55,0.,.2],[.55,0.,.2],[.55,0.,.2],[.15,0.,.2]]),projected_gravity_b=torch.tensor([[1.,0.,0.]]*4)))
 env=SimpleNamespace(scene={'robot':r},_tb_anchor=torch.zeros(4,2),_tb_done=torch.tensor([False,True,True,True]),_tb_escape_age=torch.tensor([1.,.25,1.,1.]))
 with patch.object(t.mt,'height',return_value=torch.full((4,),.3)):
  assert torch.equal(t.post_clear_position(env),torch.zeros(4)) # Low posture: old standing gate is off.
  assert torch.allclose(t.post_clear_position(env,'after_clear'),torch.tensor([0.,.5,1.,0.]),atol=1e-6)
 env._tb_anchor=r.data.root_link_pos_w[:,:2].clone()
 with patch.object(t.mt,'height',return_value=torch.full((4,),1.3)):
  assert not t.post_clear_position(env,'after_clear').any() # New anchor has no cost.
 c={a:config(128,a,False,20261007)[0] for a in ['T1','T2','T3','T4','T5']}
 for a in ['T3','T4','T5']:
  assert set(c[a].rewards)-set(c['T1'].rewards)=={'table_post_clear_position'}
  for k in c['T1'].rewards:assert c[a].rewards[k]==c['T1'].rewards[k],(a,k)
 assert abs(c['T5'].rewards['table_post_clear_position'].weight/c['T4'].rewards['table_post_clear_position'].weight-3)<1e-6
 assert set(c['T2'].rewards)-set(c['T3'].rewards)=={'table_stand_pose','table_stand_feet','table_stand_body'}
 print('DRIFT_CONTRACT_PASS: no pre-clear cost; 25cm free zone; low-posture activation; 0.5s ramp; matched reward interventions')
if __name__=='__main__':main()
