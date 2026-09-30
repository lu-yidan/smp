"""Reward semantics, intervention isolation, and episode-local anchor lifecycle."""
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import torch
from test_a6_paper_config import differences
from train_a6_alpha_convergence import build_config, ap
from train_a6_paper_ablation import build_config as frozen_config


class Contracts(unittest.TestCase):
  def test_configs(self):
    base, agent = build_config(4096, 'AC0_a005')
    old, oldagent = frozen_config(4096, 'A6_full')
    self.assertEqual(asdict(agent), asdict(oldagent))
    self.assertEqual(differences(asdict(old), asdict(base)), {
      '/events/gsi_reset/func', '/metrics/ac_anchor_cost',
      '/rewards/task_smp_product/func', '/rewards/task_smp_product/params/blocked_alpha'})
    for arm in ap.ARMS:
      cfg, otheragent = build_config(4096, arm)
      expected = set()
      if arm != 'AC0_a005': expected.add('/rewards/task_smp_product/params/blocked_alpha')
      if ap.settings(arm)['close']:
        expected |= {'/rewards/plate_separation/func', '/rewards/plate_clearance/func'}
      if ap.settings(arm)['anchor']: expected.add('/rewards/post_clear_anchor')
      self.assertEqual(differences(asdict(base), asdict(cfg)), expected, arm)
      self.assertEqual(asdict(agent), asdict(otheragent))

  def test_alpha_only_obstructed(self):
    env = SimpleNamespace(_mt_scene=torch.tensor([0,1,1,2]),
                          _mt_escaped=torch.tensor([False,False,True,False]))
    raw = torch.tensor([1.,2.,3.,4.])
    for alpha in (.05,.2,.5,1.):
      with patch.object(ap.mt, 'recorded_task_smp_product', return_value=raw):
        value = ap.scaled_task(env, [], blocked_alpha=alpha)
      self.assertTrue(torch.equal(value, raw*torch.tensor([1.,alpha,1.,alpha])))

  def test_closure(self):
    env = SimpleNamespace(_mt_scene=torch.tensor([0,1,1,2]),
      _mt_escaped=torch.tensor([False,False,True,True]),
      _mt_invalid=torch.tensor([False,False,False,True]))
    with patch.object(ap.mt, 'escape_reward', return_value=torch.ones(4)):
      self.assertTrue(torch.equal(ap.closed_separation(env), torch.tensor([1.,1.,0.,0.])))
    with patch.object(ap.pa, 'reward', return_value=torch.tensor([0.,.5,.9,0.])):
      self.assertTrue(torch.equal(ap.saturated_clearance(env), torch.tensor([0.,.5,1.,0.])))

  def test_free_disk_and_saturation(self):
    xy = torch.tensor([[0.,0.],[.25,0.],[.375,0.],[.5,0.],[10.,0.]])
    value = ap.position_cost(xy, torch.zeros_like(xy), torch.ones(5), torch.ones(5,dtype=torch.bool))
    self.assertTrue(torch.allclose(value, torch.tensor([0.,0.,.25,1.,1.])))
    self.assertEqual(ap.position_cost(xy,torch.zeros_like(xy),torch.zeros(5),torch.ones(5,dtype=torch.bool)).sum(),0)

  def test_anchor_latches_once_pauses_on_reblock(self):
    n=3
    data=SimpleNamespace(root_link_pos_w=torch.zeros(n,3), projected_gravity_b=torch.tensor([[0.,0.,-1.]]*n))
    env=SimpleNamespace(scene={'robot':SimpleNamespace(data=data)}, common_step_counter=1, step_dt=.02,
      _mt_scene=torch.tensor([0,1,2]), _mt_escaped=torch.tensor([False,True,False]),
      _mt_invalid=torch.zeros(n,dtype=torch.bool), _ac_anchor=torch.zeros(n,2),
      _ac_anchor_set=torch.zeros(n,dtype=torch.bool), _ac_ready_time=torch.zeros(n),
      _ac_cost=torch.zeros(n), _ac_tick=-1)
    forces=torch.zeros(n,2,3); forces[:,:,2]=30.
    with patch.object(ap.mt,'height',return_value=torch.full((n,),1.2)), patch.object(ap.mt,'ground_force',return_value=forces):
      ap.anchor_cost(env)
      self.assertEqual(env._ac_anchor_set.tolist(),[False,True,False])
      data.root_link_pos_w[1,0]=.5; env.common_step_counter+=1
      self.assertAlmostEqual(float(ap.anchor_cost(env)[1]),1.)
      env._mt_escaped[1]=False; env.common_step_counter+=1
      self.assertEqual(float(ap.anchor_cost(env)[1]),0.)
      env._mt_escaped[1]=True; env.common_step_counter+=1
      self.assertAlmostEqual(float(ap.anchor_cost(env)[1]),1.)
      self.assertEqual(float(env._ac_anchor[1,0]),0.)
      for _ in range(12): env.common_step_counter+=1; ap.anchor_cost(env)
      self.assertTrue(env._ac_anchor_set[0])


if __name__ == '__main__': unittest.main()
