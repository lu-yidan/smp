"""Only declared post-clear rewards differ; useful recovery motion is unpenalized."""
import unittest
from dataclasses import asdict
import torch
from test_a6_paper_config import differences
from train_a6_clear_settle import build_config
from train_a6_alpha_convergence import build_config as historical
from smp.rl.tasks.getup.a6_clear_settle import settle_cost

class Contracts(unittest.TestCase):
    def test_factorial_and_historical_baselines(self):
        for seed in (20261026, 20261027):
            for label, oldarm in [('a020','AC1_a020'),('a050','AC2_a050'),('a100','AC3_a100')]:
                base, agent = build_config(4096, 'CS_'+label+'_base', seed=seed)
                old, old_agent = historical(4096, oldarm, seed=seed)
                self.assertEqual(differences(asdict(base),asdict(old)),set())
                self.assertEqual(asdict(agent),asdict(old_agent))
                fixed, fixed_agent = build_config(4096,'CS_'+label+'_settle',seed=seed)
                self.assertEqual(asdict(agent),asdict(fixed_agent))
                self.assertEqual(differences(asdict(base),asdict(fixed)),{
                    '/rewards/plate_separation/func','/rewards/plate_clearance/func',
                    '/rewards/post_clear_anchor'})
                self.assertTrue(fixed.observations['actor'].enable_corruption)
                self.assertIn('push_robot',fixed.events)
                self.assertNotIn('smp_too_low',fixed.terminations)

    def test_cost_scope_and_gradation(self):
        d=torch.tensor([.1,.35,.60,.85,1.10,1.35,2.0])
        z=torch.full_like(d,1.2);u=torch.ones_like(d);valid=torch.ones_like(d,dtype=torch.bool)
        cost=settle_cost(d,z,u,valid)
        self.assertTrue(torch.allclose(cost,torch.tensor([0.,0.,.25,1.,2.25,4.,4.]),atol=1e-5))
        self.assertTrue(torch.equal(settle_cost(d,torch.full_like(d,.9),u,valid),torch.zeros_like(d)))
        self.assertTrue(torch.equal(settle_cost(d,z,torch.full_like(d,.7),valid),torch.zeros_like(d)))
        self.assertTrue(torch.equal(settle_cost(d,z,u,~valid),torch.zeros_like(d)))
        intermediate=settle_cost(d,torch.full_like(d,1.05),torch.full_like(d,.865),valid)
        self.assertTrue(torch.allclose(intermediate,cost*.25,atol=1e-5))

if __name__=='__main__':unittest.main()
