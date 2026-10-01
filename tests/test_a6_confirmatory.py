"""Only declared weights differ; seeds do not change reset/DR contracts."""
import unittest
from dataclasses import asdict
from test_a6_paper_config import differences
from train_a6_confirmatory import build_config, ap
from train_a6_alpha_convergence import build_config as ac_config


class Contracts(unittest.TestCase):
    def test_isolated_interventions(self):
        for seed in (20261021, 20261022, 20261023):
            base, agent = build_config(4096, 'C20_full', seed=seed)
            old, oldagent = ac_config(4096, 'AC3_a100', seed=seed)
            self.assertEqual(differences(asdict(base), asdict(old)), set())
            self.assertEqual(asdict(agent), asdict(oldagent))
            for arm in ap.ARMS:
                cfg, other = build_config(4096, arm, seed=seed)
                changed = {'C20_full': set(),
                           'C20_no_geometry': {f'/rewards/{n}/weight' for n in ap.GEOMETRY_TERMS},
                           'C20_no_Q': {'/rewards/path_quiet_feet/weight'},
                           'C20_no_L': {'/rewards/path_joint_stall/weight'}}[arm]
                self.assertEqual(differences(asdict(base), asdict(cfg)), changed)
                self.assertEqual(asdict(agent), asdict(other))
                self.assertEqual(cfg.rewards['task_smp_product'].params['blocked_alpha'], 1.)
                self.assertTrue(cfg.observations['actor'].enable_corruption)
                self.assertIn('push_robot', cfg.events)
                self.assertNotIn('smp_too_low', cfg.terminations)
                self.assertEqual(cfg.episode_length_s, 10.)
                self.assertEqual(cfg.scene.num_envs, 4096)


if __name__ == '__main__':
    unittest.main()
