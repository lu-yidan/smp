"""Guard against accidentally changing physics/reset in an A6 reward ablation."""
import unittest
from dataclasses import asdict
import inspect

from train_a6_paper_ablation import build_config, ap, legacy


def differences(a, b, path=''):
  if isinstance(a, dict) and isinstance(b, dict):
    return set().union(*(differences(a.get(k), b.get(k), path+'/'+str(k)) for k in a.keys() | b.keys()))
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
    if len(a) != len(b): return {path}
    return set().union(*(differences(x, y, path+'/'+str(i)) for i, (x, y) in enumerate(zip(a, b))))
  if path == '/scene/terrain/spec_fn':
    # EntityCfg creates a fresh no-capture empty-MjSpec lambda per config.
    assert a.__closure__ is None and b.__closure__ is None
    assert inspect.getsource(a) == inspect.getsource(b)
    return set()
  return set() if a == b else {path}


class ConfigContract(unittest.TestCase):
  def test_full_matches_frozen_a6(self):
    original, agent = legacy.build_config(4096, 'outputs/multiterrain_bank/train.npz', False, 'A6')
    actual, other_agent = build_config(4096, 'A6_full', seed=original.seed)
    self.assertEqual(differences(asdict(original), asdict(actual)), set())
    self.assertEqual(asdict(agent), asdict(other_agent))

  def test_only_named_interventions(self):
    base, _ = build_config(4096, 'A6_full')
    geometry = {'/rewards/'+key+'/weight' for key in ap.GEOMETRY_TERMS}
    alpha = {'/rewards/task_smp_product/func'}
    expected = {'A6_full':set(), 'A6_no_bundle':geometry|alpha,
        'A6_no_geometry':geometry, 'A6_no_alpha':alpha,
        'A6_no_hand':{'/events/gsi_reset/params/arm'},
        'A6_no_Q':{'/rewards/path_quiet_feet/weight'},
        'A6_no_L':{'/rewards/path_joint_stall/weight'}}
    for arm in ap.ARMS:
      with self.subTest(arm=arm):
        cfg, _ = build_config(4096, arm)
        self.assertEqual(differences(asdict(base), asdict(cfg)), expected[arm])


if __name__ == '__main__': unittest.main()
