"""Create a physically checked, clip-split, right-emphasized natural reset bank."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from train_termination_ablation import build_config
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--source", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  a = p.parse_args()
  b = np.load(a.source, allow_pickle=False)
  indices = np.flatnonzero(b["stages"] == "low")
  groups = np.array([s.replace("__mirror", "") for s in b["clips"]])
  test = np.array([int(hashlib.sha256(("reset-v1:" + s).encode()).hexdigest()[:8], 16) % 5 == 0 for s in groups])
  cfg, _ = build_config("B1", num_envs=1)
  model = Scene(cfg.scene, device="cpu").compile()
  data = mujoco.MjData(model)
  joints = [int(model.joint("robot/" + n).qposadr[0]) for n in JOINT_NAMES]
  limits = np.array([model.joint("robot/" + n).range for n in JOINT_NAMES])
  minimum = 0.0
  for index in indices:
    q = b["qpos"][index]
    assert q[2] < 0.3 and np.isfinite(q).all()
    assert ((q[7:] >= limits[:, 0]) & (q[7:] <= limits[:, 1])).all()
    data.qpos[:7] = q[:7]
    data.qpos[joints] = q[7:]
    mujoco.mj_forward(model, data)
    dist = min((float(c.dist) for c in data.contact), default=0.0)
    minimum = min(minimum, dist)
    assert dist >= -0.002, (index, dist)
  mass = {"supine": 0.1, "prone": 0.2, "left_side_down": 0.2, "right_side_down": 0.5}
  a.output.parent.mkdir(parents=True, exist_ok=True)
  report = dict(source=str(a.source), source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                direction_probability=mass, min_contact_distance=minimum, velocities="zero; repeated current-state SMP history",
                split="sha256('reset-v1:'+clip_without_mirror) mod 5 == 0 is held out; mirror pairs stay together", partitions={})
  for name, ids in (("train", indices[~test[indices]]), ("heldout", indices[test[indices]])):
    weights = np.zeros(len(ids))
    for label, prob in mass.items():
      rows = np.flatnonzero(b["labels"][ids] == label)
      clips = sorted(set(groups[ids[rows]]))
      assert clips
      for clip in clips:
        subset = rows[groups[ids[rows]] == clip]
        weights[subset] = prob / len(clips) / len(subset)
    path = a.output if name == "train" else a.output.with_name(a.output.stem + "_heldout.npz")
    np.savez_compressed(path, qpos=b["qpos"][ids], labels=b["labels"][ids], clips=b["clips"][ids],
                        origins=b["origins"][ids], sampling_weights=weights, source_indices=ids)
    report["partitions"][name] = dict(n=len(ids), directions=dict(Counter(b["labels"][ids].tolist())),
                                      groups=len(set(groups[ids])), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  assert not set(groups[indices[test[indices]]]) & set(groups[indices[~test[indices]]])
  a.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
