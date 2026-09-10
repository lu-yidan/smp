"""V33 continuation on a frozen RoboMimic deployment dynamics contract."""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
import yaml
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "deploy_g1"
CONTROL = yaml.safe_load((ASSETS / "control.yaml").read_text())
_ROOT = ET.parse(ASSETS / "source.xml").getroot()
JOINT_NAMES = tuple(a.attrib["joint"] for a in _ROOT.find("actuator"))
COLLISION_NAMES = tuple(
  geom.attrib["name"]
  for geom in _ROOT.find(".//body[@name='pelvis']").iter("geom")
  if geom.get("class") == "collision"
)
COLLISION_PATTERN = "(?:" + "|".join(re.escape(name) for name in COLLISION_NAMES) + ")"


def deployment_robot_spec():
  """Keep deployment robot/constraints; scene owns floor and V33 guided plate."""
  root = copy.deepcopy(_ROOT)
  root.find("compiler").set("meshdir", str(ASSETS / "meshes"))
  for tag in ("actuator", "sensor", "contact", "keyframe"):
    for item in root.findall(tag):
      root.remove(item)
  for world in root.findall("worldbody"):
    for child in list(world):
      if child.tag != "body" or child.get("name") != "pelvis":
        world.remove(child)
  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu_in_pelvis")
  ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu_in_pelvis")
  ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu_in_pelvis")
  # Reward marker is unchanged from V33; it is not a collision shape.
  torso = root.find(".//body[@name='torso_link']")
  ET.SubElement(torso, "site", name="head", pos="0 0 0.43", size="0.005")
  return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def deployment_contacts(spec):
  """Restore exact explicit foot-floor pairs after entity namespace attachment."""
  ground = spec.geom("terrain")
  ground.solref = (0.001, 0.8)
  ground.solimp = (0.95, 0.99, 0.005, 0.5, 2)
  ground.friction = (1.0, 0.005, 0.0001)
  for pair in _ROOT.find("contact").findall("pair"):
    spec.add_pair(
      name=pair.get("name"),
      geomname1="robot/" + pair.get("geom1"),
      geomname2="terrain",
      condim=3,
      solref=(0.01, 1.0),
      friction=(1.0, 1.0, 0.005, 0.0001, 0.0001),
    )


def zero_velocity(env):
  return torch.zeros((env.num_envs, 3), device=env.device)


def previous_clipped_action(env):
  return env.action_manager.get_term("joint_pos").raw_action


class DeploymentPositionAction(JointPositionAction):
  """Match deployment raw clipping, entry blend and policy-rate torque projection."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._entry = self._processed_actions.clone()
    self._age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    lookup = {name: i for i, name in enumerate(JOINT_NAMES)}
    order = [lookup[name] for name in self._target_names]
    self._kp = torch.tensor(CONTROL["kps"], device=self.device)[order]
    self._kd = torch.tensor(CONTROL["kds"], device=self.device)[order]
    self._limit = torch.tensor(CONTROL["tau_limit"], device=self.device)[order]

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    super().reset(ids)
    self._age[ids] = 0
    self._entry[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]

  def process_actions(self, actions):
    super().process_actions(
      actions.clamp(-CONTROL["clip_actions"], CONTROL["clip_actions"])
    )
    q = self._entity.data.joint_pos[:, self._target_ids]
    dq = self._entity.data.joint_vel[:, self._target_ids]
    alpha = ((self._age + 1).float() / CONTROL["warmup_steps"]).clamp(max=1)[:, None]
    target = (1 - alpha) * self._entry + alpha * self._processed_actions
    self._processed_actions = torch.clamp(
      target,
      q + (self._kd * dq - self._limit) / self._kp,
      q + (self._kd * dq + self._limit) / self._kp,
    )
    self._age += 1


@dataclass(kw_only=True)
class DeploymentPositionActionCfg(JointPositionActionCfg):
  def build(self, env):
    return DeploymentPositionAction(self, env)


def apply_deployment_contract(cfg):
  """Apply only deployment physics/actions to an independently constructed task."""
  robot = copy.deepcopy(cfg.scene.entities["robot"])
  robot.spec_fn = deployment_robot_spec
  robot.collisions = ()  # Preserve source XML contacts; don't overwrite with mjlab defaults.
  robot.articulation.actuators = tuple(
    IdealPdActuatorCfg(
      target_names_expr=(name,),
      stiffness=CONTROL["kps"][i],
      damping=CONTROL["kds"][i],
      effort_limit=CONTROL["tau_limit"][i],
      armature=None,
      frictionloss=None,
    )
    for i, name in enumerate(JOINT_NAMES)
  )
  robot.init_state.joint_pos = dict(
    zip(JOINT_NAMES, CONTROL["default_joint_pos"], strict=True)
  )
  cfg.scene.entities["robot"] = robot
  cfg.scene.spec_fn = deployment_contacts
  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.integrator = "euler"
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.decimation = 10
  cfg.actions["joint_pos"] = DeploymentPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    use_default_offset=True,
    scale=dict(zip(JOINT_NAMES, CONTROL["action_scale"], strict=True)),
  )
  cfg.observations["actor"].terms["actions"] = ObservationTermCfg(
    func=previous_clipped_action
  )
  cfg.observations["critic"].terms["actions"] = ObservationTermCfg(
    func=previous_clipped_action
  )
  return cfg
