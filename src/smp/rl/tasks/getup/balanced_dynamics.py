"""Four-direction plate recovery; paired reset-only / robot-dynamics study."""
from dataclasses import dataclass
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import multiterrain as mt
from smp.rl.tasks.getup.master_deployment_contract import DeploymentPositionAction,DeploymentPositionActionCfg
from smp.rl.tasks.getup.multiterrain_geometry import scene_spec as original_scene

def scene_spec(spec):
    original_scene(spec)
    spec.body('surfaces').pos=(100,100,0) # retained sensor target; no nearby terrain

class DelayedPositionAction(DeploymentPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self._bd_env=env
        self._bd_buffer=self._processed_actions.unsqueeze(0).repeat(6,1,1)
        self._bd_cursor=0;self._bd_rows=torch.arange(self.num_envs,device=self.device)
    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        if hasattr(self,'_bd_buffer'):self._bd_buffer[:,ids]=self._entry[ids].unsqueeze(0)
    def apply_actions(self):
        lag=getattr(self._bd_env,'_bd_lag',torch.zeros(self.num_envs,dtype=torch.long,device=self.device))
        self._bd_buffer[self._bd_cursor]=self._processed_actions
        target=self._bd_buffer[(self._bd_cursor-lag)%6,self._bd_rows]
        self._bd_cursor=(self._bd_cursor+1)%6
        bias=self._entity.data.encoder_bias[:,self._target_ids]
        self._entity.set_joint_position_target(target-bias,joint_ids=self._target_ids)
@dataclass(kw_only=True)
class DelayedPositionActionCfg(DeploymentPositionActionCfg):
    def build(self,env):return DelayedPositionAction(self,env)

def initialize(env,bank_path,multiterrain=False):
    mt._init(env,bank_path)
    if hasattr(env,'_bd_rng'):return
    n=env.num_envs;dev=env.device;assert n%32==0
    if not multiterrain:
        env._mt_scene=torch.arange(n,device=dev)//(n//2)
        env._mt_stratum=env._mt_scene.clone()
        env._mt_direction=(torch.arange(n,device=dev)%(n//2))//(n//8)
        env._mt_source=((torch.arange(n,device=dev)%(n//8))<(n//32)).long()
        env._fixed_group=env._mt_direction+2;env._fixed_source=env._mt_source
    env._bd_flat_support=not multiterrain
    env._bd_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+131071)
    env._bd_lag=torch.zeros(n,dtype=torch.long,device=dev)
    env._bd_mass=torch.ones((n,4),device=dev);env._bd_gain=torch.ones((n,6),device=dev)
    env._bd_nominal=torch.ones(n,dtype=torch.bool,device=dev)
    r=env.scene['robot'];model=env.sim.mj_model
    names=[model.body(int(b)).name.split('/')[-1] for b in r.indexing.body_ids]
    # Coherent lower limb / upper subtree scaling, symmetric legs.
    def descendants(name):
        root=model.body('robot/'+name).id;out=[]
        for i,b in enumerate(r.indexing.body_ids.tolist()):
            j=b
            while j>0 and j!=root:j=int(model.body_parentid[j])
            if j==root:out.append(i)
        return out
    legs=set(descendants('left_hip_pitch_link')+descendants('right_hip_pitch_link'))
    upper=set(descendants('torso_link'))
    env._bd_groups=torch.tensor([0 if i in legs else 1 if i in upper else 2 for i in range(len(names))],device=dev)
    env._bd_bodies=r.indexing.body_ids.long()
    env._bd_act_groups=[]
    for act in r.actuators:
        name=act._target_names[0]
        g=0 if 'waist' in name else 1 if 'hip' in name else 2 if 'knee' in name else 3 if 'ankle' in name else 4 if ('shoulder' in name or 'elbow' in name) else 5
        env._bd_act_groups.append(g)

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,bank_path='outputs/multiterrain_bank/train.npz',dynamics=False,stress_upper=1.,evaluation=False,multiterrain=False):
    initialize(env,bank_path,multiterrain)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    n=len(ids);dev=env.device
    # Width grows from +/-10% to +/-20% over the first 2000 PPO updates.
    width=.1+.1*min(env.common_step_counter/(24*2000),1.)
    rand=lambda shape:torch.rand(shape,generator=env._bd_rng,device=dev)
    enabled=(rand((n,))>=.25) if dynamics else torch.zeros(n,dtype=torch.bool,device=dev)
    factors=1+(2*rand((n,4))-1)*width if dynamics else torch.ones((n,4),device=dev)
    gains=1+(2*rand((n,6))-1)*width if dynamics else torch.ones((n,6),device=dev)
    factors[~enabled]=1;gains[~enabled]=1
    # Column 3 reserved in audit; first three scale legs, upper, pelvis/waist bodies.
    factors[:,3]=1
    if stress_upper!=1.:factors[:,1]=stress_upper
    env._bd_mass[ids]=factors;env._bd_gain[ids]=gains;env._bd_nominal[ids]=~enabled
    lag=torch.randint(0,6,(n,),generator=env._bd_rng,device=dev) if dynamics else torch.zeros(n,dtype=torch.long,device=dev)
    env._bd_lag[ids]=torch.where(enabled,lag,0)
    bodies=env._bd_bodies;scale=factors[:,env._bd_groups]
    for field in ('body_mass','body_inertia'):
        base=env.sim.get_default_field(field)[bodies]
        values=base[None]* (scale if field=='body_mass' else scale[:,:,None])
        getattr(env.sim.model,field)[ids[:,None],bodies[None,:]]=values
    for act,g in zip(env.scene['robot'].actuators,env._bd_act_groups):
        act.set_gains(ids,kp=act.default_stiffness[ids]*gains[:,g,None],kd=act.default_damping[ids]*gains[:,g,None])
    mt.reset(env,ids,bank_path=bank_path)
    if evaluation:
        for scene,name in [(1,'escape_obstacle'),(2,'free_obstacle')]:
            e=env.scene[name];bid=e.indexing.body_ids[-1].long();ei=ids[env._mt_scene[ids]==scene]
            env.sim.model.body_mass[ei,bid]=6.
            env.sim.model.body_inertia[ei,bid]=env.sim.get_default_field('body_inertia')[bid]*6./env.sim.get_default_field('body_mass')[bid]


def audit_dynamics(env):
    ids=env._bd_bodies
    expected=env.sim.get_default_field('body_mass')[ids][None]*env._bd_mass[:,env._bd_groups]
    assert torch.allclose(env.sim.model.body_mass[:,ids],expected)
    expected_i=env.sim.get_default_field('body_inertia')[ids][None]*env._bd_mass[:,env._bd_groups,None]
    assert torch.allclose(env.sim.model.body_inertia[:,ids],expected_i)
    for act,g in zip(env.scene['robot'].actuators,env._bd_act_groups):
        assert torch.allclose(act.stiffness,act.default_stiffness*env._bd_gain[:,g,None])
        assert torch.allclose(act.damping,act.default_damping*env._bd_gain[:,g,None])
        assert torch.equal(act.force_limit,act.default_force_limit)
    # Test actual entity target at zero and five physics-step lags, no sim steps.
    a=env.action_manager.get_term('joint_pos');saved_lag=env._bd_lag.clone()
    env._bd_lag[:]=5;env._bd_lag[0]=0
    a._bd_buffer.zero_();a._processed_actions.fill_(.123)
    for step in range(6):
        a.apply_actions();target=env.scene['robot'].data.joint_pos_target[:,a._target_ids]+env.scene['robot'].data.encoder_bias[:,a._target_ids]
        assert torch.allclose(target[0],torch.full_like(target[0],.123),atol=1e-6)
        expected=.123 if step==5 else 0.
        assert torch.allclose(target[1:],torch.full_like(target[1:],expected),atol=1e-6)
    env._bd_lag[:]=saved_lag
    return {'model_mass_inertia_verified':True,'actuator_gain_verified':True,'effort_caps_unchanged':True,'delay_0_and_10ms_verified':True}
