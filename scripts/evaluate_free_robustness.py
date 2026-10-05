"""Frozen-checkpoint free-plate stress evaluation with paired initial states."""
import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from smp.rl.tasks.getup import multiterrain as mt, v33_path_ablation as pa, a6_alpha_convergence as ac
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
import evaluate_sr3_paper as sr3

CASES = {
    'nominal': dict(scale=1.,mass=6.),
    'size125': dict(scale=1.25,mass=6.),
    'size150': dict(scale=1.5,mass=6.),
    'size175': dict(scale=1.75,mass=6.),
    'mass9': dict(scale=1.,mass=9.),
    'mass12': dict(scale=1.,mass=12.),
    'mass18': dict(scale=1.,mass=18.),
    'mixed': dict(scale_xy=[1.,1.75],mass_range=[6.,18.],offset_xy_m=[-.15,.15],yaw_deg=[-45.,45.]),
}


def cuboid_inertia(half_size, mass):
    sq=half_size.square()
    return mass[:,None]/3.*torch.stack((sq[:,1]+sq[:,2],sq[:,0]+sq[:,2],sq[:,0]+sq[:,1]),-1)


@requires_model_fields('body_mass','body_inertia','geom_size','geom_aabb','geom_rbound',
                       recompute=RecomputeLevel.set_const)
def reset(env, env_ids=None, case_name='nominal', base_reset='path', **kwargs):
    (ac.reset if base_reset=='alpha' else pa.reset)(env,env_ids,**kwargs)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    ei=ids[env._mt_scene[ids]==2]
    if not len(ei):return
    if not hasattr(env,'_free_test_params'):
        # Independent RNG: perturbations never consume the reset-bank RNG.
        rng=np.random.default_rng(env.cfg.seed+20261005)
        n=env.num_envs;c=CASES[case_name]
        params=np.zeros((n,6),np.float32)  # sx,sy,mass,dx,dy,yaw
        params[:,:2]=c.get('scale',1.);params[:,2]=c.get('mass',6.)
        if case_name=='mixed':
            params[:,:2]=rng.uniform(*c['scale_xy'],size=(n,2))
            params[:,2]=rng.uniform(*c['mass_range'],size=n)
            params[:,3:5]=rng.uniform(*c['offset_xy_m'],size=(n,2))
            params[:,5]=np.deg2rad(rng.uniform(*c['yaw_deg'],size=n))
        env._free_test_params=torch.tensor(params,device=env.device)
    if case_name=='nominal':return  # Preserve historical baseline bit-for-bit at reset.
    par=env._free_test_params[ei];free=env.scene['free_obstacle']
    gid=env._mt_plate_ids[1];bid=free.indexing.body_ids[-1].long()
    size=par.new_tensor([.45,.32,.035]).expand(len(ei),3).clone();size[:,:2]*=par[:,:2]
    env.sim.model.geom_size[ei,gid]=size
    env.sim.model.geom_aabb[ei,gid,0]=0.;env.sim.model.geom_aabb[ei,gid,1]=size
    env.sim.model.geom_rbound[ei,gid]=size.norm(dim=-1)
    env.sim.model.body_mass[ei,bid]=par[:,2]
    env.sim.model.body_inertia[ei,bid]=cuboid_inertia(size,par[:,2])

    # Conservative collision-free placement: the lowest board face starts 2 mm
    # above every robot collision bound intersecting its rotated footprint.
    pos,ext=mt.robot_bounds(env);r=env.scene['robot']
    board=free.data.default_root_state[ei].clone();board[:,7:]=0.
    board[:,:2]=r.data.root_link_pos_w[ei,:2]+par[:,3:5]
    angle=par[:,5];c=angle.cos();s=angle.sin()
    delta=pos[ei,:,:2]-board[:,None,:2]
    x=c[:,None]*delta[:,:,0]+s[:,None]*delta[:,:,1]
    y=-s[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1]
    ex=c.abs()[:,None]*ext[ei,:,0]+s.abs()[:,None]*ext[ei,:,1]
    ey=s.abs()[:,None]*ext[ei,:,0]+c.abs()[:,None]*ext[ei,:,1]
    covered=(x.abs()<ex+size[:,None,0])&(y.abs()<ey+size[:,None,1])
    assert covered.any(-1).all(), 'Perturbation placed a board away from all robot geometry'
    top=torch.where(covered,pos[ei,:,2]+ext[ei,:,2],-torch.inf).amax(-1)
    board[:,2]=top+size[:,2]+.002
    board[:,3:7]=0.;board[:,3]=(angle/2).cos();board[:,6]=(angle/2).sin()
    free.write_root_state_to_sim(board,env_ids=ei)
    env.sim.forward()
    # Refresh reward histories to avoid fictitious progress from the reset edit.
    score,_=mt.plate_geometry(env)
    env._mt_best[ei]=score[ei];env._mt_initial_overlap[ei]=(-score[ei]).clamp_min(.01)
    count,score,clearance=pa.geometry(env)
    env._pa_best_score[ei]=score[ei];env._pa_best_clearance[ei]=clearance[ei]
    env._pa_initial_count[ei]=count[ei].float()
    from smp.rl.tasks.getup.r2_ablation import planar_score
    env._ra_initial[ei]=planar_score(env)[ei]
    prime_static_history(env,ei)


def audit(env,a,case):
    cpu=lambda x:x.detach().cpu().numpy()
    ids=torch.where(env._mt_scene==2)[0];gid=int(env._mt_plate_ids[1])
    bid=int(env.scene['free_obstacle'].indexing.body_ids[-1])
    size=env.sim.model.geom_size[ids,gid];mass=env.sim.model.body_mass[ids,bid]
    inertia=env.sim.model.body_inertia[ids,bid]
    assert torch.allclose(inertia,cuboid_inertia(size,mass),rtol=2e-5,atol=1e-6)
    assert torch.allclose(env.sim.model.geom_rbound[ids,gid],size.norm(dim=-1),atol=1e-6)
    assert torch.allclose(env.sim.model.geom_aabb[ids,gid,1],size,atol=1e-6)
    assert torch.equal(env.sim.model.geom_aabb[ids,gid,0],torch.zeros_like(size))
    q=cpu(env.sim.data.qpos);v=cpu(env.sim.data.qvel)
    mp=cpu(env.sim.data.mocap_pos);mq=cpu(env.sim.data.mocap_quat)
    sizes=cpu(size);masses=cpu(mass);inertias=cpu(inertia)
    model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model)
    minimum=[];board_minimum=[]
    for k,j in enumerate(cpu(ids)):
        model.geom_size[gid]=sizes[k]
        model.geom_aabb.reshape(-1,2,3)[gid,0]=0;model.geom_aabb.reshape(-1,2,3)[gid,1]=sizes[k]
        model.geom_rbound[gid]=np.linalg.norm(sizes[k]);model.body_mass[bid]=masses[k]
        model.body_inertia[bid]=inertias[k]
        data.qpos[:]=q[j];data.qvel[:]=v[j];data.mocap_pos[:]=mp[j];data.mocap_quat[:]=mq[j]
        mujoco.mj_forward(model,data)
        minimum.append(min([0.]+[float(c.dist) for c in data.contact]))
        board_minimum.append(min([0.]+[float(c.dist) for c in data.contact if gid in (c.geom1,c.geom2)]))
    assert min(minimum)>-.0011,(case,min(minimum))
    assert min(board_minimum)>-1e-5,(case,min(board_minimum))
    for key in ('body_mass','body_inertia'):
        b=env._bd_bodies
        assert torch.allclose(getattr(env.sim.model,key)[:,b],env.sim.get_default_field(key)[b][None].expand_as(getattr(env.sim.model,key)[:,b]))
    np.savez_compressed(a.out/'free_initial_parameters.npz',env_ids=cpu(ids),
        source=cpu(env._mt_source[ids]),direction=cpu(env._mt_direction[ids]),
        half_size=sizes,mass=masses,inertia=inertias,parameters=cpu(env._free_test_params[ids]),
        robot_pos=cpu(env.scene['robot'].data.root_link_pos_w),
        robot_quat=cpu(env.scene['robot'].data.root_link_quat_w),
        robot_joint_pos=cpu(env.scene['robot'].data.joint_pos),
        qpos=q,qvel=v,mocap_pos=mp,mocap_quat=mq,board_min_distance=np.asarray(board_minimum))
    result=dict(case=case,settings=CASES[case],free_n=len(ids),
        initial_qpos_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        parameters_sha256=hashlib.sha256(cpu(env._free_test_params[ids]).tobytes()).hexdigest(),
        robot_joint_pos_sha256=hashlib.sha256(cpu(env.scene['robot'].data.joint_pos).tobytes()).hexdigest(),
        min_initial_contact_distance=min(minimum),min_initial_board_distance=min(board_minimum),
        mass_min_max=[float(mass.min()),float(mass.max())],
        full_dimensions_min=(2*size.amin(0)).tolist(),full_dimensions_max=(2*size.amax(0)).tolist(),
        pass_inertia_bounds_and_no_initial_penetration=True)
    (a.out/'free_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FREE_AUDIT_PASS',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--family',choices=('paper','confirmatory'),required=True)
    parser.add_argument('--smoke',action='store_true')
    args,rest=parser.parse_known_args()
    assert '--eval' in rest and '--video' not in rest
    module=importlib.import_module('train_a6_paper_ablation' if args.family=='paper' else 'train_a6_confirmatory')
    build=module.build_config
    def build_config(*a,**kw):
        cfg,agent=build(*a,**kw);event=cfg.events['gsi_reset']
        assert event.func in (pa.reset,ac.reset)
        event.params['base_reset']='alpha' if event.func is ac.reset else 'path'
        event.func=reset;event.params['case_name']=args.case
        return cfg,agent
    module.build_config=build_config
    def evaluate(env,wrapper,policy,obs,a):
        audit(env,a,args.case)
        if args.smoke:
            with torch.inference_mode():
                for _ in range(25):
                    obs,_,done,_=wrapper.step(policy(obs))
                    assert not done.any() and torch.isfinite(env.sim.data.qpos).all()
            (a.out/'smoke_pass.json').write_text(json.dumps(dict(case=args.case,physics_s=.5,passed=True,
                evaluator_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))+'\n')
            print('SMOKE_PASS',flush=True)
        else:sr3.evaluate(env,wrapper,policy,obs,a)
    module.evaluate=evaluate;sys.argv=[sys.argv[0],*rest];module.main()

if __name__=='__main__':main()
