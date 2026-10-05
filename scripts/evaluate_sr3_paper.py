"""Evaluate historical or confirmatory SMP actors with one frozen SR3 rule.

Pass --family paper|confirmatory|clear_settle, followed by the family's normal
--eval --arm ... --checkpoint ... --num-envs 2048 --steps 1150 --out ... options.
Reward/reset configuration is reused, but reward-derived success is not used.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sr3_protocol import PROTOCOL, score, summarize, self_test


def evaluate(env, wrapper, policy, obs, a):
    from smp.rl.tasks.getup import multiterrain as mt
    from smp.rl.tasks.getup import a6_paper_ablation as ap
    assert abs(env.step_dt-.02)<1e-8 and a.steps==1150
    self_test(); ap.init_eval_loads(env)
    cpu=lambda t:t.detach().cpu().numpy()
    scene=cpu(env._mt_scene).copy(); direction=cpu(env._mt_direction).copy()
    r=env.scene['robot']
    initial=dict(qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
                 env_origins=cpu(env.scene.env_origins), scene=scene, direction=direction,
                 robot_pos_local=cpu(r.data.root_link_pos_w-env.scene.env_origins),
                 robot_quat=cpu(r.data.root_link_quat_w),joint_pos=cpu(r.data.joint_pos),
                 joint_names=np.asarray(r.joint_names))
    for name in ('mocap_pos','mocap_quat'):
        initial[name]=cpu(getattr(env.sim.data,name))
    np.savez_compressed(a.out/'sr3_initial.npz', **initial)
    (a.out/'sr3_protocol.json').write_text(json.dumps(dict(PROTOCOL,
        step_dt=env.step_dt, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).with_name('sr3_protocol.py').read_bytes()).hexdigest(),
        failure_rule='actual first-episode termination or nonfinite robot state; plate force/penetration history diagnostic only',
        termination_keys=list(env.cfg.terminations)),indent=2)+'\n')
    n=env.num_envs; dev=env.device
    first_clear=torch.full((n,),-1,device=dev,dtype=torch.long); first_stand=first_clear.clone()
    hold=torch.zeros(n,device=dev,dtype=torch.long)
    anchors=torch.zeros(n,2,2,device=dev); drift=torch.zeros(n,2,device=dev)
    samples=torch.zeros(n,2,device=dev,dtype=torch.long)
    slip=torch.zeros(n,device=dev); near_time=slip.clone()
    physical=[]; alive=[]; historical=[]; escaped=[]
    with torch.inference_mode():
        for k in range(a.steps):
            actions=policy(obs)
            finite_actions=torch.isfinite(actions).all(-1)
            env._ap_alive &= finite_actions
            actions=torch.where(env._ap_alive[:,None],actions,0.)
            obs,_,done,_=wrapper.step(actions)
            finite=torch.isfinite(r.data.root_link_pos_w).all(-1)&torch.isfinite(r.data.joint_pos).all(-1)&torch.isfinite(r.data.joint_vel).all(-1)
            env._ap_alive &= ~done.bool() & finite
            values=torch.stack((mt.height(env),-r.data.projected_gravity_b[:,2],
                               mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)),-1)
            env._ap_alive &= torch.isfinite(values).all(-1)
            live=env._ap_alive
            basic=(values[:,0]>=1.15)&(values[:,1]>=.90)&(values[:,2]>20)&live
            hold=torch.where(basic,hold+1,0)
            clear=((env._mt_scene==0)|env._mt_escaped)&live
            xy=r.data.root_link_pos_w[:,:2]
            for index,(condition,first) in enumerate(((clear,first_clear),(hold>=50,first_stand))):
                new=condition&(first<0);first[new]=k;anchors[new,index]=xy[new]
                window=live&(first>=0)&(k>first)&(k<=first+150)
                drift[:,index]=torch.maximum(drift[:,index],(xy-anchors[:,index]).norm(dim=-1)*window)
                samples[:,index]+=window.long()
            near=live&(values[:,0]>.78)&(values[:,1]>.7)
            contact=mt.ground_force(env,'quality_feet')[...,2].abs()>20
            slip+=(r.data.body_link_lin_vel_w[:,env._r_feet,:2].norm(dim=-1)*contact).sum(-1)*near*env.step_dt
            near_time+=near*env.step_dt
            physical.append(cpu(values)); alive.append(cpu(env._ap_alive).copy())
            historical.append(cpu(env._mt_invalid).copy()); escaped.append(cpu(env._mt_escaped).copy())
            if (k+1)%250==0: print('SR3_STEP',k+1,flush=True)
    physical=np.asarray(physical); alive=np.asarray(alive)
    good=(physical[...,0]>=1.15)&(physical[...,1]>=.90)&(physical[...,2]>20)
    result=score(good,alive,env.step_dt); summary=summarize(result,scene,direction)
    np.savez_compressed(a.out/'sr3_trace.npz',physical=physical,alive=alive,
                        historical_invalid=np.asarray(historical),escaped=np.asarray(escaped))
    np.savez_compressed(a.out/'sr3_per_trial.npz',scene=scene,direction=direction,
                        peaks=cpu(env._ap_peaks),stall_time=cpu(env._ap_stall_time),
                        foot_slip=cpu(slip),near_time=cpu(near_time),first_clear=cpu(first_clear),
                        first_stand=cpu(first_stand),post_clear_max_offset=cpu(drift[:,0]),
                        post_stand_max_offset=cpu(drift[:,1]),post_clear_samples=cpu(samples[:,0]),
                        post_stand_samples=cpu(samples[:,1]),**result)
    (a.out/'sr3_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SR3_COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--family',choices=('paper','confirmatory','clear_settle'),default='paper')
    known, rest=p.parse_known_args()
    assert '--eval' in rest, 'Evaluation only; never training'
    module=importlib.import_module({'paper':'train_a6_paper_ablation',
        'confirmatory':'train_a6_confirmatory','clear_settle':'train_a6_clear_settle'}[known.family])
    module.evaluate=evaluate
    sys.argv=[sys.argv[0],*rest]; module.main()

if __name__=='__main__': main()
