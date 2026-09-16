"""Local deterministic play for 93D master-family recovery checkpoints.
Run through play_recovery_local.sh to select the audited evaluation environment.
"""
import argparse,random,tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group();group.add_argument('--preset',choices=['l4','ft12k'],default=None);group.add_argument('--checkpoint',type=Path)
    p.add_argument('--pose',choices=['all','supine','prone','left','right'],default='all')
    p.add_argument('--source',choices=['procedural','natural'],default='procedural');p.add_argument('--sample',type=int,default=0)
    p.add_argument('--speed',type=float,choices=[.03125,.0625,.125,.25,.5,1.,2.],default=.25)
    p.add_argument('--paused',action='store_true');p.add_argument('--headless',action='store_true');p.add_argument('--steps',type=int,default=100)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    checkpoints=root/'outputs/recovery_local_play/checkpoints'
    ck=(a.checkpoint or checkpoints/({'l4':'L4_9999.pt','ft12k':'V33_FT_12000.pt'}[a.preset or 'ft12k'])).resolve()
    if not ck.is_file():p.error(f'Checkpoint not found: {ck}')
    if a.sample<0:p.error('sample must be nonnegative')
    bankpath=root/'datasets/reset_banks'/('procedural_low_v1' if a.source=='procedural' else 'natural_curriculum_v1')/'validation.npz'
    bank=np.load(bankpath);names={'supine':'supine','prone':'prone','left':'left_side_down','right':'right_side_down'}
    selected=list(names) if a.pose=='all' else [a.pose];indices=[]
    for name in selected:
        mask=bank['labels']==names[name]
        if 'stages' in bank:mask&=bank['stages']=='low'
        ids=np.flatnonzero(mask)
        if a.sample>=len(ids):p.error(f'{name} has {len(ids)} samples; choose 0..{len(ids)-1}')
        indices.append(int(ids[a.sample]))
    random.seed(20260910);np.random.seed(20260910);torch.manual_seed(20260910)
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
    from smp.rl.tasks.getup.pose_reset import configure_fixed_reset
    import smp.rl.tasks
    cfg=load_env_cfg('Smp-Flat-Recovery-93D-G1',play=True)
    cfg.seed=20260910;cfg.scene.num_envs=len(indices);cfg.episode_length_s=20
    cfg.events['init_smp_state'].params['ckpt_path']='datasets/pretrain_ckpt/pretrained_getup_f2s2.pt'
    cfg.observations['actor'].enable_corruption=False
    cfg.viewer.width=1280;cfg.viewer.height=960
    with tempfile.TemporaryDirectory(prefix='recovery_play_') as tmp:
        posefile=Path(tmp)/'poses.npz';np.savez(posefile,qpos=bank['qpos'][indices],labels=bank['labels'][indices])
        configure_fixed_reset(cfg,str(posefile));env=ManagerBasedRlEnv(cfg,device='cuda:0')
        try:
            agent=unitree_g1_smp_ppo_runner_cfg();agent.logger='tensorboard';agent.upload_model=False
            w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
            runner=MjlabOnPolicyRunner(w,asdict(agent),None,'cuda:0')
            runner.load(str(ck),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
            obs,_=env.reset();assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
            policy=runner.get_inference_policy(device='cuda:0')
            print(f'CHECKPOINT: {ck}\nPOSES: {selected}, source={a.source}, bank rows={indices}\nPlayback speed={a.speed}x; physics dt={env.physics_dt}, control dt={env.step_dt}; speed changes wall-clock pacing only.\nEnter=repeat same reset; Space=pause; Right=one control step while paused; -/= slower/faster; ,/. select environment; A=show all environments.\nNominal evaluation: no automatic push/DR or actor noise. SMP shown by this evaluation environment is NOT the training SMP score.',flush=True)
            if a.headless:
                with torch.inference_mode():
                    for i in range(a.steps):
                        obs,r,d,e=w.step(policy(obs));assert torch.isfinite(r).all()
                print('PLAY_SMOKE_PASS',ck.name,'steps',a.steps,'actor_dim',obs['actor'].shape[-1],flush=True)
            else:
                from mjlab.viewer import NativeMujocoViewer
                viewer=NativeMujocoViewer(w,policy)
                while viewer._time_multiplier>a.speed:viewer.decrease_speed()
                while viewer._time_multiplier<a.speed:viewer.increase_speed()
                if a.paused:viewer.pause()
                viewer.run()
        finally:env.close()
if __name__=='__main__':main()
