"""Four controlled FT12k continuations: flat, matched prone, plate, guided plate."""
import argparse,hashlib,json,os,random,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch,ContactSensorCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from train_v33_reward_transfer import build_config as base_config
from train_fixed_low_ablation import atomic_json
from train_scratch_tradeoffs import tensor_hash
from train_r1_quality import verify_nested
from smp.rl.tasks.getup import prior_replay_transfer as prt, plate_transfer as p,plate_rewards as pr,v33_reward_transfer as v
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

SOURCE_SHA='8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768'

def legacy_config(arm,bank,n=4096):
    cfg,agent=base_config(bank,n);cfg.episode_length_s=20.
    cfg.scene.entities['escape_obstacle']=EntityCfg(spec_fn=p.plate_spec,init_state=EntityCfg.InitialStateCfg(pos=(20,20,.8),joint_pos={'escape_plate_slide':0.},joint_vel={'escape_plate_slide':0.}))
    cfg.sim.nconmax=256;cfg.sim.njmax=4000
    cfg.scene.sensors=tuple(cfg.scene.sensors)+(
        ContactSensorCfg(name='hand_ground_contact',primary=ContactMatch(mode='geom',pattern=r'(left|right)_hand_collision$',entity='robot'),secondary=ContactMatch(mode='geom',pattern='terrain'),fields=('found','force'),reduce='maxforce',num_slots=1,history_length=4),
        ContactSensorCfg(name='robot_obstacle_contact',primary=ContactMatch(mode='geom',pattern=COLLISION_PATTERN,entity='robot'),secondary=ContactMatch(mode='body',pattern='escape_plate',entity='escape_obstacle'),fields=('found','force','dist'),reduce='mindist',num_slots=1,history_length=4))
    cfg.events['gsi_reset'].func=p.reset;cfg.events['gsi_reset'].params['arm']=arm
    cfg.events['plate_phase']=EventTermCfg(func=p.update,mode='step')
    cfg.terminations['invalid_plate']=TerminationTermCfg(func=p.invalid)
    cfg.rewards['task_smp_product'].func=p.task;cfg.rewards['task_smp_product'].params['guided']=arm=='E3_guided'
    for name,fn,w in [('geometry_progress',pr.escape_geometry_progress,.45),('clearance',pr.escape_geometry_clearance_score,.08),('completion',pr.escape_completion,.60),('separation',pr.escape_separation_progress,.01)]:
        cfg.rewards['plate_'+name]=RewardTermCfg(func=fn,weight=w if arm=='E3_guided' else 0.)
    # Both plate arms share the contact force cost: E2/E3 isolate positive shaping/gating.
    cfg.rewards['plate_force']=RewardTermCfg(func=pr.escape_contact_force_excess_l2,weight=-.03)
    return cfg,agent

def build_config(arm,bank,n=4096):
    cfg,agent=legacy_config('E3_guided',bank,n)
    cfg.episode_length_s=10.
    cfg.events['gsi_reset'].func=prt.reset
    cfg.events['gsi_reset'].params['arm']=arm
    if arm in ('P1_v7','P2_v7_ws4'):
        cfg.events['init_smp_state'].params['ckpt_path']='datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt'
    cfg.rewards['task_smp_product'].params['ws']=4. if arm in ('P2_v7_ws4','P6_combined') else 6.
    if arm in ('P3_no_smp_term','P6_combined'):
        cfg.terminations.pop('smp_too_low',None)
    else:
        cfg.terminations['smp_too_low'].func=prt.termination
        # P2 changes reward ws only: termination ws stays 6 for an isolated contrast.
    if arm in prt.REPLAY_ARMS:
        from mjlab.managers.metrics_manager import MetricsTermCfg
        cfg.metrics['failure_replay_capture']=MetricsTermCfg(func=prt.record)
    return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
    def step(self,actions):
        env=self.unwrapped
        prephase=env._escape_phase.clone()
        obs,r,d,e=super().step(actions);logs=e.setdefault('log',{})
        for label,mask in [('flat',~env._plate_cohort),('special',env._plate_cohort)]:
            if not mask.any():continue
            for phase in (0,1,2,3,4):logs[f'Plate/{label}/phase_{phase}']=(env._escape_phase[mask]==phase).float().mean().detach()
            logs[f'Plate/{label}/raw_smp']=torch.exp(-env.cfg.rewards['task_smp_product'].params['ws']*env._smp_raw_err[mask]).mean().detach()
            logs[f'Plate/{label}/reward_product']=env._plate_product[mask].mean().detach()
            logs[f'Plate/{label}/covered_geoms']=env._escape_covered_geom_count[mask].float().mean().detach()
            logs[f'Plate/{label}/clearance_m']=env._escape_planar_clearance[mask].mean().detach()
            logs[f'Plate/{label}/max_penetration_m']=env._escape_peak_penetration[mask].max().detach()
            logs[f'Plate/{label}/max_contact_force_N']=env._escape_peak_contact_force[mask].max().detach()
            for j,name in enumerate(v.COST_NAMES):logs[f'V33Cost/{label}/{name}']=env._v_cost[mask,j].mean().detach()
        for name in env.cfg.terminations:logs['Termination/'+name]=env.termination_manager.get_term(name).float().mean().detach()
        logs['Plate/physical_fraction']=env._plate_active.float().mean()
        logs['Replay/actual_reset_fraction']=torch.tensor(env._pr_replay_total/max(1,env._pr_reset_total),device=env.device)
        logs['Replay/rejected_candidates']=torch.tensor(env._pr_rejected,device=env.device)
        for i,count in enumerate(env._pr_counts):logs[f'Replay/buffer_{i}']=torch.tensor(count,device=env.device)
        for name,mask in [('low',env._fixed_group>=2),('lying',env._v_stage==0),('crouch',env._v_stage==1),('rising',env._v_stage==2),('standing',env._v_stage==3)]:
            if mask.any():
                logs[f'Transition/{name}/task']=env._fixed_task_score[mask].mean().detach()
                logs[f'Transition/{name}/smp_multiplier']=env._fixed_smp_score[mask].mean().detach()
                logs[f'Transition/{name}/product']=env._plate_product[mask].mean().detach()
        assert torch.isfinite(r).all()
        return obs,r,d,e

class Runner(MjlabOnPolicyRunner):
    def save(self,path,infos=None):
        env=self.env.unwrapped
        state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu()}
        replay={'counts':env._pr_counts,'cursor':env._pr_cursor,'store':{k:t.cpu() for k,t in env._pr_store.items()}} if self.args.arm in prt.REPLAY_ARMS else None
        super().save(path,infos={**(infos or {}),'prior_replay_transfer':{'smp_reference':state,'arm':self.args.arm,'source_sha256':SOURCE_SHA,'common_step_counter':env.common_step_counter,'reward_ws':env.cfg.rewards['task_smp_product'].params['ws'],'replay':replay}})
        if not getattr(self,'active',False):return
        a=self.args;ep=self.current_learning_iteration
        # Save every 500; independent eval always includes no-plate transfer and fixed mass board.
        if ep==0:return
        out=a.log_dir/'validation'/str(ep);out.mkdir(parents=True,exist_ok=True)
        cmd=[os.sys.executable,'-u','scripts/evaluate_prior_replay_transfer.py','--arm',a.arm,'--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs',str(64 if a.preflight else 256)]
        if not a.preflight and (ep%1000==0 or ep==a.updates-1):cmd+=['--video']
        with open(out/'eval.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/'summary.json').read_text())
        for case,values in data.items():
            for key,val in values.items():
                if isinstance(val,(int,float)):self.logger.writer.add_scalar(f'Validation/{case}/{key}',val,ep)
        atomic_json(a.log_dir/'progress.json',{'iteration':ep,'validation':data})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=prt.ARMS,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--log-dir',type=Path,required=True);ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--num-envs',type=int,default=4096);ap.add_argument('--preflight',action='store_true');ap.add_argument('--probe-steps',type=int,default=0);a=ap.parse_args()
    a.log_dir=a.log_dir.resolve();a.log_dir.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==SOURCE_SHA
    cfg,agent=build_config(a.arm,Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),a.num_envs)
    agent.logger='tensorboard' if a.preflight else 'wandb';agent.upload_model=False;agent.save_interval=500;agent.max_iterations=a.updates;agent.run_name=a.arm
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent))
    env=None
    try:
        env=ManagerBasedRlEnv(cfg,device='cuda:0');runner=Runner(Wrapper(env,clip_actions=agent.clip_actions),asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
        parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);fresh=tensor_hash(runner.alg.save()['critic_state_dict'])
        runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);assert fresh==tensor_hash(runner.alg.save()['critic_state_dict'])
        ref=parent['infos']['scratch_tradeoffs']
        if a.arm in ('P1_v7','P2_v7_ws4'):
            ref=torch.load('outputs/v7_reference.pt',map_location='cpu',weights_only=False)
        env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
        random.seed(cfg.seed+177);np.random.seed(cfg.seed+177);torch.manual_seed(cfg.seed+177)
        obs,_=env.reset();v.init(env);env._v_start_counter=-24*500;env._v_ramp_updates=500 # existing safety costs already trained: keep full strength
        assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
        assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
        expected=.5
        assert float(env._plate_active.float().mean())==expected
        meta={'arm':a.arm,'source_checkpoint':str(a.checkpoint),'source_sha256':SOURCE_SHA,'updates':a.updates,'num_envs':a.num_envs,'episode_s':10,'plate_fraction':expected,'prepared_prone_fraction':float(env._plate_cohort.float().mean()),'source_actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'lr':runner.alg.learning_rate,'smp_reference':'FT12k f2s2 reference' if a.arm not in ('P1_v7','P2_v7_ws4') else 'V7 calibrated on frozen FT12k rollout', 'prior':cfg.events['init_smp_state'].params['ckpt_path'],'reward_ws':cfg.rewards['task_smp_product'].params['ws'],'termination_ws':cfg.terminations['smp_too_low'].params['ws'] if 'smp_too_low' in cfg.terminations else None,'all_low':a.arm in prt.LOW_ARMS,'failure_replay':a.arm in prt.REPLAY_ARMS,'actor_noise':True,'events':list(cfg.events),'terminations':list(cfg.terminations),'reward_weights':{k:t.weight for k,t in cfg.rewards.items()},'prepared_bank_sha256':hashlib.sha256(Path('datasets/reset_banks/plate_prone_v1/train.npz').read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        atomic_json(a.log_dir/'launch.json',meta)
        np.savez_compressed(a.log_dir/'initial_reset.npz',qpos=env.sim.data.qpos.cpu().numpy(),qvel=env.sim.data.qvel.cpu().numpy(),cohort=env._plate_cohort.cpu().numpy(),active=env._plate_active.cpu().numpy(),group=env._fixed_group.cpu().numpy(),source=env._fixed_source.cpu().numpy())
        runner.save(str(a.log_dir/'initial.pt'))
        if a.probe_steps:
            policy=runner.get_inference_policy();counts=torch.zeros(5,device=env.device)
            with torch.inference_mode():
                for i in range(a.probe_steps):
                    obs,_,_,_=runner.env.step(policy(obs));counts+=torch.bincount(env._escape_phase,minlength=5)
            atomic_json(a.log_dir/'probe.json',{'phase_step_counts':counts.tolist(),'qpos_finite':bool(torch.isfinite(env.sim.data.qpos).all())})
        else:
            runner.active=True;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False)
        atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
    except BaseException as exc:atomic_json(a.log_dir/'failed.json',{'error':repr(exc)});raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
