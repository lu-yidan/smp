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
from smp.rl.tasks.getup import plate_transfer as p,plate_rewards as pr,v33_reward_transfer as v
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

SOURCE_SHA='8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768'

def build_config(arm,bank,n=4096):
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

class Wrapper(RslRlVecEnvWrapper):
    def step(self,actions):
        env=self.unwrapped
        prephase=env._escape_phase.clone()
        obs,r,d,e=super().step(actions);logs=e.setdefault('log',{})
        for label,mask in [('flat',~env._plate_cohort),('special',env._plate_cohort)]:
            if not mask.any():continue
            for phase in (0,1,2,3,4):logs[f'Plate/{label}/phase_{phase}']=(env._escape_phase[mask]==phase).float().mean().detach()
            logs[f'Plate/{label}/raw_smp']=torch.exp(-6*env._smp_raw_err[mask]).mean().detach()
            logs[f'Plate/{label}/reward_product']=env._plate_product[mask].mean().detach()
            logs[f'Plate/{label}/covered_geoms']=env._escape_covered_geom_count[mask].float().mean().detach()
            logs[f'Plate/{label}/clearance_m']=env._escape_planar_clearance[mask].mean().detach()
            logs[f'Plate/{label}/max_penetration_m']=env._escape_peak_penetration[mask].max().detach()
            logs[f'Plate/{label}/max_contact_force_N']=env._escape_peak_contact_force[mask].max().detach()
            for j,name in enumerate(v.COST_NAMES):logs[f'V33Cost/{label}/{name}']=env._v_cost[mask,j].mean().detach()
        for name in env.cfg.terminations:logs['Termination/'+name]=env.termination_manager.get_term(name).float().mean().detach()
        logs['Plate/physical_fraction']=env._plate_active.float().mean()
        assert torch.isfinite(r).all()
        return obs,r,d,e

class Runner(MjlabOnPolicyRunner):
    def save(self,path,infos=None):
        env=self.env.unwrapped
        state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu()}
        super().save(path,infos={**(infos or {}),'plate_transfer':{'smp_reference':state,'arm':self.args.arm,'source_sha256':SOURCE_SHA,'common_step_counter':env.common_step_counter}})
        if not getattr(self,'active',False):return
        a=self.args;ep=self.current_learning_iteration
        # Save every 500; independent eval always includes no-plate transfer and fixed mass board.
        if ep==0:return
        out=a.log_dir/'validation'/str(ep);out.mkdir(parents=True,exist_ok=True)
        cmd=[os.sys.executable,'-u','scripts/evaluate_plate_transfer.py','--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs',str(64 if a.preflight else 256)]
        with open(out/'eval.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/'summary.json').read_text())
        for case,values in data.items():
            for key,val in values.items():
                if isinstance(val,(int,float)):self.logger.writer.add_scalar(f'Validation/{case}/{key}',val,ep)
        atomic_json(a.log_dir/'progress.json',{'iteration':ep,'validation':data})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=p.ARMS,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--log-dir',type=Path,required=True);ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--num-envs',type=int,default=4096);ap.add_argument('--preflight',action='store_true');ap.add_argument('--probe-steps',type=int,default=0);a=ap.parse_args()
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
        ref=parent['infos']['scratch_tradeoffs'];env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
        obs,_=env.reset();v.init(env);env._v_start_counter=-24*500;env._v_ramp_updates=500 # existing safety costs already trained: keep full strength
        assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
        assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
        expected=.5 if a.arm in ('E2_plate','E3_guided') else 0.
        assert float(env._plate_active.float().mean())==expected
        meta={'arm':a.arm,'source_checkpoint':str(a.checkpoint),'source_sha256':SOURCE_SHA,'updates':a.updates,'num_envs':a.num_envs,'episode_s':20,'plate_fraction':expected,'prepared_prone_fraction':float(env._plate_cohort.float().mean()),'source_actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'lr':runner.alg.learning_rate,'smp_reference':'exact FT12k saved mean/count','actor_noise':True,'events':list(cfg.events),'terminations':list(cfg.terminations),'reward_weights':{k:t.weight for k,t in cfg.rewards.items()},'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        atomic_json(a.log_dir/'launch.json',meta)
        np.savez_compressed(a.log_dir/'initial_reset.npz',qpos=env.sim.data.qpos.cpu().numpy(),qvel=env.sim.data.qvel.cpu().numpy(),cohort=env._plate_cohort.cpu().numpy(),active=env._plate_active.cpu().numpy())
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
