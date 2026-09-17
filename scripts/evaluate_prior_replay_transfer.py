"""Fixed 20s tests: flat four directions, matched prone, easy/hard plate.
No training resets/SMP early termination; failure and success stay separate.
"""
import argparse,gc,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_prior_replay_transfer import build_config
from train_fixed_low_ablation import atomic_json
from smp.rl.tasks.getup import plate_transfer as p,plate_geometry as g,v33_reward_transfer as v
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=256);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--video',action='store_true');ap.add_argument('--cases',nargs='+',default=['flat','prone','plate_easy','plate_hard']);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);summary={}
    for case in a.cases:
        cfg,agent=build_config(a.arm,Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),a.num_envs)
        cfg.seed=73461;random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        cfg.observations['actor'].enable_corruption=False
        for key in list(cfg.events):
            if cfg.events[key].mode in ('startup','interval') and key!='init_smp_state':cfg.events.pop(key)
        for key in ('smp_too_low','stood_up','invalid_plate'):cfg.terminations.pop(key,None)
        cfg.episode_length_s=1000.
        cfg.viewer.width=640;cfg.viewer.height=480
        env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);env._pr_disable_replay=True;writer=None
        try:
            wrapper=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(wrapper,asdict(agent),None,'cuda:0')
            runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});policy=runner.get_inference_policy()
            saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
            ref=saved['infos']['prior_replay_transfer']['smp_reference']
            env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
            del saved
            random.seed(cfg.seed+177);np.random.seed(cfg.seed+177);torch.manual_seed(cfg.seed+177)
            obs,_=env.reset();ids=torch.arange(env.num_envs,device=env.device);r=env.scene['robot']
            if case=='flat':
                bank=np.load('datasets/reset_banks/procedural_low_v1/validation.npz');rows=[]
                for label in ('supine','prone','left_side_down','right_side_down'):
                    candidates=np.flatnonzero(bank['labels']==label);rows.extend(candidates[:a.num_envs//4])
                assert len(rows)==a.num_envs
                q=torch.as_tensor(bank['qpos'][rows],device=env.device);root=r.data.default_root_state.clone();root[:,:3]=q[:,:3]+env.scene.env_origins;root[:,3:7]=q[:,3:7];root[:,7:]=0
                r.write_root_state_to_sim(root,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids);env.sim.forward()
                env._plate_cohort[:]=False;env._plate_active[:]=False
                g.reset_guided_escape_plate(env,ids,active_mask=env._plate_active,prepare_mask=env._plate_cohort,inactive_xy=(20,20),surface_gap=None)
            else:
                env._plate_cohort[:]=True;env._plate_active[:]=case.startswith('plate')
                env._plate_bank=torch.as_tensor(np.load('datasets/reset_banks/plate_prone_v1/validation.npz')['qpos'],device=env.device)
                env._plate_rng.manual_seed(986173)
                p.reset(env,ids,arm='E3_guided',**{k:val for k,val in cfg.events['gsi_reset'].params.items() if k!='arm'})
                # Nominal paired poses use same RNG in prone and plate cases.
                # Freeze plate difficulty rather than evaluating a changing training curriculum.
                hard=case=='plate_hard';g.reset_guided_escape_plate_curriculum(env,ids,plate_mass_range=(8.,8.) if hard else (4.,6.),initial_max_mass=8. if hard else 6.,mass_curriculum_steps=1,
                    active_mask=env._plate_active,prepare_mask=env._plate_cohort,crawl_ready_prone=False,align_to_body=True,longitudinal_offset=-.10,
                    longitudinal_offset_curriculum=(0.,0.) if hard else (.18,.18),lateral_offset_curriculum=(0.,0.) if hard else (.22,.22),overlap_curriculum_steps=1,
                    xy_offset_range=.005,surface_gap=.001,collision_geom_pattern=COLLISION_PATTERN,inactive_xy=(20,20))
            env.sim.forward();prime_static_history(env,ids);v.reset(env,ids)
            env.action_manager.reset(ids)  # warmup entry must match the manually installed evaluation pose
            env.observation_manager.reset(ids);obs=env.observation_manager.compute(update_history=True)
            # Verify initial collision setup in CPU model, not just floating-base height.
            cpu=mujoco.MjData(env.sim.mj_model);qinit=env.sim.data.qpos.cpu().numpy();mpos=env.sim.data.mocap_pos.cpu().numpy();mquat=env.sim.data.mocap_quat.cpu().numpy();depths=[];worst=[]
            for j in range(env.num_envs):
                cpu.qpos[:]=qinit[j];cpu.mocap_pos[:]=mpos[j];cpu.mocap_quat[:]=mquat[j];mujoco.mj_forward(env.sim.mj_model,cpu)
                depth=0.;pair=None
                for contact in cpu.contact:
                    if contact.dist<depth:depth=float(contact.dist);pair=[env.sim.mj_model.geom(int(k)).name for k in contact.geom]
                depths.append(depth);worst.append(pair)
            atomic_json(a.out/f'{case}_reset.json',{'min_dist':depths,'worst_pairs':worst})
            assert min(depths)>=-.0011, (case,'invalid initial penetration',min(depths))
            np.savez_compressed(a.out/f'{case}_initial.npz',qpos=qinit,mocap_pos=mpos,mocap_quat=mquat)
            hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();escaped=torch.zeros_like(hold,dtype=torch.bool);alive=torch.ones_like(escaped);peak=torch.zeros(env.num_envs,4,device=env.device);trace=[]
            if a.video:
                import imageio.v2 as imageio
                writer=imageio.get_writer(str(a.out/f'{case}_20s.mp4'),fps=25,codec='libx264',quality=7)
            selected=[0,env.num_envs//4,env.num_envs//2,3*env.num_envs//4]
            initial_z=r.data.site_pos_w[:,env._r_head,2].clone();initial_u=(-r.data.projected_gravity_b[:,2]).clone()
            first_progress=torch.full((env.num_envs,),-1.,device=env.device);first_escape=first_progress.clone();first_stable=first_progress.clone();low_still=torch.zeros_like(first_progress);reward_trace=[]
            for step in range(a.steps):
                with torch.inference_mode():obs,_,done,_=wrapper.step(policy(obs))
                alive &= ~done.bool();stable=env._r_stable & alive
                if case.startswith('plate'):stable &= (env._escape_phase==3)
                hold=torch.where(stable,hold+env.step_dt,0.);best=torch.maximum(best,hold);escaped|=(env._escape_phase==3)&alive
                peak=torch.maximum(peak,torch.where(alive[:,None],env._r_peaks,0.))
                z=r.data.site_pos_w[:,env._r_head,2];upright=-r.data.projected_gravity_b[:,2]
                moved=((z-initial_z)>.08)|((upright-initial_u)>.15)
                first_progress=torch.where((first_progress<0)&moved&alive,(step+1)*env.step_dt,first_progress)
                first_escape=torch.where((first_escape<0)&escaped,(step+1)*env.step_dt,first_escape)
                first_stable=torch.where((first_stable<0)&(hold>=1.-1e-4),(step+1)*env.step_dt,first_stable)
                low_still+=((z<.65)&(r.data.joint_vel.square().mean(-1).sqrt()<.2)&alive)*env.step_dt
                reward_trace.append(torch.stack([env._fixed_task_score,env._fixed_smp_score,env._plate_product,env._v_stage.float(),z,upright,env._v_cost.mul(env._v_cost.new_tensor(v.COST_WEIGHTS).abs()).sum(-1)],-1)[selected].cpu().numpy())
                trace.append(env.sim.data.qpos[selected].cpu().numpy())
                if writer is not None and step%2==0:
                    from PIL import Image,ImageDraw
                    tiles=[]
                    for j in selected:
                        cfg.viewer.env_idx=j;im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,42),fill='black');draw.text((8,8),f'{case} env {j} | t={(step+1)*env.step_dt:.2f}s | hold {float(hold[j]):.2f}s | phase {int(env._escape_phase[j])}',fill='white');tiles.append(np.asarray(im))
                    writer.append_data(np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0))
            metrics={'protocol_version':2,'stable_10s':float((best>=10.-1e-4).float().mean()),'stable_1s':float((best>=1.-1e-4).float().mean()),'escaped_fraction':float(escaped.float().mean()),'invalid_plate_fraction':float((env._escape_phase==4).float().mean()),'simulation_failure_fraction':float((~alive).float().mean()),'n':env.num_envs,'initial_penetration_max_m':float(-min(depths))}
            for j,name in enumerate(['tau','speed','power','head_force']):metrics[name+'_env_peak_p95']=float(np.percentile(peak[:,j].cpu().numpy(),95))
            if case=='flat':
                for j,label in enumerate(('supine','prone','left','right')):metrics[label+'_stable_10s']=float((best[j*a.num_envs//4:(j+1)*a.num_envs//4]>=10.-1e-4).float().mean())
            atomic_json(a.out/f'{case}_transitions.json',{'first_progress_s':first_progress.tolist(),'first_escape_s':first_escape.tolist(),'first_stable_1s_s':first_stable.tolist(),'low_stationary_s':low_still.tolist(),'note':'-1 means not reached; first_progress = head +8cm or upright +0.15; nominal 20s rollout'})
            np.savez_compressed(a.out/f'{case}_reward_trace.npz',values=np.asarray(reward_trace),selected=selected,columns=['task','smp_multiplier','gated_product','stage','head_z','upright','v33_cost'])
            for label,arr in [('first_progress',first_progress),('first_stable_1s',first_stable)]:
                vals=arr[arr>=0]
                if len(vals):metrics[label+'_conditional_median_s']=float(vals.median())
            metrics['low_stationary_mean_s']=float(low_still.mean())
            atomic_json(a.out/f'{case}_episodes.json',{'hold_max_s':best.tolist(),'escaped':escaped.tolist(),'alive':alive.tolist(),'peaks':peak.tolist()});np.savez_compressed(a.out/f'{case}_trace.npz',qpos=np.asarray(trace),selected=selected,mocap_pos=mpos[selected],mocap_quat=mquat[selected]);summary[case]=metrics;atomic_json(a.out/'summary.json',summary)
        finally:
            if writer is not None:writer.close()
            env.close();del runner,wrapper,env;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
